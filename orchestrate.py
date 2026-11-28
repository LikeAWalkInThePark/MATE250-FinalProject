#!/usr/bin/env python3
"""Generate and optionally submit the silicon HSE band-gap calculations.

The script expects this layout relative to itself::

    ReferenceFiles/
      hse_cal/si.scf.in
      hse_cal/submit.sh
      Si_ONCV_PBE-1.2.upf

By default it only generates files.  Use ``--submit`` to call ``sbatch`` for
each generated calculation after inspecting the generated inputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path


# Cutoffs are intentionally easy to change.  The production value can be
# selected after checking that the band gap is converged with respect to ecut.
DEFAULT_ECUTS = (30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 70.0, 80.0)
STRAINS = (0.0, 1.0, 2.0, -1.0, -2.0)
NQX_VALUES = (3, 4)
STAGE1_ENERGY_TOL_RY = 1.0e-4
STAGE2_ENERGY_TOL_RY = 1.0e-4
STAGE2_KPOINTS = (6, 8, 10, 12, 14)
STATE_FILENAME = ".orchestrate_state.json"

LOGGER = logging.getLogger("orchestrate")


def replace_assignment(text: str, name: str, value: str) -> str:
    """Replace a namelist assignment while preserving commas and comments."""
    pattern = re.compile(rf"^(\s*{re.escape(name)}\s*=)(.*)$",
                         flags=re.IGNORECASE | re.MULTILINE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one '{name}' assignment; found {len(matches)}")

    match = matches[0]
    remainder = match.group(2)
    comment_index = remainder.find("!")
    if comment_index >= 0:
        assignment_part = remainder[:comment_index]
        comment = remainder[comment_index:]
    else:
        assignment_part = remainder
        comment = ""

    comma = "," if assignment_part.rstrip().endswith(",") else ""
    replacement = f"{match.group(1)} {value}{comma}{comment}"
    return text[:match.start()] + replacement + text[match.end():]


def ensure_assignment(text: str, name: str, value: str, namelist: str = "SYSTEM") -> str:
    """Replace an assignment or add it to a namelist when absent."""
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*=", re.IGNORECASE | re.MULTILINE)
    if pattern.search(text):
        return replace_assignment(text, name, value)

    block = re.search(
        rf"(&{re.escape(namelist)}\b.*?)(^/\s*$)",
        text,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if not block:
        raise ValueError(f"Could not find &{namelist} namelist")
    insertion = f"  {name:<14} = {value},\n"
    return text[:block.start(2)] + insertion + text[block.start(2):]


def replace_nqx(text: str, nqx: int) -> str:
    pattern = r"^(\s*nqx1\s*=\s*)\d+(\s*,?.*)$"
    text, count = re.subn(pattern, rf"\g<1>{nqx}\g<2>", text,
                          flags=re.MULTILINE | re.IGNORECASE)
    if count != 1:
        raise ValueError(f"Expected exactly one 'nqx1' assignment; found {count}")
    for axis in ("nqx2", "nqx3"):
        text = replace_assignment(text, axis, str(nqx))
    return text


def replace_lattice(text: str, scale: float) -> str:
    """Scale the three primitive lattice vectors by the requested strain."""
    marker = re.search(
        r"(CELL_PARAMETERS\s*\{[^}]+\})(.*?)(?=\n\s*[A-Z_]+\s*(?:\{|=)|\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not marker:
        raise ValueError("Could not find CELL_PARAMETERS block")

    block = marker.group(2)
    rows = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        if len(fields) != 3:
            break
        try:
            rows.append([float(field) * scale for field in fields])
        except ValueError:
            break
    if len(rows) != 3:
        raise ValueError("CELL_PARAMETERS must contain exactly three numeric rows")

    replacement = "\n" + "\n".join(
        f"  {row[0]:.9f}   {row[1]:.9f}   {row[2]:.9f}" for row in rows
    ) + "\n"
    return text[: marker.start(2)] + replacement + text[marker.end(2) :]


def replace_kpoints(text: str, kpoint: int) -> str:
    """Replace an automatic cubic k-point grid while preserving offsets."""
    pattern = r"(^\s*K_POINTS\s+automatic\s*\n\s*)\d+\s+\d+\s+\d+(\s+.*)$"
    updated, count = re.subn(
        pattern,
        rf"\g<1>{kpoint} {kpoint} {kpoint}\g<2>",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if count != 1:
        raise ValueError(f"Expected exactly one automatic K_POINTS grid; found {count}")
    return updated


def make_input(
    template: str,
    *,
    nqx: int,
    ecut: float,
    strain_percent: float,
    kpoint: int | None = None,
    pseudo_dir: str = "'../../../../ReferenceFiles/'",
) -> str:
    scale = 1.0 + strain_percent / 100.0
    text = template
    # Each run is four directory levels below MATE250-FinalProject:
    # generated_runs/stage/nqx/ecut.  ReferenceFiles is at the project root.
    text = replace_assignment(text, "pseudo_dir", pseudo_dir)
    text = replace_assignment(text, "ecutwfc", f"{ecut:g}")
    text = replace_assignment(text, "ecutrho", f"{4.0 * ecut:g}")
    text = replace_nqx(text, nqx)
    if kpoint is not None:
        text = replace_kpoints(text, kpoint)
    # Keep fractional crystal coordinates unchanged.  Scaling the lattice
    # vectors applies the homogeneous strain to the physical positions.
    return replace_lattice(text, scale)


def stage_name(strain_percent: float) -> str:
    stage_numbers = {0.0: "1.0", 1.0: "1.1", 2.0: "1.2", -1.0: "1.3", -2.0: "1.4"}
    if strain_percent not in stage_numbers:
        raise ValueError(f"Unsupported strain percentage: {strain_percent}")
    if strain_percent == 0:
        label = "0pct"
    elif strain_percent > 0:
        label = f"plus{int(strain_percent)}pct"
    else:
        label = f"minus{abs(int(strain_percent))}pct"
    return f"stage_{stage_numbers[strain_percent]}_{label}"


def find_project_root(requested_root: Path) -> Path:
    """Support running from either FinalProject or MATE250-FinalProject."""
    requested_root = requested_root.resolve()
    if (requested_root / "ReferenceFiles").is_dir():
        return requested_root
    nested_root = requested_root / "MATE250-FinalProject"
    if (nested_root / "ReferenceFiles").is_dir():
        return nested_root
    return requested_root


def state_path(root: Path) -> Path:
    return root / STATE_FILENAME


def save_state(root: Path, state: dict) -> None:
    state_path(root).write_text(json.dumps(state, indent=2) + "\n")


def load_state(root: Path) -> dict:
    path = state_path(root)
    if not path.is_file():
        raise FileNotFoundError(
            f"No {STATE_FILENAME} found. Run Stage 1 first without --continue."
        )
    return json.loads(path.read_text())


def recover_state(root: Path, output_root: Path) -> dict:
    """Reconstruct orchestration state from existing generated directories."""
    LOGGER.info("Starting state recovery")
    LOGGER.info("Project root: %s", root)
    LOGGER.info("Output root: %s", output_root)
    state = {
        "current_stage": "stage_1",
        "completed_stages": [],
        "stages": {},
    }
    stage1_root = output_root / "stage_1.0_0pct"
    LOGGER.info("Checking Stage 1 directories: %s", stage1_root)
    ecut_values = set()
    for nqx in NQX_VALUES:
        nqx_root = stage1_root / f"nqx_{nqx}"
        for path in nqx_root.glob("ecut_*Ry"):
            match = re.fullmatch(r"ecut_([0-9]+(?:\.[0-9]+)?)Ry", path.name)
            if match:
                ecut_values.add(float(match.group(1)))

    if not ecut_values:
        LOGGER.error("No Stage 1 cutoff directories found under %s", stage1_root)
        raise RuntimeError(f"Could not recover Stage 1 cutoff directories under {stage1_root}")

    ecut_values = sorted(ecut_values)
    LOGGER.info("Recovered Stage 1 cutoff values: %s", ecut_values)
    state["stages"]["stage_1"] = {
        "status": "generated",
        "ecut_values_Ry": ecut_values,
        "nqx_values": list(NQX_VALUES),
        "energy_tolerance_Ry": STAGE1_ENERGY_TOL_RY,
    }

    ecut_map = {}
    stage1_complete = True
    for strain in STRAINS:
        strain_key = str(strain)
        ecut_map[strain_key] = {}
        for nqx in NQX_VALUES:
            try:
                ecut_map[strain_key][str(nqx)] = choose_converged_ecut(
                    output_root, strain, nqx, ecut_values
                )
            except RuntimeError as error:
                stage1_complete = False
                LOGGER.warning(
                    "Could not recover Stage 1 cutoff for strain=%s, nqx=%s: %s",
                    strain, nqx, error,
                )

    if stage1_complete:
        LOGGER.info("Stage 1 is complete; converged cutoffs recovered for all strain/nqx cases")
        state["completed_stages"].append("stage_1")
        state["stages"]["stage_1"]["converged_ecut"] = ecut_map
        state["current_stage"] = "stage_2_kpoint"
    else:
        LOGGER.warning("Stage 1 is not complete; Stage 2 and later stages cannot be recovered")

    stage2_root = output_root / "stage_2_kpoint"
    LOGGER.info("Checking Stage 2 directory: %s", stage2_root)
    if stage2_root.is_dir() and stage1_complete:
        LOGGER.info("Analyzing Stage 2 total energies")
        try:
            results = analyze_stage2(output_root, ecut_map)
            graph_path = plot_kpoint_convergence(output_root, results)
            state["completed_stages"].append("stage_2_kpoint")
            state["current_stage"] = "stage_2_kpoint_complete"
            state["stages"]["stage_2_kpoint"] = {
                "status": "analyzed",
                "kpoints": list(STAGE2_KPOINTS),
                "energy_tolerance_Ry": STAGE2_ENERGY_TOL_RY,
                "convergence_graph": str(graph_path),
                **results,
            }
            LOGGER.info("Stage 2 is complete; converged k-points recovered")
        except RuntimeError as error:
            LOGGER.warning("Stage 2 was not analyzed successfully: %s", error)
            state["current_stage"] = "stage_2_kpoint"
            state["stages"]["stage_2_kpoint"] = {
                "status": "generated",
                "kpoints": list(STAGE2_KPOINTS),
                "energy_tolerance_Ry": STAGE2_ENERGY_TOL_RY,
            }
    elif not stage2_root.is_dir():
        LOGGER.warning("Stage 2 directory does not exist: %s", stage2_root)

    stage3_root = output_root / "stage_3_vcrelax"
    LOGGER.info("Checking Stage 3 directory: %s", stage3_root)
    if stage3_root.is_dir() and stage1_complete:
        # If Stage 2 output files are missing, recover the selected k-point
        # directly from the Stage 3 directory names.  Stage 3 already stores
        # the k-point used for each completed vc-relax calculation.
        stage2_state = state["stages"].setdefault("stage_2_kpoint", {})
        kpoint_map = stage2_state.get("converged_kpoints")
        if kpoint_map is None:
            recovered_kpoints = {}
            kpoints_complete = True
            for strain in STRAINS:
                strain_key = str(strain)
                recovered_kpoints[strain_key] = {}
                for nqx in NQX_VALUES:
                    nqx_key = str(nqx)
                    ecut = float(ecut_map[strain_key][nqx_key])
                    parent = (
                        stage3_root / stage_name(strain) / f"nqx_{nqx}"
                        / f"ecut_{ecut:g}Ry"
                    )
                    candidates = []
                    if parent.is_dir():
                        for child in parent.iterdir():
                            match = re.fullmatch(
                                r"k_(\d+)x(\d+)x(\d+)", child.name
                            )
                            if (
                                child.is_dir()
                                and match
                                and match.group(1) == match.group(2)
                                and match.group(1) == match.group(3)
                            ):
                                candidates.append(int(match.group(1)))
                    if len(candidates) != 1:
                        kpoints_complete = False
                        LOGGER.warning(
                            "Cannot recover Stage 3 k-point for strain=%s, nqx=%s: "
                            "expected exactly one k_* directory under %s, found %s",
                            strain, nqx, parent, candidates,
                        )
                    else:
                        recovered_kpoints[strain_key][nqx_key] = candidates[0]

            if kpoints_complete:
                kpoint_map = recovered_kpoints
                stage2_state["converged_kpoints"] = kpoint_map
                stage2_state["recovery_source"] = "stage_3_vcrelax_directories"
                LOGGER.info("Recovered Stage 2 k-points from Stage 3 directory names")

        # Merely finding the Stage 3 directory is not enough: it may contain
        # inputs that were generated but whose vc-relax jobs are still running.
        # Verify every selected vc-relax run before advancing recovery.
        state["stages"]["stage_3_vcrelax"] = {
            "status": "generated",
            "nqx_values": list(NQX_VALUES),
            "cell_dynamics": "none",
        }
        state["current_stage"] = "stage_3_vcrelax"

        if kpoint_map is not None:
            LOGGER.info("Checking completed Stage 3 vc-relax outputs")
            try:
                results = export_results(output_root, state)
            except RuntimeError as error:
                # Stage 3 may be complete even when some Stage 2 output files
                # are unavailable, so verify Stage 3 independently below.
                LOGGER.warning("Could not export the full Stage 3 result set: %s", error)
                results = None

            stage3_complete = True
            for strain in STRAINS:
                strain_key = str(strain)
                for nqx in NQX_VALUES:
                    nqx_key = str(nqx)
                    ecut = float(ecut_map[strain_key][nqx_key])
                    kpoint = int(kpoint_map[strain_key][nqx_key])
                    run_dir = stage3_run_dir(
                        output_root, strain, nqx, ecut, kpoint
                    )
                    if total_energy(run_dir) is None or band_gap(run_dir) is None:
                        stage3_complete = False
                        LOGGER.warning(
                            "Stage 3 output is incomplete or unparsable: %s", run_dir
                        )
                        break
                if not stage3_complete:
                    break

        else:
            stage3_complete = False
            results = None

        if stage3_complete:
            LOGGER.info("Stage 3 vc-relax is complete")
            state["completed_stages"].append("stage_3_vcrelax")
            state["current_stage"] = "stage_4_results"
            stage3_update = {"status": "analyzed"}
            if results is not None:
                stage3_update["results"] = results
            else:
                stage3_update["recovery_note"] = (
                    "Recovered from completed vc-relax outputs; "
                    "Stage 2 output export was incomplete."
                )
            state["stages"]["stage_3_vcrelax"].update(stage3_update)
            state["stages"]["stage_4_results"] = {
                "status": "complete" if results is not None else "recovered",
            }
        else:
            LOGGER.warning("Stage 3 was found but is not complete")
    elif not stage3_root.is_dir():
        LOGGER.warning("Stage 3 directory does not exist: %s", stage3_root)

    save_state(root, state)
    LOGGER.info("Recovered %s from %s", STATE_FILENAME, output_root)
    LOGGER.info("Recovered current stage: %s", state["current_stage"])
    return state


def output_files(run_dir: Path) -> list[Path]:
    """Return likely Slurm/QE output files in one run directory."""
    return sorted(
        path for path in run_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".out", ".log"}
    )


def last_match(path: Path, patterns: tuple[str, ...]) -> float | None:
    text = path.read_text(errors="replace")
    values = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            values.append(float(match.group(1).replace("D", "E").replace("d", "e")))
    return values[-1] if values else None


def total_energy(run_dir: Path) -> float | None:
    number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
    patterns = (rf"!\s*total energy\s*=\s*({number})\s+Ry",)
    for path in reversed(output_files(run_dir)):
        value = last_match(path, patterns)
        if value is not None:
            return value
    return None


def ecut_energy_data(output_root: Path, strain: float, nqx: int, ecut_values: list[float]) -> list[tuple[float, float]]:
    results = []
    for ecut in sorted(ecut_values):
        run_dir = output_root / stage_name(strain) / f"nqx_{nqx}" / f"ecut_{ecut:g}Ry"
        energy = total_energy(run_dir)
        if energy is None:
            raise RuntimeError(f"Could not find a QE total energy in {run_dir}")
        results.append((ecut, energy))
    return results


def band_gap(run_dir: Path) -> float | None:
    """Extract a QE-reported highest-occupied/lowest-unoccupied gap in eV."""
    for path in reversed(output_files(run_dir)):
        text = path.read_text(errors="replace")
        paired = re.findall(
            r"highest occupied\s*,\s*lowest unoccupied level\s*\(ev\)\s*:\s*"
            r"([-+0-9.]+)\s+([-+0-9.]+)",
            text,
            flags=re.IGNORECASE,
        )
        if paired:
            occupied, unoccupied = paired[-1]
            return float(unoccupied) - float(occupied)

        occupied = re.findall(
            r"highest occupied level\s*\(ev\)\s*:\s*([-+0-9.]+)",
            text,
            flags=re.IGNORECASE,
        )
        unoccupied = re.findall(
            r"lowest unoccupied level\s*\(ev\)\s*:\s*([-+0-9.]+)",
            text,
            flags=re.IGNORECASE,
        )
        if occupied and unoccupied:
            return float(unoccupied[-1]) - float(occupied[-1])
    return None


def latest_output_file(run_dir: Path) -> Path | None:
    """Return the newest likely QE/Slurm output file containing results."""
    files = output_files(run_dir)
    return files[-1] if files else None


def parse_numeric_rows(lines: list[str], start: int, count: int) -> list[list[float]] | None:
    """Read a fixed number of numeric rows following a QE structure header."""
    rows = []
    index = start + 1
    while index < len(lines) and len(rows) < count:
        fields = lines[index].split()
        if not fields:
            index += 1
            continue
        if len(fields) < 3:
            return None
        try:
            rows.append([float(fields[0]), float(fields[1]), float(fields[2])])
        except ValueError:
            return None
        index += 1
    return rows if len(rows) == count else None


def final_relaxed_structure(run_dir: Path) -> tuple[str, list[list[float]], str, list[str]] | None:
    """Extract the last cell and atomic positions printed by vc-relax."""
    for path in reversed(output_files(run_dir)):
        lines = path.read_text(errors="replace").splitlines()
        cell = None
        atoms = None
        for index in range(len(lines) - 1, -1, -1):
            stripped = lines[index].strip()
            upper = stripped.upper()
            if cell is None and upper.startswith("CELL_PARAMETERS"):
                rows = parse_numeric_rows(lines, index, 3)
                if rows is not None:
                    cell = (stripped, rows)
            if atoms is None and upper.startswith("ATOMIC_POSITIONS"):
                atom_lines = []
                cursor = index + 1
                while cursor < len(lines):
                    fields = lines[cursor].split()
                    if len(fields) >= 4:
                        try:
                            float(fields[1])
                            float(fields[2])
                            float(fields[3])
                        except ValueError:
                            break
                        atom_lines.append("  " + "   ".join(fields[:4]))
                    elif atom_lines:
                        break
                    cursor += 1
                if atom_lines:
                    atoms = (stripped, atom_lines)
            if cell is not None and atoms is not None:
                return cell[0], cell[1], atoms[0], atoms[1]
    return None


def replace_structure_block(
    text: str,
    keyword: str,
    header: str,
    rows: list[str],
) -> str:
    """Replace a CELL_PARAMETERS or ATOMIC_POSITIONS block in an input."""
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines)
         if line.strip().upper().startswith(keyword.upper())),
        None,
    )
    if start is None:
        raise ValueError(f"Could not find {keyword} block")

    end = start + 1
    while end < len(lines):
        stripped = lines[end].strip()
        if re.match(r"^[A-Z][A-Z_ ]*(?:\s|\{|\(|=|$)", stripped):
            break
        end += 1
    replacement = [header, *rows]
    updated = lines[:start] + replacement + lines[end:]
    return "\n".join(updated) + "\n"


def final_scf_input(
    template: str,
    *,
    nqx: int,
    ecut: float,
    kpoint: int,
    pseudo_dir: str,
    cell_header: str,
    cell_rows: list[list[float]],
    atomic_header: str,
    atomic_rows: list[str],
) -> str:
    """Create a final SCF input using the relaxed vc-relax structure."""
    text = make_input(
        template,
        nqx=nqx,
        ecut=ecut,
        strain_percent=0.0,
        kpoint=kpoint,
        pseudo_dir=pseudo_dir,
    )
    formatted_cell = [
        f"  {row[0]:.9f}   {row[1]:.9f}   {row[2]:.9f}" for row in cell_rows
    ]
    text = replace_structure_block(text, "CELL_PARAMETERS", cell_header, formatted_cell)
    text = replace_structure_block(text, "ATOMIC_POSITIONS", atomic_header, atomic_rows)
    return text


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> Path:
    """Write rows with a stable column order."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_results(output_root: Path, state: dict) -> dict:
    """Tabulate Stage 1, Stage 2, and Stage 3 results in CSV files.

    Stage 3 is the final fixed-strain vc-relax calculation.  Its QE output is
    used for both the relaxed total energy and the HSE band gap.  All Stage 3
    runs must contain a total energy and a parseable band gap before export is
    marked complete.
    """
    stage1 = state["stages"]["stage_1"]
    ecut_values = [float(value) for value in stage1["ecut_values_Ry"]]
    ecut_map = stage1.get("converged_ecut", {})
    stage2 = state["stages"].get("stage_2_kpoint", {})
    kpoint_map = stage2.get("converged_kpoints", {})

    common_fields = [
        "stage", "strain_percent", "nqx", "ecut_Ry", "kpoint_grid",
        "total_energy_Ry", "band_gap_eV", "selected", "status", "output_file",
    ]
    all_rows: list[dict] = []
    stage1_rows: list[dict] = []
    stage2_rows: list[dict] = []
    relax_rows: list[dict] = []
    band_gap_rows: list[dict] = []

    for strain in STRAINS:
        strain_key = str(strain)
        for nqx in NQX_VALUES:
            nqx_key = str(nqx)
            selected_ecut = float(ecut_map[strain_key][nqx_key])
            for ecut in ecut_values:
                run_dir = output_root / stage_name(strain) / f"nqx_{nqx}" / f"ecut_{ecut:g}Ry"
                energy = total_energy(run_dir)
                output_path = latest_output_file(run_dir)
                row = {
                    "stage": "stage_1_ecut",
                    "strain_percent": strain,
                    "nqx": nqx,
                    "ecut_Ry": ecut,
                    "kpoint_grid": "",
                    "total_energy_Ry": "" if energy is None else energy,
                    "band_gap_eV": "",
                    "selected": "yes" if ecut == selected_ecut else "no",
                    "status": "complete" if energy is not None else "missing",
                    "output_file": "" if output_path is None else str(output_path),
                }
                stage1_rows.append(row)
                all_rows.append(row)

            selected_kpoint = int(kpoint_map[strain_key][nqx_key])
            for kpoint in STAGE2_KPOINTS:
                run_dir = (
                    output_root / "stage_2_kpoint" / stage_name(strain)
                    / f"nqx_{nqx}" / f"ecut_{selected_ecut:g}Ry"
                    / f"k_{kpoint}x{kpoint}x{kpoint}"
                )
                energy = total_energy(run_dir)
                output_path = latest_output_file(run_dir)
                row = {
                    "stage": "stage_2_kpoint",
                    "strain_percent": strain,
                    "nqx": nqx,
                    "ecut_Ry": selected_ecut,
                    "kpoint_grid": f"{kpoint}x{kpoint}x{kpoint}",
                    "total_energy_Ry": "" if energy is None else energy,
                    "band_gap_eV": "",
                    "selected": "yes" if kpoint == selected_kpoint else "no",
                    "status": "complete" if energy is not None else "missing",
                    "output_file": "" if output_path is None else str(output_path),
                }
                stage2_rows.append(row)
                all_rows.append(row)

            run_dir = (
                output_root / "stage_3_vcrelax" / stage_name(strain)
                / f"nqx_{nqx}" / f"ecut_{selected_ecut:g}Ry"
                / f"k_{selected_kpoint}x{selected_kpoint}x{selected_kpoint}"
            )
            energy = total_energy(run_dir)
            gap = band_gap(run_dir)
            output_path = latest_output_file(run_dir)
            if energy is None or gap is None:
                missing = []
                if energy is None:
                    missing.append("total energy")
                if gap is None:
                    missing.append("band gap")
                raise RuntimeError(
                    f"Could not find {', '.join(missing)} in completed vc-relax output: {run_dir}"
                )

            row = {
                "stage": "stage_3_vcrelax",
                "strain_percent": strain,
                "nqx": nqx,
                "ecut_Ry": selected_ecut,
                "kpoint_grid": f"{selected_kpoint}x{selected_kpoint}x{selected_kpoint}",
                "total_energy_Ry": energy,
                "band_gap_eV": gap,
                "selected": "yes",
                "status": "complete",
                "output_file": "" if output_path is None else str(output_path),
            }
            relax_rows.append(row)
            band_gap_rows.append({
                "strain_percent": strain,
                "nqx": nqx,
                "ecut_Ry": selected_ecut,
                "kpoint_grid": f"{selected_kpoint}x{selected_kpoint}x{selected_kpoint}",
                "band_gap_eV": gap,
                "relaxed_total_energy_Ry": energy,
                "output_file": "" if output_path is None else str(output_path),
            })
            all_rows.append(row)

    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results_dir / "all_results.csv", all_rows, common_fields)
    write_csv(results_dir / "stage1_ecut_results.csv", stage1_rows, common_fields)
    write_csv(results_dir / "stage2_kpoint_results.csv", stage2_rows, common_fields)
    write_csv(results_dir / "stage3_vcrelax_results.csv", relax_rows, common_fields)
    write_csv(
        results_dir / "band_gap_results.csv",
        band_gap_rows,
        [
            "strain_percent", "nqx", "ecut_Ry", "kpoint_grid", "band_gap_eV",
            "relaxed_total_energy_Ry", "output_file",
        ],
    )
    return {
        "results_directory": str(results_dir),
        "all_results_csv": str(results_dir / "all_results.csv"),
        "stage1_csv": str(results_dir / "stage1_ecut_results.csv"),
        "stage2_csv": str(results_dir / "stage2_kpoint_results.csv"),
        "stage3_csv": str(results_dir / "stage3_vcrelax_results.csv"),
        "band_gap_csv": str(results_dir / "band_gap_results.csv"),
        "band_gap_rows": band_gap_rows,
    }


def choose_converged_ecut(output_root: Path, strain: float, nqx: int, ecut_values: list[float]) -> float:
    results = ecut_energy_data(output_root, strain, nqx, ecut_values)

    for (ecut, energy), (next_ecut, next_energy) in zip(results, results[1:]):
        if abs(next_energy - energy) <= STAGE1_ENERGY_TOL_RY:
            # Select the larger cutoff from the converged pair.  This is the
            # conservative choice used for all Stage 2 calculations.
            return next_ecut
    raise RuntimeError(
        f"No converged ecut found for {stage_name(strain)}, nqx={nqx}; "
        f"increase the ecut sweep or adjust STAGE1_ENERGY_TOL_RY."
    )


def plot_ecut_convergence(output_root: Path, energy_data: dict, ecut_map: dict) -> Path:
    """Plot Stage 1 total-energy convergence used to select Stage 2 cutoffs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes = axes.ravel()
    for index, strain in enumerate(STRAINS):
        axis = axes[index]
        strain_key = str(strain)
        for nqx in NQX_VALUES:
            values = energy_data[strain_key][str(nqx)]
            x_values = [item[0] for item in values]
            y_values = [item[1] for item in values]
            axis.plot(x_values, y_values, marker="o", label=f"nqx={nqx}")
            selected = float(ecut_map[strain_key][str(nqx)])
            axis.axvline(selected, linestyle="--", alpha=0.6)
        title = "0% strain" if strain == 0 else f"{strain:+g}% strain"
        axis.set_title(title)
        axis.set_xlabel("ecutwfc (Ry)")
        axis.set_ylabel("Total energy (Ry)")
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[-1].axis("off")
    figure.suptitle("Stage 1 ecut convergence used for Stage 2", fontsize=15)
    output_path = output_root / "ecut_convergence.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def generate_stage1(root: Path, output_root: Path, template: str, submit_path: Path, ecut_values: list[float], submit: bool) -> None:
    generated = []
    for strain in STRAINS:
        for nqx in NQX_VALUES:
            for ecut in ecut_values:
                run_dir = output_root / stage_name(strain) / f"nqx_{nqx}" / f"ecut_{ecut:g}Ry"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "si.scf.in").write_text(
                    make_input(template, nqx=nqx, ecut=ecut, strain_percent=strain)
                )
                shutil.copy2(submit_path, run_dir / "submit.sh")
                generated.append(run_dir)
    print(f"Generated Stage 1: {len(generated)} calculations under {output_root}")
    if submit:
        for run_dir in generated:
            print(f"Submitting {run_dir.relative_to(root)}")
            subprocess.run(["sbatch", "submit.sh"], cwd=run_dir, check=True)


def generate_stage2(root: Path, output_root: Path, template: str, submit_path: Path, ecut_map: dict, submit: bool) -> None:
    generated = []
    for strain in STRAINS:
        strain_key = str(strain)
        for nqx in NQX_VALUES:
            ecut = float(ecut_map[strain_key][str(nqx)])
            for kpoint in STAGE2_KPOINTS:
                run_dir = (
                    output_root / "stage_2_kpoint" / stage_name(strain)
                    / f"nqx_{nqx}" / f"ecut_{ecut:g}Ry" / f"k_{kpoint}x{kpoint}x{kpoint}"
                )
                run_dir.mkdir(parents=True, exist_ok=True)
                pseudo_dir = os.path.relpath(root / "ReferenceFiles", run_dir)
                (run_dir / "si.scf.in").write_text(
                    make_input(
                        template,
                        nqx=nqx,
                        ecut=ecut,
                        strain_percent=strain,
                        kpoint=kpoint,
                        pseudo_dir=f"'{pseudo_dir}/'",
                    )
                )
                shutil.copy2(submit_path, run_dir / "submit.sh")
                generated.append(run_dir)
    print(f"Generated Stage 2: {len(generated)} calculations under {output_root / 'stage_2_kpoint'}")
    if submit:
        for run_dir in generated:
            print(f"Submitting {run_dir.relative_to(root)}")
            subprocess.run(["sbatch", "submit.sh"], cwd=run_dir, check=True)


def make_vcrelax_input(
    template: str,
    *,
    nqx: int,
    ecut: float,
    kpoint: int,
    strain_percent: float,
    pseudo_dir: str,
) -> str:
    """Prepare a fixed-strain HSE vc-relax input."""
    text = replace_assignment(text=template, name="pseudo_dir", value=pseudo_dir)
    text = replace_assignment(text, "ecutwfc", f"{ecut:g}")
    text = replace_assignment(text, "ecutrho", f"{4.0 * ecut:g}")
    text = ensure_assignment(text, "input_dft", "'HSE'")
    text = ensure_assignment(text, "nqx1", str(nqx))
    text = ensure_assignment(text, "nqx2", str(nqx))
    text = ensure_assignment(text, "nqx3", str(nqx))
    text = replace_assignment(text, "cell_dynamics", "'none'")
    text = replace_kpoints(text, kpoint)
    return replace_lattice(text, 1.0 + strain_percent / 100.0)


def generate_stage3_vcrelax(
    root: Path,
    output_root: Path,
    template: str,
    submit_path: Path,
    ecut_map: dict,
    kpoint_map: dict,
    submit: bool,
) -> None:
    generated = []
    reference_dir = root / "ReferenceFiles"
    for strain in STRAINS:
        strain_key = str(strain)
        for nqx in NQX_VALUES:
            nqx_key = str(nqx)
            ecut = float(ecut_map[strain_key][nqx_key])
            kpoint = int(kpoint_map[strain_key][nqx_key])
            run_dir = (
                output_root / "stage_3_vcrelax" / stage_name(strain)
                / f"nqx_{nqx}" / f"ecut_{ecut:g}Ry" / f"k_{kpoint}x{kpoint}x{kpoint}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            pseudo_dir = os.path.relpath(reference_dir, run_dir)
            (run_dir / "si.opt.in").write_text(
                make_vcrelax_input(
                    template,
                    nqx=nqx,
                    ecut=ecut,
                    kpoint=kpoint,
                    strain_percent=strain,
                    pseudo_dir=f"'{pseudo_dir}/'",
                )
            )
            shutil.copy2(submit_path, run_dir / "submit_gpuqs.sh")
            generated.append(run_dir)

    print(f"Generated Stage 3: {len(generated)} vc-relax calculations under {output_root / 'stage_3_vcrelax'}")
    if submit:
        for run_dir in generated:
            print(f"Submitting {run_dir.relative_to(root)}")
            subprocess.run(["sbatch", "submit_gpuqs.sh"], cwd=run_dir, check=True)


def stage3_run_dir(
    output_root: Path,
    strain: float,
    nqx: int,
    ecut: float,
    kpoint: int,
) -> Path:
    return (
        output_root / "stage_3_vcrelax" / stage_name(strain)
        / f"nqx_{nqx}" / f"ecut_{ecut:g}Ry"
        / f"k_{kpoint}x{kpoint}x{kpoint}"
    )


def generate_stage5_final_scf(
    root: Path,
    output_root: Path,
    template: str,
    submit_path: Path,
    ecut_map: dict,
    kpoint_map: dict,
    submit: bool,
) -> None:
    """Generate final SCF inputs from the last structures printed by vc-relax."""
    generated = []
    reference_dir = root / "ReferenceFiles"
    for strain in STRAINS:
        strain_key = str(strain)
        for nqx in NQX_VALUES:
            nqx_key = str(nqx)
            ecut = float(ecut_map[strain_key][nqx_key])
            kpoint = int(kpoint_map[strain_key][nqx_key])
            relax_dir = stage3_run_dir(output_root, strain, nqx, ecut, kpoint)
            structure = final_relaxed_structure(relax_dir)
            if structure is None:
                raise RuntimeError(
                    f"Could not extract final cell and atomic positions from vc-relax output: {relax_dir}"
                )
            cell_header, cell_rows, atomic_header, atomic_rows = structure
            run_dir = (
                output_root / "stage_5_final_scf" / stage_name(strain)
                / f"nqx_{nqx}" / f"ecut_{ecut:g}Ry"
                / f"k_{kpoint}x{kpoint}x{kpoint}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            pseudo_dir = os.path.relpath(reference_dir, run_dir)
            (run_dir / "si.scf.in").write_text(
                final_scf_input(
                    template,
                    nqx=nqx,
                    ecut=ecut,
                    kpoint=kpoint,
                    pseudo_dir=f"'{pseudo_dir}/'",
                    cell_header=cell_header,
                    cell_rows=cell_rows,
                    atomic_header=atomic_header,
                    atomic_rows=atomic_rows,
                )
            )
            shutil.copy2(submit_path, run_dir / "submit.sh")
            generated.append(run_dir)

    print(f"Generated Stage 5: {len(generated)} final SCF calculations under {output_root / 'stage_5_final_scf'}")
    if submit:
        for run_dir in generated:
            print(f"Submitting {run_dir.relative_to(root)}")
            subprocess.run(["sbatch", "submit.sh"], cwd=run_dir, check=True)


def analyze_final_scf(output_root: Path, state: dict) -> dict:
    """Collect final SCF energies and band gaps from relaxed structures."""
    ecut_map = state["stages"]["stage_1"]["converged_ecut"]
    kpoint_map = state["stages"]["stage_2_kpoint"]["converged_kpoints"]
    rows = []
    for strain in STRAINS:
        strain_key = str(strain)
        for nqx in NQX_VALUES:
            nqx_key = str(nqx)
            ecut = float(ecut_map[strain_key][nqx_key])
            kpoint = int(kpoint_map[strain_key][nqx_key])
            run_dir = (
                output_root / "stage_5_final_scf" / stage_name(strain)
                / f"nqx_{nqx}" / f"ecut_{ecut:g}Ry"
                / f"k_{kpoint}x{kpoint}x{kpoint}"
            )
            energy = total_energy(run_dir)
            gap = band_gap(run_dir)
            output_path = latest_output_file(run_dir)
            if energy is None or gap is None:
                missing = []
                if energy is None:
                    missing.append("total energy")
                if gap is None:
                    missing.append("band gap")
                raise RuntimeError(
                    f"Could not find {', '.join(missing)} in final SCF output: {run_dir}"
                )
            rows.append({
                "strain_percent": strain,
                "nqx": nqx,
                "ecut_Ry": ecut,
                "kpoint_grid": f"{kpoint}x{kpoint}x{kpoint}",
                "final_scf_energy_Ry": energy,
                "band_gap_eV": gap,
                "output_file": "" if output_path is None else str(output_path),
            })

    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "final_scf_results.csv"
    write_csv(
        csv_path,
        rows,
        [
            "strain_percent", "nqx", "ecut_Ry", "kpoint_grid",
            "final_scf_energy_Ry", "band_gap_eV", "output_file",
        ],
    )
    return {"final_scf_csv": str(csv_path), "rows": rows}


def submit_existing_stage(root: Path, stage_root: Path, submit_name: str = "submit.sh") -> int:
    """Submit already-generated calculation directories for the current stage."""
    submit_files = sorted(stage_root.rglob(submit_name))
    # Support Stage 3 directories generated by older script versions.
    if not submit_files and submit_name == "submit_gpuqs.sh":
        submit_files = sorted(stage_root.rglob("submit_gpuqs"))
    if not submit_files:
        raise RuntimeError(f"No {submit_name} files found under {stage_root}")
    for submit_file in submit_files:
        run_dir = submit_file.parent
        print(f"Submitting {run_dir.relative_to(root)}")
        subprocess.run(["sbatch", submit_file.name], cwd=run_dir, check=True)
    return len(submit_files)


def analyze_stage2(output_root: Path, ecut_map: dict) -> dict:
    """Find the lowest energy-converged k-point grid for every strain and nqx."""
    k_map = {}
    energies = {}
    for strain in STRAINS:
        strain_key = str(strain)
        k_map[strain_key] = {}
        energies[strain_key] = {}
        for nqx in NQX_VALUES:
            nqx_key = str(nqx)
            ecut = float(ecut_map[strain_key][nqx_key])
            values = []
            for kpoint in STAGE2_KPOINTS:
                run_dir = (
                    output_root / "stage_2_kpoint" / stage_name(strain)
                    / f"nqx_{nqx}" / f"ecut_{ecut:g}Ry" / f"k_{kpoint}x{kpoint}x{kpoint}"
                )
                energy = total_energy(run_dir)
                if energy is None:
                    raise RuntimeError(f"Could not find a QE total energy in {run_dir}")
                values.append((kpoint, energy))
            energies[strain_key][nqx_key] = {str(k): value for k, value in values}
            for (kpoint, energy), (_, next_energy) in zip(values, values[1:]):
                if abs(next_energy - energy) <= STAGE2_ENERGY_TOL_RY:
                    k_map[strain_key][nqx_key] = kpoint
                    break
            else:
                raise RuntimeError(
                    f"No converged k-point grid found for {stage_name(strain)}, nqx={nqx}; "
                    f"increase STAGE2_KPOINTS or adjust STAGE2_ENERGY_TOL_RY."
                )
    return {"converged_kpoints": k_map, "total_energies_Ry": energies}


def plot_kpoint_convergence(output_root: Path, results: dict) -> Path:
    """Plot Stage 2 total-energy convergence for every strain and nqx."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    axes = axes.ravel()
    energies = results["total_energies_Ry"]
    selected = results["converged_kpoints"]
    for index, strain in enumerate(STRAINS):
        axis = axes[index]
        strain_key = str(strain)
        for nqx in NQX_VALUES:
            nqx_key = str(nqx)
            values = energies[strain_key][nqx_key]
            x_values = [int(k) for k in values]
            y_values = [values[str(k)] for k in x_values]
            axis.plot(x_values, y_values, marker="o", label=f"nqx={nqx}")
            axis.axvline(selected[strain_key][nqx_key], linestyle="--", alpha=0.6)
        title = "0% strain" if strain == 0 else f"{strain:+g}% strain"
        axis.set_title(title)
        axis.set_xlabel("k-point grid (N×N×N)")
        axis.set_ylabel("Total energy (Ry)")
        axis.set_xticks(list(STAGE2_KPOINTS))
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].axis("off")
    figure.suptitle("Stage 2 k-point total-energy convergence", fontsize=15)
    output_path = output_root / "kpoint_convergence.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def generate(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    root = find_project_root(Path(args.root))
    reference = root / "ReferenceFiles"
    reference_hse = reference / "hse_cal"
    template_path = reference_hse / "si.scf.in"
    submit_path = reference_hse / "submit.sh"
    pseudo_path = reference / "Si_ONCV_PBE-1.2.upf"

    for path in (template_path, submit_path, pseudo_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required reference file: {path}")

    template = template_path.read_text()
    output_root = root / args.output
    output_root.mkdir(parents=True, exist_ok=True)

    if args.recover_state:
        recover_state(root, output_root)
        return 0
    if args.continue_stage:
        state = load_state(root)
        if state.get("current_stage") == "stage_1":
            ecut_map = {}
            energy_data = {}
            ecut_values = [float(value) for value in state["stages"]["stage_1"]["ecut_values_Ry"]]
            for strain in STRAINS:
                strain_key = str(strain)
                ecut_map[strain_key] = {}
                energy_data[strain_key] = {}
                for nqx in NQX_VALUES:
                    nqx_key = str(nqx)
                    energy_data[strain_key][nqx_key] = ecut_energy_data(
                        output_root, strain, nqx, ecut_values
                    )
                    ecut_map[strain_key][nqx_key] = choose_converged_ecut(
                        output_root, strain, nqx, ecut_values
                    )
            graph_path = plot_ecut_convergence(output_root, energy_data, ecut_map)
            generate_stage2(root, output_root, template, submit_path, ecut_map, args.submit)
            state["completed_stages"].append("stage_1")
            state["current_stage"] = "stage_2_kpoint"
            state["stages"]["stage_1"]["converged_ecut"] = ecut_map
            state["stages"]["stage_1"]["convergence_graph"] = str(graph_path)
            state["stages"]["stage_2_kpoint"] = {
                "status": "submitted" if args.submit else "generated",
                "kpoints": list(STAGE2_KPOINTS),
                "energy_tolerance_Ry": STAGE2_ENERGY_TOL_RY,
            }
            save_state(root, state)
            print(f"Updated {STATE_FILENAME}; current stage: stage_2_kpoint")
        elif state.get("current_stage") == "stage_2_kpoint":
            if not (output_root / "ecut_convergence.png").is_file():
                ecut_values = [
                    float(value) for value in state["stages"]["stage_1"]["ecut_values_Ry"]
                ]
                ecut_map = state["stages"]["stage_1"]["converged_ecut"]
                energy_data = {}
                for strain in STRAINS:
                    strain_key = str(strain)
                    energy_data[strain_key] = {}
                    for nqx in NQX_VALUES:
                        energy_data[strain_key][str(nqx)] = ecut_energy_data(
                            output_root, strain, nqx, ecut_values
                        )
                graph_path = plot_ecut_convergence(output_root, energy_data, ecut_map)
                state["stages"]["stage_1"]["convergence_graph"] = str(graph_path)
                save_state(root, state)
                print(f"Generated {graph_path}")
            if args.submit:
                count = submit_existing_stage(root, output_root / "stage_2_kpoint")
                state["stages"]["stage_2_kpoint"]["status"] = "submitted"
                save_state(root, state)
                print(f"Submitted {count} existing Stage 2 calculations")
                return 0
            ecut_map = state["stages"]["stage_1"]["converged_ecut"]
            results = analyze_stage2(output_root, ecut_map)
            graph_path = plot_kpoint_convergence(output_root, results)
            state["completed_stages"].append("stage_2_kpoint")
            state["current_stage"] = "stage_2_kpoint_complete"
            state["stages"]["stage_2_kpoint"].update(results)
            state["stages"]["stage_2_kpoint"]["convergence_graph"] = str(graph_path)
            state["stages"]["stage_2_kpoint"]["status"] = "analyzed"
            save_state(root, state)
            print(f"Generated {graph_path}")
            print(f"Updated {STATE_FILENAME}; Stage 2 k-point convergence is complete")
        elif state.get("current_stage") == "stage_2_kpoint_complete":
            vcrelax_template_path = root / "ReferenceFiles" / "vcrelax" / "si.opt.in"
            vcrelax_submit_path = root / "ReferenceFiles" / "vcrelax" / "submit_gpuqs.sh"
            for path in (vcrelax_template_path, vcrelax_submit_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing required vc-relax reference file: {path}")
            ecut_map = state["stages"]["stage_1"]["converged_ecut"]
            kpoint_map = state["stages"]["stage_2_kpoint"]["converged_kpoints"]
            generate_stage3_vcrelax(
                root,
                output_root,
                vcrelax_template_path.read_text(),
                vcrelax_submit_path,
                ecut_map,
                kpoint_map,
                args.submit,
            )
            state["completed_stages"].append("stage_2_kpoint_complete")
            state["current_stage"] = "stage_3_vcrelax"
            state["stages"]["stage_3_vcrelax"] = {
                "status": "submitted" if args.submit else "generated",
                "nqx_values": list(NQX_VALUES),
                "cell_dynamics": "none",
            }
            save_state(root, state)
            print(f"Updated {STATE_FILENAME}; current stage: stage_3_vcrelax")
        elif state.get("current_stage") == "stage_3_vcrelax":
            if args.submit:
                count = submit_existing_stage(root, output_root / "stage_3_vcrelax", "submit_gpuqs.sh")
                state["stages"]["stage_3_vcrelax"]["status"] = "submitted"
                save_state(root, state)
                print(f"Submitted {count} existing Stage 3 calculations")
            else:
                results = export_results(output_root, state)
                state["completed_stages"].append("stage_3_vcrelax")
                state["current_stage"] = "stage_4_results"
                state["stages"]["stage_3_vcrelax"].update({
                    "status": "analyzed",
                    "results": results,
                })
                state["stages"]["stage_4_results"] = {
                    "status": "complete",
                    **results,
                }
                save_state(root, state)
                print(f"Exported CSV results under {results['results_directory']}")
                print(f"Band gaps calculated for {len(results['band_gap_rows'])} relaxed cases")
        elif state.get("current_stage") == "stage_4_results":
            ecut_map = state["stages"]["stage_1"]["converged_ecut"]
            kpoint_map = state["stages"]["stage_2_kpoint"]["converged_kpoints"]
            generate_stage5_final_scf(
                root,
                output_root,
                template,
                submit_path,
                ecut_map,
                kpoint_map,
                args.submit,
            )
            state["completed_stages"].append("stage_4_results")
            state["current_stage"] = "stage_5_final_scf"
            state["stages"]["stage_5_final_scf"] = {
                "status": "submitted" if args.submit else "generated",
                "source": "relaxed vc-relax structures",
                "calculation": "scf",
            }
            save_state(root, state)
            print(f"Updated {STATE_FILENAME}; current stage: stage_5_final_scf")
        elif state.get("current_stage") == "stage_5_final_scf":
            if args.submit:
                count = submit_existing_stage(root, output_root / "stage_5_final_scf", "submit.sh")
                state["stages"]["stage_5_final_scf"]["status"] = "submitted"
                save_state(root, state)
                print(f"Submitted {count} existing final SCF calculations")
            else:
                results = analyze_final_scf(output_root, state)
                state["completed_stages"].append("stage_5_final_scf")
                state["current_stage"] = "stage_6_complete"
                state["stages"]["stage_5_final_scf"].update({
                    "status": "analyzed",
                    **results,
                })
                save_state(root, state)
                print(f"Final SCF analysis written to {results['final_scf_csv']}")
                print(f"Final SCF band gaps calculated for {len(results['rows'])} cases")
        else:
            raise RuntimeError(f"State is already at {state.get('current_stage')}; nothing to continue.")
    else:
        generate_stage1(root, output_root, template, submit_path, list(args.ecut), args.submit)
        state = {
            "current_stage": "stage_1",
            "completed_stages": [],
            "stages": {
                "stage_1": {
                    "status": "submitted" if args.submit else "generated",
                    "ecut_values_Ry": list(args.ecut),
                    "nqx_values": list(NQX_VALUES),
                    "energy_tolerance_Ry": STAGE1_ENERGY_TOL_RY,
                }
            },
        }
        save_state(root, state)
        print(f"Created {STATE_FILENAME}; current stage: stage_1")
        if not args.submit:
            print("After Stage 1 finishes, run: python3 orchestrate.py --continue")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="MATE250-FinalProject directory containing ReferenceFiles (default: script directory)",
    )
    parser.add_argument(
        "--output",
        default="generated_runs",
        help="Directory, relative to --root, for generated calculations",
    )
    parser.add_argument(
        "--ecut",
        type=float,
        nargs="+",
        default=DEFAULT_ECUTS,
        help="ecutwfc values in Ry to test (default: %(default)s)",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit every generated run with sbatch",
    )
    parser.add_argument(
        "--continue",
        dest="continue_stage",
        action="store_true",
        help="Analyze the current stage and generate or complete the next stage",
    )
    parser.add_argument(
        "--recover-state",
        action="store_true",
        help="Rebuild .orchestrate_state.json from existing generated calculation directories",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed diagnostic logging while generating or recovering stages",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(generate(parse_args()))
