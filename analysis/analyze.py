#!/home/019119984/miniconda3/envs/phonopy-221/bin/python
"""Rebuild the analysis tables, figures, and notes from generated_runs.

Run this again after the Quantum ESPRESSO inputs or outputs change. Every file
written here is regenerated from scratch.

    conda activate phonopy-221
    python analysis/analyze.py

The script refuses to run under any other Python. It reads finished and
unfinished run directories under generated_runs and writes:

    analysis/tables/*.csv
    analysis/figures/*.png
    analysis/notes.md
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


ENV_NAME = "phonopy-221"
RY_TO_EV = 13.605693122994
BOHR_TO_ANGSTROM = 0.529177249

# Same energy windows used by orchestrate.py when it picks the next stage.
ECUT_TOLERANCE_RY = 1.0e-4
KPOINT_TOLERANCE_RY = 1.0e-2

STAGE_ORDER = {"ecut": 0, "kpoint": 1, "vcrelax": 2, "bands": 3}
STAGE_LABELS = {
    "ecut": "Stage 1 cutoff sweep",
    "kpoint": "Stage 2 k-point sweep",
    "vcrelax": "Stage 3 fixed-cell relax",
    "bands": "Band path",
}

INPUT_NAMES = ("si.scf.in", "si.opt.in", "si.bands.in")
FIGURE_NAMES = (
    "ecut_convergence.png",
    "kpoint_convergence.png",
    "gap_kpoint_convergence.png",
    "structure_vs_strain.png",
    "energy_vs_strain.png",
    "energy_vs_strain_common_ecut.png",
    "gap_vs_strain.png",
)

STRAIN_DIRECTORY = re.compile(
    r"^stage_\d+\.\d+_(?:0pct|plus(\d+)pct|minus(\d+)pct)$"
)
ASSIGNMENT = re.compile(r"^([A-Za-z_][\w]*)\s*=\s*(.*?)\s*,?\s*$")
FINAL_COORDINATES = re.compile(
    r"Begin final coordinates\s*(.*?)\s*End final coordinates",
    re.IGNORECASE | re.DOTALL,
)
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
TOTAL_ENERGY = re.compile(
    rf"!\s*total energy\s*=\s*({NUMBER})\s+Ry",
    re.IGNORECASE,
)
OCCUPIED_PAIR = re.compile(
    r"highest occupied\s*,\s*lowest unoccupied level\s*\(ev\)\s*:\s*"
    rf"({NUMBER})\s+({NUMBER})",
    re.IGNORECASE,
)
CARDS = {
    "ATOMIC_SPECIES",
    "ATOMIC_POSITIONS",
    "CELL_PARAMETERS",
    "K_POINTS",
    "OCCUPATIONS",
    "CONSTRAINTS",
    "ATOMIC_FORCES",
}


def require_phonopy_environment() -> None:
    """Stop unless this process is the phonopy-221 interpreter."""
    executable = Path(sys.executable).resolve()
    if ENV_NAME not in executable.parts:
        raise SystemExit(
            "This analysis must run in the phonopy-221 conda environment.\n"
            f"This process is {executable}.\n"
            "Activate the environment, then run:\n"
            "  conda activate phonopy-221\n"
            "  python analysis/analyze.py"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project directory that contains generated_runs (default: parent of this script)",
    )
    return parser.parse_args()


def parse_qe_input(text: str) -> dict:
    """Read the namelist values, cell, positions, and k-mesh from a QE input."""
    assignments: dict[str, str] = {}
    cell_unit = None
    cell: list[list[float]] = []
    position_unit = None
    positions: list[tuple[str, float, float, float]] = []
    pseudopotential = ""
    k_mesh = ""

    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index].split("!")[0].strip()
        if not raw:
            index += 1
            continue
        upper = raw.upper()

        if upper.startswith("CELL_PARAMETERS"):
            cell_unit = unit_name(raw)
            cell = []
            index += 1
            while index < len(lines) and len(cell) < 3:
                row = lines[index].split("!")[0].strip()
                index += 1
                if not row:
                    continue
                cell.append([float(value) for value in row.split()[:3]])
            continue

        if upper.startswith("ATOMIC_POSITIONS"):
            position_unit = unit_name(raw)
            positions = []
            index += 1
            while index < len(lines):
                row = lines[index].split("!")[0].strip()
                if not row:
                    break
                if is_card(row):
                    break
                parts = row.split()
                if len(parts) < 4:
                    break
                positions.append(
                    (parts[0], float(parts[1]), float(parts[2]), float(parts[3]))
                )
                index += 1
            continue

        if upper.startswith("ATOMIC_SPECIES"):
            index += 1
            while index < len(lines):
                row = lines[index].split("!")[0].strip()
                if not row:
                    break
                if is_card(row):
                    break
                parts = row.split()
                if len(parts) < 3:
                    break
                if not pseudopotential:
                    pseudopotential = parts[2]
                index += 1
            continue

        if upper.startswith("K_POINTS"):
            if "AUTOMATIC" in upper:
                index += 1
                while index < len(lines) and not lines[index].split("!")[0].strip():
                    index += 1
                if index < len(lines):
                    parts = lines[index].split("!")[0].split()
                    k_mesh = "x".join(str(int(float(parts[i]))) for i in range(3))
                    index += 1
            else:
                index += 1
            continue

        match = ASSIGNMENT.match(raw)
        if match:
            assignments[match.group(1).lower()] = match.group(2).strip().strip("'\"")
        index += 1

    nat = assignments.get("nat", "")
    return {
        "calculation": assignments.get("calculation", ""),
        "functional": assignments.get("input_dft", ""),
        "ecutwfc_Ry": float(assignments["ecutwfc"]) if "ecutwfc" in assignments else None,
        "nqx": int(float(assignments["nqx1"])) if "nqx1" in assignments else None,
        "cell_dynamics": assignments.get("cell_dynamics", ""),
        "ion_dynamics": assignments.get("ion_dynamics", ""),
        "nat": int(float(nat)) if nat else None,
        "pseudopotential": pseudopotential,
        "k_mesh": k_mesh,
        "cell_unit": cell_unit,
        "cell": cell,
        "position_unit": position_unit,
        "positions": positions,
    }


def unit_name(card_line: str) -> str:
    """Return the unit token from a CELL_PARAMETERS or ATOMIC_POSITIONS card."""
    match = re.search(r"[\{\(]\s*([A-Za-z]+)", card_line)
    if match:
        return match.group(1).lower()
    parts = card_line.split()
    return parts[-1].lower() if len(parts) > 1 else "angstrom"


def is_card(line: str) -> bool:
    token = line.split()[0].upper().strip(",")
    return token.startswith("&") or token.startswith("/") or token in CARDS


def vectors_in_angstrom(cell: list[list[float]], unit: str | None) -> np.ndarray:
    vectors = np.asarray(cell, dtype=float)
    name = (unit or "angstrom").lower()
    if "bohr" in name:
        return vectors * BOHR_TO_ANGSTROM
    if "angstrom" in name:
        return vectors
    raise ValueError(f"Unsupported coordinate unit '{unit}'")


def conventional_lattice_constant(vectors: np.ndarray, nat: int | None) -> float | None:
    """Return the cubic lattice constant implied by a 2-atom or 8-atom cell.

    For the 2-atom diamond primitive cell the cell volume is a^3 / 4.
    """
    volume = abs(float(np.dot(vectors[0], np.cross(vectors[1], vectors[2]))))
    if nat == 2:
        return (4.0 * volume) ** (1.0 / 3.0)
    if nat == 8:
        return volume ** (1.0 / 3.0)
    return None


def nearest_si_distance(
    vectors: np.ndarray,
    positions: list[tuple[str, float, float, float]],
    unit: str | None,
) -> float | None:
    """Return the shortest silicon-silicon distance, including periodic images."""
    silicon = [item for item in positions if item[0].lower().startswith("si")]
    if len(silicon) < 2:
        return None
    coordinates = np.array([[item[1], item[2], item[3]] for item in silicon], dtype=float)
    name = (unit or "crystal").lower()
    if "crystal" in name:
        fractional = coordinates
    elif "angstrom" in name:
        fractional = coordinates @ np.linalg.inv(vectors)
    elif "bohr" in name:
        fractional = (coordinates * BOHR_TO_ANGSTROM) @ np.linalg.inv(vectors)
    else:
        raise ValueError(f"Unsupported position unit '{unit}'")

    shortest = None
    for left in range(len(fractional)):
        for right in range(left + 1, len(fractional)):
            delta = fractional[right] - fractional[left]
            delta -= np.round(delta)
            distance = float(np.linalg.norm(delta @ vectors))
            if shortest is None or distance < shortest:
                shortest = distance
    return shortest


def geometry(parsed: dict) -> tuple[float | None, float | None]:
    if len(parsed["cell"]) != 3 or not parsed["positions"]:
        return None, None
    vectors = vectors_in_angstrom(parsed["cell"], parsed["cell_unit"])
    lattice = conventional_lattice_constant(vectors, parsed["nat"])
    bond = nearest_si_distance(vectors, parsed["positions"], parsed["position_unit"])
    return lattice, bond


def last_number(pattern: re.Pattern[str], text: str, group: int = 1) -> float | None:
    found = list(pattern.finditer(text))
    if not found:
        return None
    return float(found[-1].group(group).replace("D", "E").replace("d", "e"))


def parse_qe_output(path: Path) -> dict:
    text = path.read_text(errors="replace")
    job_done = "JOB DONE" in text and "convergence NOT achieved" not in text
    occupied = OCCUPIED_PAIR.findall(text)
    gap = None
    if occupied:
        highest = float(occupied[-1][0])
        lowest = float(occupied[-1][1])
        gap = lowest - highest

    relaxed_lattice = None
    relaxed_bond = None
    blocks = FINAL_COORDINATES.findall(text)
    if blocks:
        relaxed = parse_qe_input(blocks[-1])
        relaxed_lattice, relaxed_bond = geometry(relaxed)

    return {
        "job_done": job_done,
        "total_energy_Ry": last_number(TOTAL_ENERGY, text),
        "eigenvalue_gap_eV": gap,
        "relaxed_lattice_constant_A": relaxed_lattice,
        "relaxed_si_si_distance_A": relaxed_bond,
        "output_file": path.name,
    }


def pw_output(run_dir: Path) -> Path | None:
    """Return the pw.x stdout in a run directory, ignoring Slurm copies."""
    preferred = ("si.scf.out", "si.opt.out", "si.bands.out")
    found = []
    for path in run_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".out":
            continue
        if re.fullmatch(r"qe-\d+\.out", path.name):
            continue
        found.append(path)
    for name in preferred:
        for path in found:
            if path.name == name:
                return path
    return sorted(found)[0] if found else None


def nominal_strain(run_dir: Path, generated: Path) -> float | None:
    for part in run_dir.relative_to(generated).parts:
        match = STRAIN_DIRECTORY.match(part)
        if not match:
            continue
        if match.group(1):
            return float(match.group(1))
        if match.group(2):
            return -float(match.group(2))
        return 0.0
    return None


def stage_family(run_dir: Path, generated: Path, calculation: str) -> str:
    parts = run_dir.relative_to(generated).parts
    joined = "/".join(parts)
    if "vcrelax" in joined or calculation == "vc-relax":
        return "vcrelax"
    if calculation == "bands" or "bands" in joined:
        return "bands"
    if "kpoint" in joined or "/k_" in f"/{joined}":
        return "kpoint"
    return "ecut"


def discover_runs(generated: Path) -> list[dict]:
    run_dirs = set()
    for name in INPUT_NAMES:
        for path in generated.rglob(name):
            run_dirs.add(path.parent)

    runs = []
    errors = []
    for run_dir in sorted(run_dirs):
        try:
            runs.append(read_run(run_dir, generated))
        except Exception as exc:  # one bad directory should not drop the other runs
            errors.append(f"{run_dir}: {exc}")
    if errors:
        print("Skipped runs that could not be parsed:", file=sys.stderr)
        for message in errors:
            print(f"  {message}", file=sys.stderr)
    return runs


def read_run(run_dir: Path, generated: Path) -> dict:
    input_path = next(path for name in INPUT_NAMES if (path := run_dir / name).is_file())
    parsed = parse_qe_input(input_path.read_text(errors="replace"))
    lattice, bond = geometry(parsed)
    output_path = pw_output(run_dir)
    output = {
        "job_done": False,
        "total_energy_Ry": None,
        "eigenvalue_gap_eV": None,
        "relaxed_lattice_constant_A": None,
        "relaxed_si_si_distance_A": None,
        "output_file": "",
    }
    if output_path is not None:
        output = parse_qe_output(output_path)

    calculation = parsed["calculation"]
    mesh_gap = output["eigenvalue_gap_eV"] if calculation != "bands" else None
    band_gap = output["eigenvalue_gap_eV"] if calculation == "bands" else None
    project_root = generated.parent
    return {
        "stage": stage_family(run_dir, generated, calculation),
        "nominal_strain_percent": nominal_strain(run_dir, generated),
        "nqx": parsed["nqx"],
        "ecutwfc_Ry": parsed["ecutwfc_Ry"],
        "k_mesh": parsed["k_mesh"],
        "calculation": calculation,
        "functional": parsed["functional"],
        "pseudopotential": parsed["pseudopotential"],
        "nat": parsed["nat"],
        "cell_dynamics": parsed["cell_dynamics"],
        "ion_dynamics": parsed["ion_dynamics"],
        "lattice_constant_A": lattice,
        "si_si_distance_A": bond,
        "relaxed_lattice_constant_A": output["relaxed_lattice_constant_A"],
        "relaxed_si_si_distance_A": output["relaxed_si_si_distance_A"],
        "total_energy_Ry": output["total_energy_Ry"] if output["job_done"] else None,
        "mesh_gap_eV": mesh_gap if output["job_done"] else None,
        "band_path_gap_eV": band_gap if output["job_done"] else None,
        "job_done": output["job_done"],
        "run_dir": run_dir.relative_to(project_root).as_posix(),
    }


def finished(run: dict) -> bool:
    return bool(run["job_done"]) and run["total_energy_Ry"] is not None


def mesh_rank(mesh: str) -> int:
    if not mesh:
        return 0
    return int(mesh.split("x")[0])


def group_runs(runs: list[dict], stage: str) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        if run["stage"] == stage and run["nominal_strain_percent"] is not None:
            grouped[(run["nominal_strain_percent"], run["nqx"])].append(run)
    return grouped


def choose_converged(sequence: list[dict], tolerance: float, pick: str) -> dict | None:
    """Pick the cutoff or k-mesh using the same pair rule as orchestrate.py.

    Cutoff selection returns the higher member of the first pair inside the
    tolerance. K-mesh selection returns the coarser member of that pair.
    """
    usable = [run for run in sequence if finished(run)]
    for left, right in zip(usable, usable[1:]):
        if abs(right["total_energy_Ry"] - left["total_energy_Ry"]) <= tolerance:
            return right if pick == "next" else left
    return None


def convergence_rows(
    runs: list[dict],
    stage: str,
    sweep: str,
    tolerance: float,
    pick: str,
) -> list[dict]:
    rows = []
    for key in sorted(group_runs(runs, stage), key=strain_nqx_key):
        group = group_runs(runs, stage)[key]
        group.sort(key=lambda run: run["ecutwfc_Ry"] if sweep == "ecut" else mesh_rank(run["k_mesh"]))
        chosen = choose_converged(group, tolerance, pick)
        previous = None
        for run in group:
            delta = None
            if finished(run) and previous is not None:
                delta = run["total_energy_Ry"] - previous["total_energy_Ry"]
            if finished(run):
                previous = run
            rows.append(
                {
                    "nominal_strain_percent": run["nominal_strain_percent"],
                    "nqx": run["nqx"],
                    "ecutwfc_Ry": run["ecutwfc_Ry"],
                    "k_mesh": run["k_mesh"],
                    "total_energy_Ry": run["total_energy_Ry"],
                    "delta_energy_Ry": delta,
                    "within_tolerance": (
                        "yes" if delta is not None and abs(delta) <= tolerance else "no"
                    ),
                    "selected": "yes" if chosen is not None and run["run_dir"] == chosen["run_dir"] else "no",
                    "mesh_gap_eV": run["mesh_gap_eV"],
                    "job_done": "yes" if run["job_done"] else "no",
                    "run_dir": run["run_dir"],
                }
            )
    return rows


def strain_nqx_key(key: tuple) -> tuple:
    strain, nqx = key
    return (strain, nqx if nqx is not None else -1)


def reference_lattices(runs: list[dict]) -> dict[tuple, float]:
    """Map (stage, nqx) to the lattice constant of the nominal 0% input."""
    references = {}
    for run in runs:
        if run["nominal_strain_percent"] == 0 and run["lattice_constant_A"] is not None:
            references[(run["stage"], run["nqx"])] = run["lattice_constant_A"]
    return references


def structure_rows(runs: list[dict]) -> list[dict]:
    references = reference_lattices(runs)
    grouped: dict[tuple, dict] = {}
    for run in runs:
        if run["lattice_constant_A"] is None or run["nominal_strain_percent"] is None:
            continue
        key = (run["stage"], run["nominal_strain_percent"], run["nqx"])
        current = grouped.get(key)
        if current is None or (
            current["relaxed_lattice_constant_A"] is None
            and run["relaxed_lattice_constant_A"] is not None
        ):
            grouped[key] = run

    rows = []
    for key in sorted(grouped, key=lambda item: (STAGE_ORDER.get(item[0], 9), item[1], item[2] or -1)):
        run = grouped[key]
        reference = references.get((run["stage"], run["nqx"]))
        measured_strain = None
        if reference:
            measured_strain = (run["lattice_constant_A"] / reference - 1.0) * 100.0
        rows.append(
            {
                "stage": run["stage"],
                "nominal_strain_percent": run["nominal_strain_percent"],
                "nqx": run["nqx"],
                "lattice_constant_A": run["lattice_constant_A"],
                "reference_lattice_constant_A": reference,
                "strain_from_stage_reference_percent": measured_strain,
                "si_si_distance_A": run["si_si_distance_A"],
                "relaxed_lattice_constant_A": run["relaxed_lattice_constant_A"],
                "relaxed_si_si_distance_A": run["relaxed_si_si_distance_A"],
                "cell_dynamics": run["cell_dynamics"],
                "run_dir": run["run_dir"],
            }
        )
    return rows


def relative_energies(rows: list[dict]) -> None:
    zeros = {}
    for row in rows:
        if (
            row.get("selected", "yes") == "yes"
            and row["nominal_strain_percent"] == 0
            and row["total_energy_Ry"] is not None
        ):
            zeros[row["nqx"]] = row["total_energy_Ry"]
    for row in rows:
        baseline = zeros.get(row["nqx"])
        if baseline is None or row["total_energy_Ry"] is None or row.get("selected", "yes") != "yes":
            row["energy_relative_to_zero_Ry"] = None
            row["energy_relative_to_zero_eV"] = None
            row["energy_relative_to_zero_meV"] = None
            continue
        delta_ry = row["total_energy_Ry"] - baseline
        row["energy_relative_to_zero_Ry"] = delta_ry
        row["energy_relative_to_zero_eV"] = delta_ry * RY_TO_EV
        row["energy_relative_to_zero_meV"] = delta_ry * RY_TO_EV * 1000.0


def energy_rows_at_selected_k(kpoint_rows: list[dict]) -> list[dict]:
    rows = [dict(row) for row in kpoint_rows if row["selected"] == "yes"]
    relative_energies(rows)
    return rows


def common_ecut_rows(ecut_rows: list[dict]) -> list[dict]:
    """Energy and mesh gap at one cutoff shared by every strain in an nqx series."""
    by_nqx: dict[int, dict[float, set]] = defaultdict(lambda: defaultdict(set))
    lookup = {}
    for row in ecut_rows:
        if row["job_done"] != "yes" or row["total_energy_Ry"] is None:
            continue
        by_nqx[row["nqx"]][row["nominal_strain_percent"]].add(row["ecutwfc_Ry"])
        lookup[(row["nqx"], row["nominal_strain_percent"], row["ecutwfc_Ry"])] = row

    selected = []
    for nqx, strain_map in sorted(by_nqx.items()):
        if not strain_map:
            continue
        shared = set.intersection(*strain_map.values()) if strain_map else set()
        if not shared:
            continue
        ecut = max(shared)
        for strain in sorted(strain_map):
            row = dict(lookup[(nqx, strain, ecut)])
            row["selected"] = "yes"
            row["common_ecutwfc_Ry"] = ecut
            selected.append(row)
    relative_energies(selected)
    return selected


def parameter_rows(
    runs: list[dict],
    ecut_rows: list[dict],
    kpoint_rows: list[dict],
) -> list[dict]:
    chosen_ecut = {
        (row["nominal_strain_percent"], row["nqx"]): row
        for row in ecut_rows
        if row["selected"] == "yes"
    }
    chosen_k = {
        (row["nominal_strain_percent"], row["nqx"]): row
        for row in kpoint_rows
        if row["selected"] == "yes"
    }
    stage3 = {}
    for run in runs:
        if run["stage"] == "vcrelax":
            stage3[(run["nominal_strain_percent"], run["nqx"])] = run

    keys = sorted(
        set(chosen_ecut) | set(chosen_k) | set(stage3),
        key=strain_nqx_key,
    )
    rows = []
    for key in keys:
        strain, nqx = key
        ecut_choice = chosen_ecut.get(key)
        k_choice = chosen_k.get(key)
        relax = stage3.get(key)
        source = k_choice or ecut_choice
        source_run = next(
            (run for run in runs if source and run["run_dir"] == source["run_dir"]),
            relax,
        )
        ecut_value = ecut_choice["ecutwfc_Ry"] if ecut_choice else None
        k_mesh = k_choice["k_mesh"] if k_choice else ""
        stage3_ecut = relax["ecutwfc_Ry"] if relax else None
        stage3_mesh = relax["k_mesh"] if relax else ""
        matches = ""
        if relax and ecut_value is not None and k_mesh:
            matches = (
                "yes"
                if stage3_ecut == ecut_value and stage3_mesh == k_mesh
                else "no"
            )
        rows.append(
            {
                "nominal_strain_percent": strain,
                "nqx": nqx,
                "functional": (source_run or {}).get("functional", ""),
                "pseudopotential": (source_run or {}).get("pseudopotential", ""),
                "ecutwfc_Ry": ecut_value,
                "ecut_converged": "yes" if ecut_choice else "no",
                "k_mesh": k_mesh,
                "kpoint_converged": "yes" if k_choice else "no",
                "stage3_ecutwfc_Ry": stage3_ecut,
                "stage3_k_mesh": stage3_mesh,
                "stage3_matches_rule": matches,
                "lattice_constant_A": None if source_run is None else source_run["lattice_constant_A"],
                "run_dir": "" if source_run is None else source_run["run_dir"],
            }
        )
    return rows


def gap_rows(runs: list[dict], selected_energy: list[dict]) -> list[dict]:
    bands = {
        (run["nominal_strain_percent"], run["nqx"]): run
        for run in runs
        if run["stage"] == "bands" and run["band_path_gap_eV"] is not None
    }
    rows = []
    for energy in selected_energy:
        key = (energy["nominal_strain_percent"], energy["nqx"])
        band = bands.pop(key, None)
        rows.append(
            {
                "nominal_strain_percent": energy["nominal_strain_percent"],
                "nqx": energy["nqx"],
                "ecutwfc_Ry": energy["ecutwfc_Ry"],
                "k_mesh": energy["k_mesh"],
                "mesh_gap_eV": energy["mesh_gap_eV"],
                "band_path_gap_eV": None if band is None else band["band_path_gap_eV"],
                "run_dir": energy["run_dir"],
            }
        )
    for key in sorted(bands, key=strain_nqx_key):
        band = bands[key]
        rows.append(
            {
                "nominal_strain_percent": band["nominal_strain_percent"],
                "nqx": band["nqx"],
                "ecutwfc_Ry": band["ecutwfc_Ry"],
                "k_mesh": band["k_mesh"],
                "mesh_gap_eV": None,
                "band_path_gap_eV": band["band_path_gap_eV"],
                "run_dir": band["run_dir"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_value(column, row.get(column)) for column in columns})


def format_value(column: str, value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if column.endswith("_percent") and abs(value - round(value)) < 1.0e-8:
            return str(int(round(value)))
        if "energy_Ry" in column or column.endswith("_Ry"):
            return f"{value:.8f}"
        if column.endswith("_eV"):
            return f"{value:.6f}"
        if column.endswith("_meV"):
            return f"{value:.3f}"
        if column.endswith("_A"):
            return f"{value:.6f}"
        return f"{value:.8g}"
    return str(value)


def write_tables(table_dir: Path, tables: dict[str, tuple[list[dict], list[str]]]) -> None:
    for name, (rows, columns) in tables.items():
        write_csv(table_dir / name, rows, columns)


RUN_COLUMNS = [
    "stage",
    "nominal_strain_percent",
    "nqx",
    "ecutwfc_Ry",
    "k_mesh",
    "calculation",
    "functional",
    "pseudopotential",
    "nat",
    "cell_dynamics",
    "ion_dynamics",
    "lattice_constant_A",
    "si_si_distance_A",
    "relaxed_lattice_constant_A",
    "relaxed_si_si_distance_A",
    "total_energy_Ry",
    "mesh_gap_eV",
    "band_path_gap_eV",
    "job_done",
    "run_dir",
]
PARAMETER_COLUMNS = [
    "nominal_strain_percent",
    "nqx",
    "functional",
    "pseudopotential",
    "ecutwfc_Ry",
    "ecut_converged",
    "k_mesh",
    "kpoint_converged",
    "stage3_ecutwfc_Ry",
    "stage3_k_mesh",
    "stage3_matches_rule",
    "lattice_constant_A",
    "run_dir",
]
ECUT_COLUMNS = [
    "nominal_strain_percent",
    "nqx",
    "ecutwfc_Ry",
    "total_energy_Ry",
    "delta_energy_Ry",
    "within_tolerance",
    "selected",
    "mesh_gap_eV",
    "job_done",
    "run_dir",
]
KPOINT_COLUMNS = [
    "nominal_strain_percent",
    "nqx",
    "ecutwfc_Ry",
    "k_mesh",
    "total_energy_Ry",
    "delta_energy_Ry",
    "within_tolerance",
    "selected",
    "mesh_gap_eV",
    "job_done",
    "run_dir",
]
STRUCTURE_COLUMNS = [
    "stage",
    "nominal_strain_percent",
    "nqx",
    "lattice_constant_A",
    "reference_lattice_constant_A",
    "strain_from_stage_reference_percent",
    "si_si_distance_A",
    "relaxed_lattice_constant_A",
    "relaxed_si_si_distance_A",
    "cell_dynamics",
    "run_dir",
]
ENERGY_COLUMNS = [
    "nominal_strain_percent",
    "nqx",
    "ecutwfc_Ry",
    "k_mesh",
    "total_energy_Ry",
    "energy_relative_to_zero_Ry",
    "energy_relative_to_zero_eV",
    "energy_relative_to_zero_meV",
    "mesh_gap_eV",
    "run_dir",
]
COMMON_ECUT_COLUMNS = ["common_ecutwfc_Ry", *ENERGY_COLUMNS]
GAP_COLUMNS = [
    "nominal_strain_percent",
    "nqx",
    "ecutwfc_Ry",
    "k_mesh",
    "mesh_gap_eV",
    "band_path_gap_eV",
    "run_dir",
]
MASS_COLUMNS = [
    "nominal_strain_percent",
    "carrier",
    "valley",
    "effective_mass_m0",
    "band_edge",
    "method",
    "run_dir",
]


def write_figures(
    figure_dir: Path,
    ecut_rows: list[dict],
    kpoint_rows: list[dict],
    structure: list[dict],
    selected_energy: list[dict],
    common_energy: list[dict],
    gaps: list[dict],
) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Skipped figures because matplotlib could not be imported: {exc}", file=sys.stderr)
        return written

    tasks = (
        ("ecut_convergence.png", lambda: plot_sweep(plt, ecut_rows, "ecutwfc_Ry", "Plane-wave cutoff (Ry)", "Cutoff convergence of the total energy")),
        ("kpoint_convergence.png", lambda: plot_sweep(plt, kpoint_rows, "k_mesh", "k-point mesh", "K-point convergence of the total energy")),
        ("gap_kpoint_convergence.png", lambda: plot_gap_sweep(plt, kpoint_rows)),
        ("structure_vs_strain.png", lambda: plot_structure(plt, structure)),
        ("energy_vs_strain.png", lambda: plot_energy(plt, selected_energy, "Total energy relative to nominal 0%")),
        ("energy_vs_strain_common_ecut.png", lambda: plot_energy(plt, common_energy, "Total energy relative to nominal 0% at a shared cutoff")),
        ("gap_vs_strain.png", lambda: plot_gaps(plt, gaps)),
    )
    for name, draw in tasks:
        path = figure_dir / name
        figure = draw()
        if figure is None:
            if path.exists():
                path.unlink()
            continue
        figure.savefig(path, dpi=160)
        plt.close(figure)
        written.append(name)
    return written


def strains_in(rows: list[dict]) -> list[float]:
    return sorted({row["nominal_strain_percent"] for row in rows if row["nominal_strain_percent"] is not None})


def nqx_values(rows: list[dict]) -> list[int]:
    return sorted({row["nqx"] for row in rows if row["nqx"] is not None})


def strain_title(strain: float) -> str:
    if strain == 0:
        return "0% strain"
    return f"{strain:+.0f}% strain"


def plot_sweep(plt, rows: list[dict], x_key: str, x_label: str, title: str):
    usable = [row for row in rows if row["total_energy_Ry"] is not None]
    if not usable:
        return None
    strain_values = strains_in(usable)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    flat_axes = axes.ravel()
    for axis, strain in zip(flat_axes, strain_values):
        for nqx in nqx_values(usable):
            series = [
                row for row in usable
                if row["nominal_strain_percent"] == strain and row["nqx"] == nqx
            ]
            series.sort(key=lambda row: row["ecutwfc_Ry"] if x_key == "ecutwfc_Ry" else mesh_rank(row["k_mesh"]))
            if not series:
                continue
            x_values = [
                row[x_key] if x_key == "ecutwfc_Ry" else mesh_rank(row["k_mesh"])
                for row in series
            ]
            y_values = [row["total_energy_Ry"] for row in series]
            axis.plot(x_values, y_values, marker="o", label=f"nqx={nqx}")
            selected = next((row for row in series if row["selected"] == "yes"), None)
            if selected is not None:
                selected_x = selected[x_key] if x_key == "ecutwfc_Ry" else mesh_rank(selected["k_mesh"])
                axis.axvline(selected_x, color="0.4", linestyle="--", linewidth=1)
        axis.set_title(strain_title(strain))
        axis.set_xlabel(x_label)
        axis.set_ylabel("Total energy (Ry)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    for axis in flat_axes[len(strain_values):]:
        axis.axis("off")
    figure.suptitle(title)
    return figure


def plot_gap_sweep(plt, rows: list[dict]):
    usable = [row for row in rows if row["mesh_gap_eV"] is not None]
    if not usable:
        return None
    strain_values = strains_in(usable)
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    flat_axes = axes.ravel()
    for axis, strain in zip(flat_axes, strain_values):
        for nqx in nqx_values(usable):
            series = [
                row for row in usable
                if row["nominal_strain_percent"] == strain and row["nqx"] == nqx
            ]
            series.sort(key=lambda row: mesh_rank(row["k_mesh"]))
            if not series:
                continue
            axis.plot(
                [mesh_rank(row["k_mesh"]) for row in series],
                [row["mesh_gap_eV"] for row in series],
                marker="o",
                label=f"nqx={nqx}",
            )
        axis.set_title(strain_title(strain))
        axis.set_xlabel("k-point mesh")
        axis.set_ylabel("SCF mesh gap (eV)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
    for axis in flat_axes[len(strain_values):]:
        axis.axis("off")
    figure.suptitle("K-point convergence of the SCF mesh gap")
    return figure


def plot_structure(plt, rows: list[dict]):
    if not rows:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    # Stages that share one lattice would otherwise draw on top of each other.
    signatures: dict[tuple, list[tuple]] = defaultdict(list)
    for stage in sorted({row["stage"] for row in rows}, key=lambda item: STAGE_ORDER.get(item, 9)):
        for nqx in nqx_values(rows):
            series = [
                row for row in rows
                if row["stage"] == stage and row["nqx"] == nqx and row["lattice_constant_A"] is not None
            ]
            series.sort(key=lambda row: row["nominal_strain_percent"])
            if not series:
                continue
            signature = tuple(
                (
                    round(row["nominal_strain_percent"], 6),
                    round(row["lattice_constant_A"], 6),
                    None if row["si_si_distance_A"] is None else round(row["si_si_distance_A"], 6),
                )
                for row in series
            )
            signatures[signature].append((stage, nqx, series))

    markers = ["o", "s", "^", "D"]
    linestyles = ["-", "--", ":", "-."]
    for index, members in enumerate(signatures.values()):
        _stage, _nqx, series = members[0]
        by_stage: dict[str, list[int]] = defaultdict(list)
        for stage, nqx, _series in members:
            by_stage[stage].append(nqx)
        label = "; ".join(
            f"{STAGE_LABELS.get(stage, stage)} (nqx={', '.join(str(value) for value in sorted(nqxes))})"
            for stage, nqxes in sorted(by_stage.items(), key=lambda item: STAGE_ORDER.get(item[0], 9))
        )
        x_values = [row["nominal_strain_percent"] for row in series]
        style = {
            "marker": markers[index % len(markers)],
            "linestyle": linestyles[index % len(linestyles)],
            "label": label,
        }
        axes[0].plot(x_values, [row["lattice_constant_A"] for row in series], **style)
        bond = [row["si_si_distance_A"] for row in series]
        if any(value is not None for value in bond):
            axes[1].plot(x_values, bond, **style)
    axes[0].set_ylabel("Conventional lattice constant (Å)")
    axes[1].set_ylabel("Nearest Si–Si distance (Å)")
    for axis in axes:
        axis.set_xlabel("Nominal strain (%)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7)
    figure.suptitle("Imposed structure versus nominal strain")
    return figure


def plot_energy(plt, rows: list[dict], title: str):
    usable = [row for row in rows if row.get("energy_relative_to_zero_eV") is not None]
    if not usable:
        return None
    figure, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for nqx in nqx_values(usable):
        series = [row for row in usable if row["nqx"] == nqx]
        series.sort(key=lambda row: row["nominal_strain_percent"])
        axis.plot(
            [row["nominal_strain_percent"] for row in series],
            [row["energy_relative_to_zero_eV"] for row in series],
            marker="o",
            label=f"nqx={nqx}",
        )
    axis.axhline(0.0, color="0.5", linewidth=1)
    axis.set_xlabel("Nominal strain (%)")
    axis.set_ylabel("Energy relative to nominal 0% (eV / cell)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.set_title(title)
    return figure


def plot_gaps(plt, rows: list[dict]):
    mesh = [row for row in rows if row["mesh_gap_eV"] is not None]
    bands = [row for row in rows if row["band_path_gap_eV"] is not None]
    if not mesh and not bands:
        return None
    figure, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for nqx in nqx_values(mesh or bands):
        series = [row for row in mesh if row["nqx"] == nqx]
        series.sort(key=lambda row: row["nominal_strain_percent"])
        if series:
            axis.plot(
                [row["nominal_strain_percent"] for row in series],
                [row["mesh_gap_eV"] for row in series],
                marker="o",
                label=f"SCF mesh gap, nqx={nqx}",
            )
        band_series = [row for row in bands if row["nqx"] == nqx]
        band_series.sort(key=lambda row: row["nominal_strain_percent"])
        if band_series:
            axis.plot(
                [row["nominal_strain_percent"] for row in band_series],
                [row["band_path_gap_eV"] for row in band_series],
                marker="s",
                linestyle="--",
                label=f"Band-path gap, nqx={nqx}",
            )
    axis.set_xlabel("Nominal strain (%)")
    axis.set_ylabel("Gap (eV)")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    axis.set_title("Band gap versus nominal strain")
    return figure


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(headers) + " |", divider]
    for row in rows:
        body.append("| " + " | ".join(row) + " |")
    return "\n".join(body)


def display(value, digits: int | None = None) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        if digits is None:
            return f"{value:.6g}"
        return f"{value:.{digits}f}"
    return str(value)


def strain_text(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:+.0f}"


def write_notes(
    path: Path,
    executable: str,
    runs: list[dict],
    parameters: list[dict],
    structure: list[dict],
    selected_energy: list[dict],
    gaps: list[dict],
    figures: list[str],
) -> None:
    finished_count = sum(1 for run in runs if run["job_done"] == "yes")
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        by_stage[run["stage"]].append(run)

    inventory = []
    for stage in sorted(by_stage, key=lambda item: STAGE_ORDER.get(item, 9)):
        group = by_stage[stage]
        done = sum(1 for run in group if run["job_done"] == "yes")
        inventory.append([STAGE_LABELS.get(stage, stage), str(len(group)), str(done)])

    parameter_view = [
        [
            strain_text(row["nominal_strain_percent"]),
            display(row["nqx"]),
            display(row["ecutwfc_Ry"], 0),
            row["k_mesh"],
            row["stage3_matches_rule"],
            display(row["lattice_constant_A"], 4),
        ]
        for row in parameters
    ]
    structure_view = [
        [
            STAGE_LABELS.get(row["stage"], row["stage"]),
            strain_text(row["nominal_strain_percent"]),
            display(row["nqx"]),
            display(row["lattice_constant_A"], 4),
            display(row["strain_from_stage_reference_percent"], 3),
            display(row["si_si_distance_A"], 4),
            display(row["relaxed_si_si_distance_A"], 4),
        ]
        for row in structure
        if row["nqx"] == 3 or row["nqx"] is None
    ]
    energy_view = [
        [
            strain_text(row["nominal_strain_percent"]),
            display(row["nqx"]),
            row["k_mesh"],
            display(row["energy_relative_to_zero_meV"], 1),
            display(row["mesh_gap_eV"], 4),
        ]
        for row in selected_energy
    ]

    references = {}
    for row in structure:
        if row["nominal_strain_percent"] == 0 and row["lattice_constant_A"] is not None:
            references.setdefault(row["stage"], set()).add(round(row["lattice_constant_A"], 6))
    reference_lines = []
    for stage, values in references.items():
        shown = ", ".join(f"{value:.4f} Å" for value in sorted(values))
        reference_lines.append(f"- {STAGE_LABELS.get(stage, stage)} nominal 0% lattice constant: {shown}.")
    distinct = {value for values in references.values() for value in values}
    if len(distinct) > 1:
        reference_lines.append(
            "- These stages do not share one 0% lattice constant. "
            "Nominal strain percentages are therefore not one common physical axis."
        )

    relaxed_ready = any(row["relaxed_lattice_constant_A"] is not None for row in structure)
    band_ready = any(row["band_path_gap_eV"] is not None for row in gaps)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    text = f"""# Analysis notes

Generated by `analysis/analyze.py` on {timestamp} with `{executable}`.
Re-run the script after `generated_runs` changes. This file is overwritten each time.

```bash
conda activate phonopy-221
python analysis/analyze.py
```

## Inventory

{finished_count} of {len(runs)} run directories contain a finished pw.x output (`JOB DONE`).

{markdown_table(["Stage", "Run directories", "Finished"], inventory)}

The full directory list is `tables/runs.csv`.

## Chosen settings

A cutoff is selected when the total energy of the next higher cutoff changes by no more than {ECUT_TOLERANCE_RY:.0e} Ry. A k-mesh is selected when the next finer mesh changes the total energy by no more than {KPOINT_TOLERANCE_RY:.0e} Ry. These are the same windows used by `orchestrate.py`.

{markdown_table(["Strain (%)", "nqx", "ecut (Ry)", "k-mesh", "Stage 3 matches", "a (Å)"], parameter_view)}

`Stage 3 matches` compares the fixed-cell relax inputs with the cutoff and k-mesh selected above. Details are in `tables/parameters.csv`. Convergence values for every cutoff and k-mesh are in `tables/convergence_ecut.csv` and `tables/convergence_kpoints.csv`.

## Structure

`lattice_constant_A` is the conventional cubic lattice constant of the 2-atom cell, `(4V)^(1/3)`. `strain_from_stage_reference_percent` compares that lattice constant with the nominal 0% cell of the same stage and `nqx`. The Si–Si column is the nearest periodic silicon-silicon distance in the input. The relaxed column is filled from `Begin final coordinates` when a relaxation output is present.

{markdown_table(["Stage", "Nominal strain (%)", "nqx", "a (Å)", "Measured strain (%)", "Si–Si (Å)", "Relaxed Si–Si (Å)"], structure_view)}

The table above shows `nqx` = 3. `tables/structure_vs_strain.csv` also contains `nqx` = 4.

{chr(10).join(reference_lines)}

## Energy and gap

Energies below use the selected stage 2 k-mesh. The zero of energy is the nominal 0% run at the same `nqx`. `tables/energy_vs_strain_common_ecut.csv` repeats the comparison at one cutoff shared by every strain, so a cutoff change is not mixed into the strain trend.

{markdown_table(["Strain (%)", "nqx", "k-mesh", "Relative energy (meV/cell)", "SCF mesh gap (eV)"], energy_view)}

The SCF mesh gap is the difference between the highest occupied and lowest unoccupied eigenvalues on the automatic k-mesh. Silicon's gap is indirect, so this value can sit above the gap found on a band path, and a converged total energy does not by itself show that the gap is converged. `gap_kpoint_convergence.png` shows the mesh gap across the k-point sweep.

{"Band-path gaps are included in `tables/gap_vs_strain.csv`." if band_ready else "No band-path calculation has been read yet. `band_path_gap_eV` in `tables/gap_vs_strain.csv` is empty."}

{"Relaxed lattice constants and bond lengths were read from relaxation outputs." if relaxed_ready else "No relaxation output with final coordinates has been read yet. The relaxed columns are empty."}

## Effective mass

`tables/effective_mass.csv` has headers only. Electron and hole masses need the curvature of a band path near the band edges. Conductivity is not estimated here.

## Figures

{chr(10).join(f"- `figures/{name}`" for name in figures) if figures else "- No figures were written."}
"""
    path.write_text(text)


def main() -> int:
    require_phonopy_environment()
    args = parse_args()
    root = args.root.resolve()
    generated = root / "generated_runs"
    analysis = Path(__file__).resolve().parent
    if not generated.is_dir():
        raise SystemExit(f"Could not find generated runs at {generated}")

    runs = discover_runs(generated)
    runs.sort(
        key=lambda run: (
            STAGE_ORDER.get(run["stage"], 9),
            run["nominal_strain_percent"] if run["nominal_strain_percent"] is not None else 999,
            run["nqx"] if run["nqx"] is not None else -1,
            run["ecutwfc_Ry"] if run["ecutwfc_Ry"] is not None else -1,
            mesh_rank(run["k_mesh"]),
        )
    )

    ecut_rows = convergence_rows(runs, "ecut", "ecut", ECUT_TOLERANCE_RY, "next")
    kpoint_rows = convergence_rows(runs, "kpoint", "k", KPOINT_TOLERANCE_RY, "current")
    structure = structure_rows(runs)
    parameters = parameter_rows(runs, ecut_rows, kpoint_rows)
    selected_energy = energy_rows_at_selected_k(kpoint_rows)
    common_energy = common_ecut_rows(ecut_rows)
    gaps = gap_rows(runs, selected_energy)

    for row in runs:
        row["job_done"] = "yes" if row["job_done"] else "no"

    tables = {
        "runs.csv": (runs, RUN_COLUMNS),
        "parameters.csv": (parameters, PARAMETER_COLUMNS),
        "convergence_ecut.csv": (ecut_rows, ECUT_COLUMNS),
        "convergence_kpoints.csv": (kpoint_rows, KPOINT_COLUMNS),
        "structure_vs_strain.csv": (structure, STRUCTURE_COLUMNS),
        "energy_vs_strain.csv": (selected_energy, ENERGY_COLUMNS),
        "energy_vs_strain_common_ecut.csv": (common_energy, COMMON_ECUT_COLUMNS),
        "gap_vs_strain.csv": (gaps, GAP_COLUMNS),
        "effective_mass.csv": ([], MASS_COLUMNS),
    }
    write_tables(analysis / "tables", tables)
    figures = write_figures(
        analysis / "figures",
        ecut_rows,
        kpoint_rows,
        structure,
        selected_energy,
        common_energy,
        gaps,
    )
    write_notes(
        analysis / "notes.md",
        sys.executable,
        runs,
        parameters,
        structure,
        selected_energy,
        gaps,
        figures,
    )
    print(f"Wrote {len(tables)} tables, {len(figures)} figures, and notes.md under {analysis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
