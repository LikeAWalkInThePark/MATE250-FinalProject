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
import json
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
STAGE2_ENERGY_TOL_RY = 1.0e-2
STAGE2_KPOINTS = (6, 8, 10, 12, 14)
STATE_FILENAME = ".orchestrate_state.json"


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
    state = {
        "current_stage": "stage_1",
        "completed_stages": [],
        "stages": {},
    }
    stage1_root = output_root / "stage_1.0_0pct"
    ecut_values = set()
    for nqx in NQX_VALUES:
        nqx_root = stage1_root / f"nqx_{nqx}"
        for path in nqx_root.glob("ecut_*Ry"):
            match = re.fullmatch(r"ecut_([0-9]+(?:\.[0-9]+)?)Ry", path.name)
            if match:
                ecut_values.add(float(match.group(1)))

    if not ecut_values:
        raise RuntimeError(f"Could not recover Stage 1 cutoff directories under {stage1_root}")

    ecut_values = sorted(ecut_values)
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
            except RuntimeError:
                stage1_complete = False

    if stage1_complete:
        state["completed_stages"].append("stage_1")
        state["stages"]["stage_1"]["converged_ecut"] = ecut_map
        state["current_stage"] = "stage_2_kpoint"

    stage2_root = output_root / "stage_2_kpoint"
    if stage2_root.is_dir() and stage1_complete:
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
        except RuntimeError:
            state["current_stage"] = "stage_2_kpoint"
            state["stages"]["stage_2_kpoint"] = {
                "status": "generated",
                "kpoints": list(STAGE2_KPOINTS),
                "energy_tolerance_Ry": STAGE2_ENERGY_TOL_RY,
            }

    stage3_root = output_root / "stage_3_vcrelax"
    if stage3_root.is_dir():
        state["completed_stages"].append(state["current_stage"])
        state["current_stage"] = "stage_3_vcrelax"
        state["stages"]["stage_3_vcrelax"] = {
            "status": "generated",
            "nqx_values": list(NQX_VALUES),
            "cell_dynamics": "none",
        }

    save_state(root, state)
    print(f"Recovered {STATE_FILENAME} from {output_root}")
    print(f"Recovered current stage: {state['current_stage']}")
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
            if not args.submit:
                raise RuntimeError("Stage 3 inputs are already generated. Use --continue --submit to submit them.")
            count = submit_existing_stage(root, output_root / "stage_3_vcrelax", "submit_gpuqs.sh")
            state["stages"]["stage_3_vcrelax"]["status"] = "submitted"
            save_state(root, state)
            print(f"Submitted {count} existing Stage 3 calculations")
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
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(generate(parse_args()))
