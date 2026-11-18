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
import re
import shutil
import subprocess
from pathlib import Path


# Cutoffs are intentionally easy to change.  The production value can be
# selected after checking that the band gap is converged with respect to ecut.
DEFAULT_ECUTS = (30.0, 35.0, 40.0, 45.0, 50.0, 60.0)
STRAINS = (0.0, 1.0, 2.0, -1.0, -2.0)
NQX_VALUES = (3, 4)


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


def make_input(template: str, *, nqx: int, ecut: float, strain_percent: float) -> str:
    scale = 1.0 + strain_percent / 100.0
    text = template
    # Each run is four directory levels below MATE250-FinalProject:
    # generated_runs/stage/nqx/ecut.  ReferenceFiles is at the project root.
    text = replace_assignment(text, "pseudo_dir", "'../../../../ReferenceFiles/'")
    text = replace_assignment(text, "ecutwfc", f"{ecut:g}")
    text = replace_assignment(text, "ecutrho", f"{4.0 * ecut:g}")
    text = replace_nqx(text, nqx)
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
    generated = []

    for strain in STRAINS:
        stage_dir = output_root / stage_name(strain)
        for nqx in NQX_VALUES:
            nqx_dir = stage_dir / f"nqx_{nqx}"
            for ecut in args.ecut:
                run_dir = nqx_dir / f"ecut_{ecut:g}Ry"
                run_dir.mkdir(parents=True, exist_ok=True)

                (run_dir / "si.scf.in").write_text(
                    make_input(template, nqx=nqx, ecut=ecut, strain_percent=strain)
                )
                shutil.copy2(submit_path, run_dir / "submit.sh")
                generated.append(run_dir)

    print(f"Generated {len(generated)} calculations under {output_root}")

    if args.submit:
        for run_dir in generated:
            print(f"Submitting {run_dir.relative_to(root)}")
            subprocess.run(["sbatch", "submit.sh"], cwd=run_dir, check=True)
    else:
        print("No jobs submitted. Re-run with --submit after reviewing the inputs.")
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
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(generate(parse_args()))
