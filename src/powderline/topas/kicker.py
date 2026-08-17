"""``topas-kicker`` CLI: recipe JSON -> TOPAS v7 INP -> run -> standardized tables.

The default command generates the ``.inp`` + ``.xye``, runs TOPAS if a ``tc``
executable is discoverable, and parses the results into the same standardized
tables the GSAS-II path emits (``refined_parameters.csv``,
``<phase>_unit_cell_report.csv``, a fit summary). With no TOPAS present (e.g. this
Linux dev box) it degrades to generate-only with a clear note (exit 0)::

    pixi run topas-kicker examples/example_LaB6/input.json            # generate (+run if tc found)
    pixi run topas-kicker recipe.json --run --topas-version 8         # force a run, pick TOPAS 8
    pixi run topas-kicker recipe.json --no-run                        # generate only
    pixi run topas-kicker recipe.json --validate-only                 # translate + report, no files
    pixi run topas-kicker recipe.json --parse-results out_results.csv # round-trip an existing result

Entry point: ``python -m powderline.topas.kicker`` (D9). Not part of the 6-name
public API and no ``run()`` dispatch yet -- both follow once the demo lands.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import roundtrip as rt
from . import runner as tc_runner
from .errors import TopasTranslationError
from .writer import RESULTS_SUFFIX, render_topas, write_topas_inp


def _load_recipe(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _derive_base_name(input_json: Path) -> str:
    """Base name for the emitted files: the example dir name for ``.../input.json``."""
    if input_json.stem == "input":
        return input_json.parent.name
    return input_json.stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="topas-kicker",
        description="Generate a TOPAS v7 .inp + .xye from a GSASII_Rietveld recipe (no GSAS-II required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pixi run topas-kicker examples/example_LaB6/input.json
  pixi run topas-kicker examples/example_DRX_33/input.json --output out/
  pixi run topas-kicker recipe.json --validate-only
        """,
    )
    parser.add_argument("input_json", type=Path, help="Path to the input JSON recipe file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <recipe_parent>/output/topas)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Translate and report without writing any files (exit 1 on a translation error)",
    )
    parser.add_argument(
        "--parse-results",
        type=Path,
        default=None,
        metavar="RESULTS_CSV",
        help="Parse a TOPAS <base>_results.csv into standardized refined-parameter "
        "tables (round-trip) instead of generating an INP",
    )
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument(
        "--run",
        action="store_true",
        help="Run TOPAS after generating (error out if tc.exe cannot be found)",
    )
    run_group.add_argument(
        "--no-run",
        action="store_true",
        help="Only generate the INP + xye; never invoke TOPAS",
    )
    parser.add_argument("--topas-dir", type=Path, default=None, help="TOPAS install directory")
    parser.add_argument("--topas-version", default=None, help="TOPAS version (selects C:/TOPAS<version>)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print the full INP to stdout")
    return parser


def _write_roundtrip_tables(recipe, base_name, results_text, output_dir, source_dir=None) -> rt.RoundTripResult:
    """Parse a results CSV and write the standardized (bare-named) tables.

    Delegates to the shared ``engine.write_standardized_outputs`` so the CLI and
    ``run(engine="topas")`` produce one identical naming scheme. ``source_dir`` is
    where TOPAS's headerless profile/peak intermediates live (defaults to
    ``output_dir``). Raises ``ValueError`` (loud) on an empty/unusable results file.
    """
    from .engine import write_standardized_outputs

    outputs = write_standardized_outputs(recipe, base_name, results_text, output_dir, source_dir)
    for path in outputs["written"]:
        print(f"wrote {path}")
    stats = ", ".join(f"{k}={v:g}" for k, v in outputs["result"].fit.items())
    if stats:
        print(f"  fit: {stats}")
    return outputs["result"]


def _run_parse_results(input_json: Path, base_name: str, results_path: Path, output_dir: Path) -> int:
    if not results_path.exists():
        print(f"error: results file not found: {results_path}", file=sys.stderr)
        return 1
    recipe = _load_recipe(input_json)
    try:
        _write_roundtrip_tables(
            recipe, base_name, results_path.read_text(), output_dir, source_dir=results_path.parent
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _run_pipeline(recipe, base_name, output_dir, args) -> int:
    """generate INP+xye -> (optionally) run TOPAS -> parse results into tables."""
    try:
        gen = write_topas_inp(recipe, str(output_dir), base_name=base_name)
    except TopasTranslationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {gen['inp_path']}")
    print(f"wrote {gen['xye_path']}")
    if gen["dropped_points"]:
        print(f"  ({gen['dropped_points']} non-positive-weight point(s) dropped from .xye)")
    for msg in gen["warnings"]:
        print(f"  warning: {msg}", file=sys.stderr)
    if args.verbose:
        print(Path(gen["inp_path"]).read_text())

    if args.no_run:
        return 0

    run_result = tc_runner.run_topas(gen["inp_path"], args.topas_dir, args.topas_version)
    if run_result.status == "tc_not_found":
        if args.run:
            print(
                "error: --run given but no tc.exe found (set topas.dir/topas.version in "
                ".powderline_config.yaml or pass --topas-dir)",
                file=sys.stderr,
            )
            return 1
        print("note: TOPAS (tc.exe) not found -> generated INP + xye only; skipped run + parse.")
        print("      configure topas.dir/topas.version or pass --topas-dir to run end-to-end.")
        return 0
    if run_result.status == "failed":
        print(f"error: TOPAS returned exit code {run_result.returncode}", file=sys.stderr)
        return 1

    out_dir = Path(output_dir)
    results_path = out_dir / f"{base_name}{RESULTS_SUFFIX}"
    if not results_path.exists():
        print(
            f"error: TOPAS ran (rc 0) but {results_path.name} was not produced; "
            "check the INP results-export block",
            file=sys.stderr,
        )
        return 1
    print(f"TOPAS ran via {run_result.tc_path}")
    try:
        _write_roundtrip_tables(recipe, base_name, results_path.read_text(), out_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_json: Path = args.input_json

    if not input_json.exists():
        print(f"error: recipe not found: {input_json}", file=sys.stderr)
        return 1

    base_name = _derive_base_name(input_json)

    if args.parse_results is not None:
        output_dir = args.output if args.output else input_json.parent / "output" / "topas"
        return _run_parse_results(input_json, base_name, args.parse_results, output_dir)

    recipe = _load_recipe(input_json)

    if args.validate_only:
        try:
            rendered = render_topas(recipe, base_name)
        except TopasTranslationError as exc:
            print(f"INVALID: {exc}", file=sys.stderr)
            return 1
        print(f"OK: {input_json} translates ({len(rendered.refined_names)} refined prm(s))")
        for msg in rendered.warnings:
            print(f"  warning: {msg}", file=sys.stderr)
        if args.verbose:
            print(rendered.inp_text)
        return 0

    output_dir = args.output if args.output else input_json.parent / "output" / "topas"
    return _run_pipeline(recipe, base_name, output_dir, args)


if __name__ == "__main__":
    sys.exit(main())
