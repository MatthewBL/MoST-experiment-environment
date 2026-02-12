import argparse
import subprocess
from pathlib import Path

from fmperf.utils.constants import REQUESTS_DIR, REQUESTS_FILENAME

DEFAULT_PROMPTS = Path(__file__).resolve().with_name("oasst_roots_en_max1000_tokens.jsonl")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a workload file for a specific input token range"
    )
    parser.add_argument("min_input", type=int, help="Minimum input tokens")
    parser.add_argument("max_input", type=int, help="Maximum input tokens")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing workload file")
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=DEFAULT_PROMPTS,
        help="Path to the prompts JSONL dataset (defaults to oasst_roots_en_max1000_tokens.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional custom output path for the generated requests JSON file",
    )
    return parser


def _validate_ranges(min_value: int, max_value: int, label: str) -> None:
    if min_value <= 0 or max_value <= 0:
        raise ValueError(f"{label} tokens must be positive")
    if min_value > max_value:
        raise ValueError(f"{label} min tokens cannot exceed max tokens")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _validate_ranges(args.min_input, args.max_input, "Input")

    prompts_path = args.prompts_file
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    requests_dir = Path(REQUESTS_DIR)
    requests_dir.mkdir(parents=True, exist_ok=True)
    target_path = args.output if args.output is not None else requests_dir / REQUESTS_FILENAME

    if target_path.exists() and not args.overwrite:
        print(f"Existing workload found at {target_path}. Use --overwrite to regenerate.")
        return

    command = [
        "python",
        "-u",
        "-m",
        "fmperf.loadgen.generate-input",
        "--min-input",
        str(args.min_input),
        "--max-input",
        str(args.max_input),
        "--prompts-file",
        str(prompts_path),
        "--output",
        str(target_path),
    ]
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    print(f"Generated workload at {target_path}")


if __name__ == "__main__":
    main()
