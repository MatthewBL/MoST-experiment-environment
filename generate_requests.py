import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

from fmperf.utils.constants import REQUESTS_DIR, REQUESTS_FILENAME

DEFAULT_PROMPTS = Path(__file__).resolve().with_name("oasst_roots_en_max1000_tokens.jsonl")
TOKEN_FIELD = "gpt2_token_count"


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
        "--min-output",
        type=int,
        default=None,
        help="Minimum generated tokens (defaults to the minimum input tokens if omitted)",
    )
    parser.add_argument(
        "--max-output",
        type=int,
        default=None,
        help="Maximum generated tokens (defaults to the maximum input tokens if omitted)",
    )
    parser.add_argument(
        "--frac-greedy",
        type=float,
        default=None,
        help="Optional override for the FRAC_GREEDY probability",
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


def _build_filtered_prompts_file(prompts_path: Path, min_tokens: int, max_tokens: int) -> Tuple[Path, int]:
    """Create a temporary JSONL containing only prompts within the token interval."""
    filtered_count = 0
    tmp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl", encoding="utf-8")
    try:
        with prompts_path.open("r", encoding="utf-8") as source, tmp_file:
            for line in source:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token_value = record.get(TOKEN_FIELD)
                if token_value is None:
                    continue
                try:
                    token_count = int(token_value)
                except (TypeError, ValueError):
                    continue
                if min_tokens <= token_count <= max_tokens:
                    tmp_file.write(json.dumps(record, ensure_ascii=False))
                    tmp_file.write("\n")
                    filtered_count += 1
    except Exception:
        os.unlink(tmp_file.name)
        raise

    if filtered_count == 0:
        os.unlink(tmp_file.name)
        raise ValueError(
            f"No prompts with {TOKEN_FIELD} between {min_tokens} and {max_tokens} were found in {prompts_path}"
        )

    return Path(tmp_file.name), filtered_count


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _validate_ranges(args.min_input, args.max_input, "Input")

    prompts_path = args.prompts_file
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    filtered_prompts_path, filtered_count = _build_filtered_prompts_file(
        prompts_path, args.min_input, args.max_input
    )
    print(
        f"Using {filtered_count} prompts from {prompts_path} with {TOKEN_FIELD} between {args.min_input} and {args.max_input}"
    )

    requests_dir = Path(REQUESTS_DIR)
    requests_dir.mkdir(parents=True, exist_ok=True)
    base_name = Path(REQUESTS_FILENAME)
    name_root = base_name.stem
    suffix = base_name.suffix or ".json"
    default_output = requests_dir / f"{name_root}_{args.min_input}-{args.max_input}{suffix}"
    target_path = args.output if args.output is not None else default_output

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
        str(filtered_prompts_path),
        "--output",
        str(target_path),
    ]
    if args.min_output is not None:
        command.extend(["--min-output", str(args.min_output)])
    if args.max_output is not None:
        command.extend(["--max-output", str(args.max_output)])
    if args.frac_greedy is not None:
        command.extend(["--frac-greedy", str(args.frac_greedy)])

    try:
        result = subprocess.run(command)
    finally:
        try:
            filtered_prompts_path.unlink()
        except FileNotFoundError:
            pass

    if result.returncode != 0:
        raise SystemExit(result.returncode)

    print(f"Generated workload at {target_path}")


if __name__ == "__main__":
    main()
