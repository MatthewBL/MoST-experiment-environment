import argparse
import os
import subprocess
from pathlib import Path

from experiment_automation import set_process_env_for_run
from fmperf.utils.constants import REQUESTS_DIR, REQUESTS_FILENAME


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a workload file for a specific input token range"
    )
    parser.add_argument("min_input", type=int, help="Minimum input tokens")
    parser.add_argument("max_input", type=int, help="Maximum input tokens")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite existing workload file")
    parser.add_argument("--min-output", type=int, default=None, help="Minimum output tokens (defaults to min_input)")
    parser.add_argument("--max-output", type=int, default=None, help="Maximum output tokens (defaults to max_input)")
    parser.add_argument("--req-min", type=int, default=1, help="REQ_MIN value to store in the environment")
    parser.add_argument("--prompts-file", help="Optional prompts JSONL file to pass to generate-input")
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
    min_output = args.min_output if args.min_output is not None else args.min_input
    max_output = args.max_output if args.max_output is not None else args.max_input
    _validate_ranges(min_output, max_output, "Output")

    if args.prompts_file:
        os.environ["PROMPTS_FILE"] = args.prompts_file

    set_process_env_for_run(
        args.req_min,
        input_interval=(args.min_input, args.max_input),
        output_interval=(min_output, max_output),
    )

    requests_dir = Path(REQUESTS_DIR)
    requests_dir.mkdir(parents=True, exist_ok=True)
    target_name = os.environ.get("REQUESTS_FILENAME", REQUESTS_FILENAME)
    target_path = requests_dir / target_name

    if target_path.exists() and not args.overwrite:
        print(f"Existing workload found at {target_path}. Use --overwrite to regenerate.")
        return

    os.environ["OVERWRITE"] = "true" if args.overwrite else "false"

    command = ["python", "-u", "-m", "fmperf.loadgen.generate-input"]
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    print(f"Generated workload at {target_path}")


if __name__ == "__main__":
    main()
