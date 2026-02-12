import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

from dotenv import load_dotenv

from fmperf.utils.constants import REQUESTS_DIR, REQUESTS_FILENAME

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_FILE = REPO_ROOT / "oasst_roots_en_max1000_tokens.jsonl"
TOKEN_FIELD = "gpt2_token_count"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter prompts within a token window")
    parser.add_argument("--min-input", dest="min_input", type=int, default=None, help="Minimum allowed gpt2_token_count")
    parser.add_argument("--max-input", dest="max_input", type=int, default=None, help="Maximum allowed gpt2_token_count")
    parser.add_argument(
        "--prompts-file",
        dest="prompts_file",
        type=Path,
        default=None,
        help="Path to the JSONL dataset to scan",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=Path,
        default=None,
        help="Destination JSON file for the filtered prompts",
    )
    return parser


def _resolve_bound(cli_value: Optional[int], env_key: str) -> int:
    if cli_value is not None:
        return cli_value
    env_value = os.environ.get(env_key)
    if env_value is None:
        raise ValueError(f"Missing required value for {env_key}")
    try:
        return int(env_value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {env_key}: {env_value}") from exc


def _load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping malformed JSON on line {line_number}: {exc}")


def _filter_prompts(records: Iterable[dict], min_tokens: int, max_tokens: int) -> List[dict]:
    kept = []
    for record in records:
        token_value = record.get(TOKEN_FIELD)
        if token_value is None:
            continue
        try:
            token_count = int(token_value)
        except (TypeError, ValueError):
            continue
        if min_tokens <= token_count <= max_tokens:
            kept.append(record)
    return kept


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    min_tokens = _resolve_bound(args.min_input, "MIN_INPUT_TOKENS")
    max_tokens = _resolve_bound(args.max_input, "MAX_INPUT_TOKENS")
    if min_tokens <= 0 or max_tokens <= 0:
        raise ValueError("Token bounds must be positive integers")
    if min_tokens > max_tokens:
        raise ValueError("Minimum input tokens cannot exceed the maximum")

    prompts_file = args.prompts_file or Path(os.environ.get("PROMPTS_FILE", DEFAULT_PROMPTS_FILE))
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    output_path = args.output
    if output_path is None:
        filename = os.environ.get("REQUESTS_FILENAME", REQUESTS_FILENAME)
        output_path = Path(REQUESTS_DIR) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompts = list(_load_jsonl(prompts_file))
    filtered = _filter_prompts(prompts, min_tokens, max_tokens)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(filtered, handle, ensure_ascii=False, indent=2)

    print(f">> Read {len(prompts)} prompts from {prompts_file}")
    print(f">> Kept {len(filtered)} prompts with {TOKEN_FIELD} between {min_tokens} and {max_tokens}")
    print(f">> Wrote filtered prompts to {output_path}")


if __name__ == "__main__":
    main()
