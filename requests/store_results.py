import os
import re
import csv
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Best-effort helpers to enrich results.csv with requested fields
def _read_env_value(env_path: Path, key: str, default: str = "") -> str:
    try:
        if env_path.exists():
            with env_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith(key + "="):
                        return line.split("=", 1)[1]
    except Exception:
        pass
    return default

def _find_slurm_log(job_id: str | None) -> tuple[str | None, str | None]:
    """Return (job_id, slurm_log_path) if found.
    Strategy:
    - Prefer explicit job: look for slurm-<jobid>.out
      in likely roots: CWD, script dir, and script dir's parent (project root).
    - If not found, fallback to most recent slurm-*.out among those roots
      and their parents up to a few levels.
    """
    job = job_id or os.environ.get("SLURM_JOB_ID")

    candidates: list[Path] = []

    # Build a robust search set of directories
    roots: list[Path] = []
    try:
        cwd = Path.cwd()
        roots.extend([cwd, *cwd.parents[:4]])
    except Exception:
        pass
    try:
        here = Path(__file__).resolve()
        script_dir = here.parent
        roots.append(script_dir)
        # If this file lives in 'requests/', the project root is its parent
        if script_dir.name.lower() == "requests":
            roots.append(script_dir.parent)
        roots.extend([*script_dir.parents[:4]])
    except Exception:
        pass

    # De-duplicate while preserving order
    seen = set()
    unique_roots: list[Path] = []
    for r in roots:
        try:
            rp = r.resolve()
        except Exception:
            rp = r
        if rp not in seen:
            seen.add(rp)
            unique_roots.append(rp)

    # First pass: exact slurm-<job>.out in likely dirs
    if job:
        try:
            for up in unique_roots:
                p = up / f"slurm-{job}.out"
                if p.exists():
                    return job, str(p)
        except Exception:
            pass

    # Fallback: collect all slurm-*.out files in the search dirs
    try:
        for up in unique_roots:
            for entry in up.glob("slurm-*.out"):
                candidates.append(entry)
    except Exception:
        pass

    if candidates:
        try:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            picked = candidates[0]
            m = re.search(r"slurm-(\d+)\.out$", picked.name)
            jid = m.group(1) if m else (job or None)
            return jid, str(picked)
        except Exception:
            pass

    return job, None

def _extract_model_from_slurm(slurm_path: str | None) -> str:
    """Extract model name from Slurm log.
    Handles both free-text lines (e.g., 'MODEL: foo') and JSON lines like
    '"model": "google/gemma-7b"'. Returns an empty string if not found.
    """
    if not slurm_path:
        return ""
    # Put JSON-aware pattern first to match the provided log format
    patterns = [
        r"\"model\"\s*:\s*\"([^\"]+)\"",  # JSON key: "model": "..."
        r"\bMODEL\s*[:=]\s*([^\s,]+)",
        r"\bModel used\s*[:=]\s*(.+)",
        r"\bUsing model\s*[:=]?\s*(.+)",
        r"\b--model\s+([^\s]+)",
        r"\bmodel\s*[:=]\s*([^\s,]+)",
    ]
    try:
        with open(slurm_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                for pat in patterns:
                    m = re.search(pat, s, re.IGNORECASE)
                    if m:
                        val = m.group(1).strip()
                        # Sanitize quotes/padding/nulls
                        val = val.replace("\x00", "").strip().strip('"\'')
                        return val
    except Exception:
        pass
    return ""

def _extract_median_tokens_from_log(slurm_path: str | None) -> str | None:
    """Parse the latest 'Median tokens per response: <value>' printed by experiment_automation.
    Prefer the last occurrence in the Slurm log; return None if unavailable.
    """
    if not slurm_path:
        return None
    try:
        last_val: str | None = None
        pat = re.compile(r"Median tokens per response:\s*([0-9]+(?:\.[0-9]+)?)")
        with open(slurm_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pat.search(line)
                if m:
                    last_val = m.group(1)
        return last_val
    except Exception:
        return None

def _read_prompt_info(sample_path: Path) -> tuple[str | None, str | None]:
    """Return (prompt_text, prompt_token_count) from sample_requests.json if available."""
    if not sample_path.exists():
        alt = Path("..") / sample_path.name
        if not alt.exists():
            return None, None
        sample_path = alt
    try:
        data = json.loads(sample_path.read_text(encoding="utf-8"))
        items = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # try common container keys
            for key in ("requests", "data", "items"):
                if isinstance(data.get(key), list):
                    items = data[key]
                    break
            if not items:
                items = [data]

        for it in items:
            if not isinstance(it, dict):
                continue
            # prompt text keys in order of likelihood
            for tkey in ("prompt", "text", "inputs", "input", "query"):
                if isinstance(it.get(tkey), str) and it.get(tkey).strip():
                    prompt_text = it.get(tkey).strip()
                    break
            else:
                prompt_text = None

            # token count keys seen in generators
            for k in ("prompt_token_count", "input_token_count", "prompt_len", "input_tokens"):
                v = it.get(k)
                if isinstance(v, (int, float, str)):
                    return prompt_text, str(v)
            # If not present, still return text (count unknown)
            if prompt_text:
                return prompt_text, None
        return None, None
    except Exception:
        return None, None

def _to_float(value) -> float | None:
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def _to_int(value) -> int | None:
    fv = _to_float(value)
    if fv is None:
        return None
    try:
        return int(fv)
    except Exception:
        return None


def _format_number(value: float | int | None) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except Exception:
        return ""
    if math.isfinite(v) and float(v).is_integer():
        return str(int(v))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def _population_variance(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _percentile(values_sorted: list[float], percentile: float) -> float | None:
    if not values_sorted:
        return None
    if len(values_sorted) == 1:
        return values_sorted[0]
    p = max(0.0, min(100.0, float(percentile)))
    pos = (p / 100.0) * (len(values_sorted) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values_sorted[lo]
    frac = pos - lo
    return values_sorted[lo] + (values_sorted[hi] - values_sorted[lo]) * frac


def _serialize_percentiles(values: list[float]) -> str:
    if not values:
        return ""
    vals = sorted(values)
    pts = {
        "p50": _percentile(vals, 50),
        "p75": _percentile(vals, 75),
        "p90": _percentile(vals, 90),
        "p95": _percentile(vals, 95),
        "p99": _percentile(vals, 99),
    }
    clean = {k: float(_format_number(v)) for k, v in pts.items() if v is not None}
    return json.dumps(clean, separators=(",", ":"), ensure_ascii=True)


def _load_prompt_tokens_by_sample_idx() -> dict[int, float]:
    out: dict[int, float] = {}
    req_path = _find_requests_file()
    if not req_path:
        return out
    try:
        payload = json.loads(req_path.read_text(encoding="utf-8"))
    except Exception:
        return out

    if isinstance(payload, dict):
        items = payload.get("requests") or payload.get("data") or payload.get("items") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        raw = it.get("prompt_token_count")
        if raw is None:
            raw = it.get("input_token_count")
        if raw is None:
            raw = it.get("prompt_len")
        if raw is None:
            raw = it.get("input_tokens")
        if raw is None and isinstance(it.get("config"), dict):
            raw = it["config"].get("in_tokens")
        fv = _to_float(raw)
        if fv is not None and fv >= 0:
            out[idx] = fv
    return out


def _parse_numeric_bounds(min_value: str | None, max_value: str | None) -> tuple[float | None, float | None]:
    lo = _to_float(min_value)
    hi = _to_float(max_value)
    if lo is None and hi is None:
        return None, None
    if lo is None:
        lo = hi
    if hi is None:
        hi = lo
    if lo is not None and hi is not None and lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _compute_request_token_stats(
    results_path: Path,
    expected_min_output: float | None,
    expected_max_output: float | None,
) -> dict[str, str | None]:
    """Compute output/input token statistics and interval compliance from results.json."""
    out: dict[str, str | None] = {
        "median_response_tokens": None,
        "total_requests": None,
        "success_rate": None,
        "responses_within_interval": None,
        "responses_outside_interval": None,
        "avg_tokens_per_request": None,
        "avg_tokens_per_response": None,
        "input_token_variance": None,
        "output_token_variance": None,
        "input_token_percentiles": None,
        "output_token_percentiles": None,
        "request_total_token_percentiles": None,
    }

    # success rate from output.csv if present
    try:
        out_csv = Path("output.csv")
        if out_csv.exists():
            with out_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
                if row:
                    col = None
                    for c in ("success_rate", "success_ratio", "success", "pass_rate", "accuracy"):
                        if c in row:
                            col = c
                            break
                    if not col:
                        for k in row.keys():
                            if "success" in k.lower():
                                col = k
                                break
                    if col:
                        out["success_rate"] = str(row.get(col, "") or "")
    except Exception:
        pass

    request_entries: dict[tuple[int | None, int | None], dict[str, float | int | None]] = {}

    # Compute request-level output lengths from results.json if available
    try:
        if results_path.exists():
            data = json.loads(results_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("results"), list):
                items = data["results"]
            elif isinstance(data, list):
                items = data
            else:
                items = []

            current_rid = -1
            for it in items:
                if not isinstance(it, dict):
                    continue

                rid = _to_int(it.get("request_idx"))
                wid = _to_int(it.get("worker_idx"))
                rsi = _to_int(it.get("response_idx"))
                n_tokens = _to_float(it.get("n_tokens"))
                sample_idx = _to_int(it.get("sample_idx"))

                # Fallback request derivation when request_idx is missing.
                if rid is None:
                    if rsi is not None and rsi == 0:
                        current_rid += 1
                    if current_rid < 0:
                        current_rid = 0
                    rid = current_rid

                key = (wid, rid)
                if key not in request_entries:
                    request_entries[key] = {
                        "token_sum": 0.0,
                        "max_response_idx": -1,
                        "sample_idx": sample_idx,
                    }

                rec = request_entries[key]
                if sample_idx is not None and rec.get("sample_idx") is None:
                    rec["sample_idx"] = sample_idx
                if n_tokens is not None and n_tokens > 0:
                    rec["token_sum"] = float(rec.get("token_sum", 0.0) or 0.0) + n_tokens
                if rsi is not None:
                    prev = int(rec.get("max_response_idx", -1) or -1)
                    if rsi > prev:
                        rec["max_response_idx"] = rsi
    except Exception:
        pass

    output_tokens_per_request: list[float] = []
    request_sample_indices: list[int | None] = []
    if request_entries:
        for rec in request_entries.values():
            token_sum = _to_float(rec.get("token_sum")) or 0.0
            max_idx = _to_int(rec.get("max_response_idx"))
            if token_sum > 0:
                out_len = token_sum
            elif max_idx is not None and max_idx >= 0:
                out_len = float(max_idx + 1)
            else:
                out_len = 0.0
            output_tokens_per_request.append(out_len)
            request_sample_indices.append(_to_int(rec.get("sample_idx")))

    if output_tokens_per_request:
        sorted_out = sorted(output_tokens_per_request)
        out["median_response_tokens"] = _format_number(_percentile(sorted_out, 50))
        out["total_requests"] = str(len(output_tokens_per_request))
        out["avg_tokens_per_response"] = _format_number(sum(output_tokens_per_request) / len(output_tokens_per_request))
        out["output_token_variance"] = _format_number(_population_variance(output_tokens_per_request))
        out["output_token_percentiles"] = _serialize_percentiles(output_tokens_per_request)

        if expected_min_output is not None or expected_max_output is not None:
            within = 0
            for out_len in output_tokens_per_request:
                ok_lo = expected_min_output is None or out_len >= expected_min_output
                ok_hi = expected_max_output is None or out_len <= expected_max_output
                if ok_lo and ok_hi:
                    within += 1
            outside = len(output_tokens_per_request) - within
            out["responses_within_interval"] = str(within)
            out["responses_outside_interval"] = str(outside)

    # If results.json unavailable, attempt to derive total from first/second_half.csv
    if out["total_requests"] is None:
        try:
            fh = Path("first_half.csv")
            sh = Path("second_half.csv")
            count = 0
            for p in (fh, sh):
                if p.exists():
                    with p.open("r", encoding="utf-8", newline="") as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                        if rows:
                            count += max(0, len(rows) - 1)
            if count:
                out["total_requests"] = str(count)
        except Exception:
            pass

    prompt_tokens_by_idx = _load_prompt_tokens_by_sample_idx()
    if request_sample_indices and prompt_tokens_by_idx:
        input_tokens_per_request: list[float] = []
        total_tokens_per_request: list[float] = []
        for i, sample_idx in enumerate(request_sample_indices):
            if sample_idx is None:
                continue
            input_tokens = prompt_tokens_by_idx.get(sample_idx)
            if input_tokens is None:
                continue
            input_tokens_per_request.append(input_tokens)
            if i < len(output_tokens_per_request):
                total_tokens_per_request.append(input_tokens + output_tokens_per_request[i])

        if input_tokens_per_request:
            out["input_token_variance"] = _format_number(_population_variance(input_tokens_per_request))
            out["input_token_percentiles"] = _serialize_percentiles(input_tokens_per_request)
        if total_tokens_per_request:
            out["avg_tokens_per_request"] = _format_number(sum(total_tokens_per_request) / len(total_tokens_per_request))
            out["request_total_token_percentiles"] = _serialize_percentiles(total_tokens_per_request)

    # Fallback for avg tokens/request if input tokens are not available.
    if out["avg_tokens_per_request"] is None:
        out["avg_tokens_per_request"] = out["avg_tokens_per_response"]

    return out

def _parse_range(value: str | None) -> tuple[str | None, str | None]:
    """Parse a token range string like '32-64' or '32' into (min,max) strings.
    Returns (None, None) if input is falsy.
    """
    if value is None:
        return None, None
    s = str(value).strip()
    if not s:
        return None, None
    if '-' in s:
        a, b = s.split('-', 1)
        return a.strip() or None, b.strip() or None
    return s, s

def _compute_median_prompt_tokens() -> str | None:
    """Compute median of prompt_token_count from the generated requests file.

    Falls back to None if unavailable or unparseable.
    """
    try:
        req_path = _find_requests_file()
        if not req_path:
            return None
        payload = json.loads(req_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = payload.get("requests") or payload.get("data") or payload.get("items") or []
        elif isinstance(payload, list):
            items = payload
        else:
            items = []

        vals: list[float] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            v = it.get("prompt_token_count")
            if isinstance(v, (int, float)):
                try:
                    vals.append(float(v))
                except Exception:
                    continue
            elif isinstance(v, str) and v.strip():
                try:
                    vals.append(float(v))
                except Exception:
                    continue

        if not vals:
            return None
        vals.sort()
        n = len(vals)
        if n % 2 == 1:
            return f"{vals[n//2]:.0f}" if vals[n//2].is_integer() else f"{vals[n//2]:.3f}"
        m = (vals[n//2 - 1] + vals[n//2]) / 2.0
        return f"{m:.0f}" if m.is_integer() else f"{m:.3f}"
    except Exception:
        return None

def _find_requests_file() -> Optional[Path]:
    """Locate the requests JSON file generated by generate-input.

    Strategy:
    - Use REQUESTS_FILENAME/REQUESTS_DIR from env if available.
    - Try common relative locations from current dir and its parent.
    - Return the first existing Path, else None.
    """
    fname = os.environ.get("REQUESTS_FILENAME", None)
    rdir = os.environ.get("REQUESTS_DIR", None)

    candidates: list[Path] = []

    # Direct filename in CWD
    if fname:
        candidates.append(Path(fname))
    # REQUESTS_DIR + filename
    if fname and rdir:
        candidates.append(Path(rdir) / fname)
    # Parent dir + filename
    if fname:
        candidates.append(Path("..") / fname)
    # Parent dir + REQUESTS_DIR + filename
    if fname and rdir:
        candidates.append(Path("..") / rdir / fname)

    # Common defaults
    candidates.extend([
        Path("requests.json"),
        Path("..") / "requests.json",
        Path("requests") / "requests.json",
        Path("..") / "requests" / "requests.json",
    ])

    for p in candidates:
        try:
            if p.exists():
                return p
        except Exception:
            continue
    return None

def _write_prompts_csv(full_dir_path: str) -> None:
    """Create prompts.csv with unique prompts used in this run.

    Reads the requests JSON produced by generate-input (cases with
    'prompt_text' and optional 'prompt_token_count') and writes a CSV
    containing each unique prompt aggregated with occurrences.

    Occurrences reflect how many times the prompt appeared in the
    requests payload (i.e., how many requests were sent using that
    prompt), independent of response success.
    """
    try:
        req_path = _find_requests_file()
        if not req_path:
            print("Warning: requests file not found; skipping prompts.csv generation")
            return

        try:
            payload = json.loads(req_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: unable to parse requests from {req_path}: {e}")
            return

        # Normalize to list of items
        if isinstance(payload, dict):
            items = payload.get("requests") or payload.get("data") or payload.get("items") or []
        elif isinstance(payload, list):
            items = payload
        else:
            items = []

        # Build ordered lists of (prompt_text, prompt_token_count) per request index
        prompts_by_index: list[tuple[str, Optional[str]]] = []
        for it in items:
            if isinstance(it, dict):
                txt = it.get("prompt_text")
                tok = it.get("prompt_token_count")
                if isinstance(txt, str) and txt.strip():
                    tok_str: Optional[str] = None
                    if isinstance(tok, (int, float)):
                        tok_str = str(int(tok)) if isinstance(tok, int) or float(tok).is_integer() else str(tok)
                    elif isinstance(tok, str) and tok.strip():
                        tok_str = tok.strip()
                    prompts_by_index.append((txt, tok_str))

        if not prompts_by_index:
            print("Warning: no prompts found in requests payload; skipping prompts.csv generation")
            return

        # Aggregate by prompt text: count occurrences
        from collections import OrderedDict
        agg: "OrderedDict[str, dict]" = OrderedDict()
        for idx, (txt, tok) in enumerate(prompts_by_index):
            if txt not in agg:
                agg[txt] = {"count": 0, "token_count": tok}
            entry = agg[txt]
            entry["count"] += 1
            if entry["token_count"] in (None, "") and tok not in (None, ""):
                entry["token_count"] = tok

        out_path = os.path.join(full_dir_path, "prompts.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["PROMPT_TEXT", "PROMPT_TOKEN_COUNT", "OCCURRENCES"])
            for txt, entry in agg.items():
                safe_txt = txt.replace("\n", " ").strip()
                w.writerow([
                    safe_txt,
                    "" if entry["token_count"] in (None, "") else str(entry["token_count"]),
                    str(entry["count"]),
                ])
        print(f"Created prompts.csv in {full_dir_path} with {len(agg)} unique prompts")
    except Exception as e:
        print(f"Warning: failed to create prompts.csv: {e}")


_TIMESTAMP_FORMATS: tuple[str, ...] = (
    "%d/%m/%Y  %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y%m%dT%H%M%S",
)


def _parse_timestamp_string(value: str | None) -> datetime | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _extract_timestamp_from_csv(path: Path) -> tuple[datetime | None, str | None]:
    if not path.exists():
        return None, None
    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            next(reader, None)  # header
            row = next(reader, None)
    except Exception:
        return None, None
    if not row:
        return None, None
    raw_value = row[0] if row else None
    return _parse_timestamp_string(raw_value), raw_value


def _sanitize_dir_component(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


def _derive_directory_name() -> tuple[str, str]:
    candidates = [
        Path("first_half.csv"),
        Path("second_half.csv"),
        Path("output.csv"),
    ]
    for candidate in candidates:
        dt, raw = _extract_timestamp_from_csv(candidate)
        if dt:
            return dt.strftime("%Y-%m-%d_%H-%M-%S"), candidate.name
        if raw:
            return _sanitize_dir_component(raw), candidate.name
    return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S"), "current_time"

def main():
    try:
        # Check if required CSV files exist
        if not os.path.exists("output.csv"):
            print("Error: output.csv not found")
            return

        missing_halves = [name for name in ("first_half.csv", "second_half.csv") if not os.path.exists(name)]
        if missing_halves:
            print(f"Warning: {', '.join(missing_halves)} not found; proceeding without filtered halves.")
        
        # Determine directory name based on the first available timestamp source
        dir_name, timestamp_source = _derive_directory_name()
        print(f"Using timestamp from {timestamp_source} to create directory: {dir_name}")
        
        # Parse CLI arguments from experiment_automation.py.
        # Supported formats:
        # - Compact (current): model, stage, parent_dir, in_range, out_range, req_min, evaluation, [median], [prompt_token_count]
        # - Legacy:            model, gpus, cpus, node, stage, parent_dir, in_range, out_range, req_min, evaluation, [median], [prompt_token_count]
        args = sys.argv[1:]
        is_compact_cli = len(args) >= 7 and ('_' in str(args[2]) or '/' in str(args[2]) or '\\' in str(args[2]))

        parent_dir = None
        model = os.environ.get('MODEL', '')
        stage = os.environ.get('STAGE', '')
        
        # Resolve tokens (min/max), REQ_MIN, EVALUATION, MEDIAN: prefer CLI args, then env/log, then .env
        min_input_tokens = ''
        max_input_tokens = ''
        min_output_tokens = ''
        max_output_tokens = ''
        req_min = ''
        evaluation_flag = ''
        median_cli = ''
        prompt_token_count_cli = ''  # deprecated: will use median over prompts

        if is_compact_cli:
            model = args[0]
            stage = args[1]
            parent_dir = args[2]

            in_range_str = args[3]
            out_range_str = args[4]
            mi, ma = _parse_range(in_range_str)
            mo, moa = _parse_range(out_range_str)
            min_input_tokens = mi or ''
            max_input_tokens = ma or ''
            min_output_tokens = mo or ''
            max_output_tokens = moa or ''
            req_min = args[5]
            evaluation_flag = args[6]
            if len(args) >= 8:
                median_cli = args[7]
            if len(args) >= 9:
                prompt_token_count_cli = args[8]
        elif len(args) >= 10:
            # Backward-compatible parsing for legacy positional arguments.
            model = args[0]
            stage = args[4]
            parent_dir = args[5]

            in_range_str = args[6]
            out_range_str = args[7]
            mi, ma = _parse_range(in_range_str)
            mo, moa = _parse_range(out_range_str)
            min_input_tokens = mi or ''
            max_input_tokens = ma or ''
            min_output_tokens = mo or ''
            max_output_tokens = moa or ''
            req_min = args[8]
            evaluation_flag = args[9]
            if len(args) >= 11:
                median_cli = args[10]
            if len(args) >= 12:
                prompt_token_count_cli = args[11]
        else:
            # Environment variables set in-process by experiment_automation.py
            min_input_tokens = os.environ.get('MIN_INPUT_TOKENS', '')
            max_input_tokens = os.environ.get('MAX_INPUT_TOKENS', '')
            min_output_tokens = os.environ.get('MIN_OUTPUT_TOKENS', '')
            max_output_tokens = os.environ.get('MAX_OUTPUT_TOKENS', '')
            req_min = os.environ.get('REQ_MIN', '')
            evaluation_flag = os.environ.get('EVALUATION', '')
            prompt_token_count_cli = os.environ.get('PROMPT_TOKEN_COUNT', '')

        # Create the full directory path
        if parent_dir:
            full_dir_path = os.path.join(parent_dir, dir_name)
            os.makedirs(parent_dir, exist_ok=True)  # Ensure parent directory exists
        else:
            full_dir_path = dir_name
        
        os.makedirs(full_dir_path, exist_ok=True)
        print(f"Created directory: {full_dir_path}")

        # Fallback to .env only if still missing
        if (min_input_tokens == '' or max_input_tokens == '' or min_output_tokens == '' or max_output_tokens == '' or req_min == '') and os.path.exists('../.env'):
            with open('../.env', 'r') as env_file:
                for line in env_file:
                    line = line.strip()
                    if min_input_tokens == '' and line.startswith('MIN_INPUT_TOKENS='):
                        min_input_tokens = line.split('=', 1)[1]
                    elif max_input_tokens == '' and line.startswith('MAX_INPUT_TOKENS='):
                        max_input_tokens = line.split('=', 1)[1]
                    elif min_output_tokens == '' and line.startswith('MIN_OUTPUT_TOKENS='):
                        min_output_tokens = line.split('=', 1)[1]
                    elif max_output_tokens == '' and line.startswith('MAX_OUTPUT_TOKENS='):
                        max_output_tokens = line.split('=', 1)[1]
                    elif req_min == '' and line.startswith('REQ_MIN='):
                        req_min = line.split('=', 1)[1]
                    elif evaluation_flag == '' and line.startswith('EVALUATION='):
                        evaluation_flag = line.split('=', 1)[1]

        expected_min_output, expected_max_output = _parse_numeric_bounds(min_output_tokens, max_output_tokens)
        
        # Evaluation: use explicit flag from CLI/env; also attempt to read success rate from output.csv
        evaluation = (evaluation_flag or '').strip()
        # Normalize evaluation to TRUE/FALSE if possible
        if evaluation.lower() in {'true', '1', 'yes'}:
            evaluation = 'TRUE'
        elif evaluation.lower() in {'false', '0', 'no'}:
            evaluation = 'FALSE'

        success_rate = ''
        try:
            with open("output.csv", 'r', encoding="utf-8") as file:
                reader = csv.DictReader(file)
                row = next(reader, None)
                if row:
                    for k in row.keys():
                        if 'success' in k.lower():
                            success_rate = str(row.get(k) or '')
                            break
        except Exception:
            pass

        # Duration from .env
        duration = _read_env_value(Path('..') / '.env', 'DURATION', '')

        # Endpoint URL used for this run
        url = (os.environ.get('URL') or '').strip()
        if not url:
            url = _read_env_value(Path('..') / '.env', 'URL', '')

        # Additive proportion telemetry (JSON strings keyed by token interval label)
        additive_expected_proportions = (os.environ.get('ADDITIVE_EXPECTED_PROPORTIONS') or '').strip()
        additive_true_proportions = (os.environ.get('ADDITIVE_TRUE_PROPORTIONS') or '').strip()

        # Prompt token count: use median across prompts in requests
        prompt_token_count = _compute_median_prompt_tokens() or (prompt_token_count_cli or '').strip()

        # Job ID and Slurm model extraction
        job_id_env = os.environ.get('SLURM_JOB_ID')
        job_id, slurm_path = _find_slurm_log(job_id_env)
        model_from_slurm = _extract_model_from_slurm(slurm_path)

        # Median response tokens: prefer CLI-provided value from experiment_automation;
        # fall back to log-parsed value, then computed from results.json/output.csv
        median_resp_tokens = median_cli if (median_cli and str(median_cli).strip() != '') else None
        if median_resp_tokens is None:
            log_median = _extract_median_tokens_from_log(slurm_path)
            median_resp_tokens = log_median if log_median else None
        stats = _compute_request_token_stats(Path('results.json'), expected_min_output, expected_max_output)
        comp_median = stats.get("median_response_tokens")
        total_requests = stats.get("total_requests")
        sr_from_results = stats.get("success_rate")
        if median_resp_tokens is None:
            median_resp_tokens = comp_median
        if not success_rate and sr_from_results:
            success_rate = sr_from_results

        responses_within_interval = stats.get("responses_within_interval")
        responses_outside_interval = stats.get("responses_outside_interval")
        avg_tokens_per_request = stats.get("avg_tokens_per_request")
        avg_tokens_per_response = stats.get("avg_tokens_per_response")
        input_token_variance = stats.get("input_token_variance")
        output_token_variance = stats.get("output_token_variance")
        input_token_percentiles = stats.get("input_token_percentiles")
        output_token_percentiles = stats.get("output_token_percentiles")
        request_total_token_percentiles = stats.get("request_total_token_percentiles")

        # Create new CSV file
        new_csv_path = os.path.join(full_dir_path, "results.csv")
        with open(new_csv_path, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write header with requested fields (remove GPUS/CPUS)
            writer.writerow([
                "MODEL_USED",
                "MIN_INPUT_TOKENS", "MAX_INPUT_TOKENS",
                "MIN_OUTPUT_TOKENS", "MAX_OUTPUT_TOKENS",
                "REQ_MIN", "EVALUATION",
                "DURATION", "URL", "TOTAL_REQUESTS", "SUCCESS_RATE", "MEDIAN_PROMPT_TOKENS",
                "MEDIAN_RESPONSE_TOKENS", "JOB_ID", "STAGE",
                "RESPONSES_WITHIN_EXPECTED_INTERVAL", "RESPONSES_OUTSIDE_EXPECTED_INTERVAL",
                "AVG_TOKENS_PER_REQUEST", "AVG_TOKENS_PER_RESPONSE",
                "INPUT_TOKEN_VARIANCE", "OUTPUT_TOKEN_VARIANCE",
                "INPUT_TOKEN_PERCENTILES", "OUTPUT_TOKEN_PERCENTILES", "REQUEST_TOTAL_TOKEN_PERCENTILES",
                "ADDITIVE_EXPECTED_PROPORTIONS", "ADDITIVE_TRUE_PROPORTIONS"
            ])
            # Write data row
            writer.writerow([
                model_from_slurm or model,
                min_input_tokens,
                max_input_tokens,
                min_output_tokens,
                max_output_tokens,
                req_min,
                evaluation,
                duration,
                url,
                total_requests or '',
                success_rate or '',
                prompt_token_count or '',
                median_resp_tokens or '',
                job_id or '',
                stage,
                responses_within_interval or '',
                responses_outside_interval or '',
                avg_tokens_per_request or '',
                avg_tokens_per_response or '',
                input_token_variance or '',
                output_token_variance or '',
                input_token_percentiles or '',
                output_token_percentiles or '',
                request_total_token_percentiles or '',
                additive_expected_proportions,
                additive_true_proportions
            ])
        
        print(f"Created results.csv in {full_dir_path}")
        
        # Create prompts.csv with unique prompts used
        _write_prompts_csv(full_dir_path)
        
        # Move CSV files to the new directory when available
        for csv_name in ("output.csv", "first_half.csv", "second_half.csv"):
            if os.path.exists(csv_name):
                shutil.move(csv_name, os.path.join(full_dir_path, csv_name))
            else:
                print(f"Warning: {csv_name} not found; skipping move.")
        # Keep a copy of results.json in the working directory for downstream readers
        shutil.copyfile("results.json", os.path.join(full_dir_path, "results.json"))
        
        print("Moved CSV files and copied results.json to the directory")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()