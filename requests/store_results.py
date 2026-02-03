import os
import re
import csv
import json
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

def _compute_median_response_tokens(results_path: Path) -> tuple[str | None, str | None, str | None]:
    """Compute (median_response_tokens, total_requests, success_rate) from results/output CSV/JSON.
    - total_requests: unique requests inferred from response_idx resets or request_idx values.
    - success_rate: best-effort read from output.csv if a column with 'success' exists.
    """
    median_tokens = None
    total_requests = None
    success_rate = None

    # success rate from output.csv if present
    try:
        out_csv = Path("output.csv")
        if out_csv.exists():
            with out_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
                if row:
                    # direct median columns if present
                    for k in row.keys():
                        kl = k.lower()
                        if "median" in kl and "token" in kl:
                            try:
                                val = str(row.get(k, ""))
                                if val:
                                    median_tokens = val
                            except Exception:
                                pass
                    # prioritize fields named like success_rate/ratio
                    def find_col(cols: list[str]) -> str | None:
                        for c in cols:
                            if c in row:
                                return c
                        return None
                    # try exact/common names
                    col = find_col([
                        "success_rate", "success_ratio", "success", "pass_rate", "accuracy"
                    ])
                    if not col:
                        # case-insensitive contains 'success'
                        for k in row.keys():
                            if "success" in k.lower():
                                col = k
                                break
                    if col:
                        success_rate = str(row.get(col, ""))
    except Exception:
        pass

    # Compute tokens and requests from results.json if available
    try:
        if results_path.exists():
            data = json.loads(results_path.read_text(encoding="utf-8"))
            # Normalize iterable of events or responses
            if isinstance(data, dict) and isinstance(data.get("results"), list):
                items = data["results"]
            elif isinstance(data, list):
                items = data
            else:
                items = []

            # Track per-request max response_idx
            max_resp_idx: dict[tuple[int | None, int | None], int] = {}
            # Fallback request sequencing if request_idx missing
            current_rid = -1

            for it in items:
                if not isinstance(it, dict):
                    continue
                rid = it.get("request_idx")
                wid = it.get("worker_idx")
                rsi = it.get("response_idx")

                # derive rid when missing: response_idx==0 indicates new request
                if rid is None:
                    if isinstance(rsi, (int, float)) and int(rsi) == 0:
                        current_rid += 1
                    if current_rid < 0:
                        current_rid = 0
                    rid = current_rid

                key = (int(wid) if isinstance(wid, (int, float, str)) and str(wid).isdigit() else None,
                       int(rid) if isinstance(rid, (int, float, str)) and str(rid).lstrip("-+").isdigit() else None)

                if isinstance(rsi, (int, float, str)):
                    try:
                        val = int(float(rsi))
                        prev = max_resp_idx.get(key, -1)
                        if val > prev:
                            max_resp_idx[key] = val
                    except Exception:
                        pass

            if max_resp_idx:
                # token count per request = max response_idx + 1
                token_counts = [v + 1 for v in max_resp_idx.values() if isinstance(v, int) and v >= 0]
                if token_counts:
                    token_counts.sort()
                    n = len(token_counts)
                    if n % 2 == 1:
                        median_tokens = str(token_counts[n // 2])
                    else:
                        median_tokens = str((token_counts[n // 2 - 1] + token_counts[n // 2]) / 2)
                total_requests = str(len(max_resp_idx))
    except Exception:
        pass

    # If results.json unavailable, attempt to derive total from first/second_half.csv
    if total_requests is None:
        try:
            fh = Path("first_half.csv")
            sh = Path("second_half.csv")
            count = 0
            for p in (fh, sh):
                if p.exists():
                    with p.open("r", encoding="utf-8", newline="") as f:
                        reader = csv.reader(f)
                        rows = list(reader)
                        # naive: subtract header
                        if rows:
                            count += max(0, len(rows) - 1)
            if count:
                total_requests = str(count)
        except Exception:
            pass

    return median_tokens, total_requests, success_rate

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

def main():
    try:
        # Check if required CSV files exist
        if not os.path.exists("output.csv"):
            print("Error: output.csv not found")
            return
        
        if not os.path.exists("first_half.csv"):
            print("Error: first_half.csv not found")
            return
            
        if not os.path.exists("second_half.csv"):
            print("Error: second_half.csv not found")
            return
        
        # Read date from first_half.csv and create directory
        with open("first_half.csv", 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)
            if len(rows) < 2:
                print("Error: first_half.csv doesn't have at least 2 rows")
                return
            
            date_str = rows[1][0]  # First column of second row
            
            # Parse and format the date for directory name
            try:
                # Try parsing with the format you provided
                date_obj = datetime.strptime(date_str, '%d/%m/%Y  %H:%M:%S')
                dir_name = date_obj.strftime('%Y-%m-%d_%H-%M-%S')
            except ValueError:
                # If that format fails, try some common alternatives
                try:
                    date_obj = datetime.strptime(date_str, '%d/%m/%Y %H:%M:%S')
                    dir_name = date_obj.strftime('%Y-%m-%d_%H-%M-%S')
                except ValueError:
                    # If parsing fails, use a sanitized version of the original string
                    dir_name = date_str.replace('/', '-').replace(':', '-').replace(' ', '_')
        
        # Get parent directory from command line arguments
        parent_dir = None
        if len(sys.argv) >= 7:
            parent_dir = sys.argv[6]
        
        # Create the full directory path
        if parent_dir:
            full_dir_path = os.path.join(parent_dir, dir_name)
            os.makedirs(parent_dir, exist_ok=True)  # Ensure parent directory exists
        else:
            full_dir_path = dir_name
        
        os.makedirs(full_dir_path, exist_ok=True)
        print(f"Created directory: {full_dir_path}")
        
        # Get values from command line arguments
        if len(sys.argv) >= 5:
            model = sys.argv[1]
        else:
            # Fallback to environment variables if arguments not provided
            model = os.environ.get('MODEL', '')
        
        # Get stage from command line arguments or environment
        if len(sys.argv) >= 6:
            stage = sys.argv[5]
        else:
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

        # CLI args provided from experiment_automation.py
        if len(sys.argv) >= 11:
            in_range_str = sys.argv[7]
            out_range_str = sys.argv[8]
            mi, ma = _parse_range(in_range_str)
            mo, moa = _parse_range(out_range_str)
            min_input_tokens = mi or ''
            max_input_tokens = ma or ''
            min_output_tokens = mo or ''
            max_output_tokens = moa or ''
            req_min = sys.argv[9]
            evaluation_flag = sys.argv[10]
            if len(sys.argv) >= 12:
                median_cli = sys.argv[11]
            if len(sys.argv) >= 13:
                prompt_token_count_cli = sys.argv[12]
        else:
            # Environment variables set in-process by experiment_automation.py
            min_input_tokens = os.environ.get('MIN_INPUT_TOKENS', '')
            max_input_tokens = os.environ.get('MAX_INPUT_TOKENS', '')
            min_output_tokens = os.environ.get('MIN_OUTPUT_TOKENS', '')
            max_output_tokens = os.environ.get('MAX_OUTPUT_TOKENS', '')
            req_min = os.environ.get('REQ_MIN', '')
            evaluation_flag = os.environ.get('EVALUATION', '')
            prompt_token_count_cli = os.environ.get('PROMPT_TOKEN_COUNT', '')

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
        comp_median, total_requests, sr_from_results = _compute_median_response_tokens(Path('results.json'))
        if median_resp_tokens is None:
            median_resp_tokens = comp_median
        if not success_rate and sr_from_results:
            success_rate = sr_from_results

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
                "DURATION", "TOTAL_REQUESTS", "SUCCESS_RATE", "MEDIAN_PROMPT_TOKENS",
                "MEDIAN_RESPONSE_TOKENS", "JOB_ID", "STAGE"
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
                total_requests or '',
                success_rate or '',
                prompt_token_count or '',
                median_resp_tokens or '',
                job_id or '',
                stage
            ])
        
        print(f"Created results.csv in {full_dir_path}")
        
        # Create prompts.csv with unique prompts used
        _write_prompts_csv(full_dir_path)
        
        # Move CSV files to the new directory
        shutil.move("output.csv", os.path.join(full_dir_path, "output.csv"))
        shutil.move("first_half.csv", os.path.join(full_dir_path, "first_half.csv"))
        shutil.move("second_half.csv", os.path.join(full_dir_path, "second_half.csv"))
        # Keep a copy of results.json in the working directory for downstream readers
        shutil.copyfile("results.json", os.path.join(full_dir_path, "results.json"))
        
        print("Moved CSV files and copied results.json to the directory")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()