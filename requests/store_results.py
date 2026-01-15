import os
import re
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

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
    - Prefer $SLURM_JOB_ID
    - Look for slurm-<jobid>.out upward from CWD
    - Fallback to most recent slurm-*.out in CWD or parents
    """
    # Normalize to string
    job = job_id or os.environ.get("SLURM_JOB_ID")
    candidates: list[Path] = []

    # Search upwards a few levels
    try:
        start = Path.cwd()
        for up in [start, *start.parents[:6]]:
            if job:
                p = up / f"slurm-{job}.out"
                if p.exists():
                    return job, str(p)
            # collect any slurm-*.out as fallback
            for entry in up.glob("slurm-*.out"):
                candidates.append(entry)
    except Exception:
        pass

    if candidates:
        # Pick the most recently modified
        try:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            picked = candidates[0]
            # extract job id from filename if possible
            m = re.search(r"slurm-(\d+)\.out$", picked.name)
            jid = m.group(1) if m else (job or None)
            return jid, str(picked)
        except Exception:
            pass

    return job, None

def _extract_model_from_slurm(slurm_path: str | None) -> str:
    """Heuristically extract model name from the beginning of a Slurm log.
    Looks for common patterns like 'MODEL: <name>' or 'Model used: <name>'.
    """
    if not slurm_path:
        return ""
    patterns = [
        r"\bMODEL\s*[:=]\s*([^\s,]+)",
        r"\bModel used\s*[:=]\s*(.+)",
        r"\bUsing model\s*[:=]?\s*(.+)",
        r"\b--model\s+([^\s]+)",
        r"\bmodel\s*[:=]\s*([^\s,]+)",
    ]
    try:
        with open(slurm_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                s = line.strip()
                for pat in patterns:
                    m = re.search(pat, s, re.IGNORECASE)
                    if m:
                        val = m.group(1).strip()
                        return val.replace("\x00", "").strip()
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
            gpus = sys.argv[2]
            cpus = sys.argv[3]
            node = sys.argv[4]
        else:
            # Fallback to environment variables if arguments not provided
            model = os.environ.get('MODEL', '')
            gpus = os.environ.get('GPUS', '')
            cpus = os.environ.get('CPUS', '')
            node = os.environ.get('NODE', '')
        
        # Get stage from command line arguments or environment
        if len(sys.argv) >= 6:
            stage = sys.argv[5]
        else:
            stage = os.environ.get('STAGE', '')
        
        # Resolve tokens, REQ_MIN, EVALUATION, MEDIAN, and PROMPT INFO: prefer CLI args, then env/log, then .env
        min_input_tokens = ''
        min_output_tokens = ''
        req_min = ''
        evaluation_flag = ''
        median_cli = ''
        prompt_token_count_cli = ''
        prompt_text_cli = ''

        # CLI args provided from experiment_automation.py
        # Expect: 1:model 2:gpus 3:cpus 4:node 5:stage 6:parent_dir 7:min_in 8:min_out 9:req_min 10:evaluation 11:median 12:prompt_token_count 13:prompt_text
        if len(sys.argv) >= 11:
            min_input_tokens = sys.argv[7]
            min_output_tokens = sys.argv[8]
            req_min = sys.argv[9]
            evaluation_flag = sys.argv[10]
            if len(sys.argv) >= 12:
                median_cli = sys.argv[11]
            if len(sys.argv) >= 13:
                prompt_token_count_cli = sys.argv[12]
            if len(sys.argv) >= 14:
                prompt_text_cli = sys.argv[13]
        else:
            # Environment variables set in-process by experiment_automation.py
            min_input_tokens = os.environ.get('MIN_INPUT_TOKENS', '')
            min_output_tokens = os.environ.get('MAX_OUTPUT_TOKENS', '') or os.environ.get('MIN_OUTPUT_TOKENS', '')
            req_min = os.environ.get('REQ_MIN', '')
            evaluation_flag = os.environ.get('EVALUATION', '')
            prompt_token_count_cli = os.environ.get('PROMPT_TOKEN_COUNT', '')
            prompt_text_cli = os.environ.get('PROMPT_TEXT', '')

        # Fallback to .env only if still missing
        if (min_input_tokens == '' or min_output_tokens == '' or req_min == '') and os.path.exists('../.env'):
            with open('../.env', 'r') as env_file:
                for line in env_file:
                    line = line.strip()
                    if min_input_tokens == '' and line.startswith('MIN_INPUT_TOKENS='):
                        min_input_tokens = line.split('=', 1)[1]
                    elif min_output_tokens == '' and line.startswith('MIN_OUTPUT_TOKENS='):
                        min_output_tokens = line.split('=', 1)[1]
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

        # Prompt info: prefer CLI/env-provided values, then fallback to sample_requests.json
        prompt_text = (prompt_text_cli or '').strip()
        prompt_token_count = (prompt_token_count_cli or '').strip()
        if not prompt_text or not prompt_token_count:
            pt, ptc = _read_prompt_info(Path('sample_requests.json'))
            if not prompt_text:
                prompt_text = (pt or '')
            if not prompt_token_count:
                prompt_token_count = (ptc or '')

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
                "MODEL_USED", "INPUT_TOKENS", "OUTPUT_TOKENS", "REQ_MIN", "EVALUATION",
                "DURATION", "TOTAL_REQUESTS", "SUCCESS_RATE", "PROMPT_TOKEN_COUNT",
                "PROMPT_TEXT", "MEDIAN_RESPONSE_TOKENS", "JOB_ID", "NODE", "STAGE"
            ])
            # Write data row
            writer.writerow([
                model_from_slurm or model,
                min_input_tokens,
                min_output_tokens,
                req_min,
                evaluation,
                duration,
                total_requests or '',
                success_rate or '',
                prompt_token_count or '',
                (prompt_text or '').replace('\n', ' ').strip(),
                median_resp_tokens or '',
                job_id or '',
                node, stage
            ])
        
        print(f"Created results.csv in {full_dir_path}")
        
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