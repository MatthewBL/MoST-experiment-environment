import csv
import json
import os
import random
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from fmperf.utils.constants import REQUESTS_DIR, REQUESTS_FILENAME, RESULTS_FILENAME

REQUESTS_PROMPTS_FILE = Path("oasst_roots_en_max1000_tokens.jsonl")

#
# Configuration loader: read values from .env without modifying the file.
#
def _parse_tokens_list(value):
    """Parse TOKENS_LIST into input/output intervals.

    Supported formats (comma-separated):
    - "in:out"                -> [[in_min,in_max,out_min,out_max]] with in_min=in_max=in, out_min=out_max=out
    - "inMin-inMax:out"       -> [[inMin,inMax,out,out]]
    - "in:outMin-outMax"      -> [[in,in,outMin,outMax]]
    - "inMin-inMax:outMin-outMax" -> [[inMin,inMax,outMin,outMax]]

    Malformed entries are skipped. Whitespace is ignored.
    """
    tokens = []
    if not value:
        return tokens
    for item in value.split(','):
        s = item.strip()
        if not s:
            continue
        if ':' not in s:
            continue
        in_part, out_part = s.split(':', 1)
        in_part = in_part.strip()
        out_part = out_part.strip()
        try:
            if '-' in in_part:
                in_min_str, in_max_str = in_part.split('-', 1)
                in_min = int(in_min_str.strip())
                in_max = int(in_max_str.strip())
            else:
                in_min = in_max = int(in_part)
            if '-' in out_part:
                out_min_str, out_max_str = out_part.split('-', 1)
                out_min = int(out_min_str.strip())
                out_max = int(out_max_str.strip())
            else:
                out_min = out_max = int(out_part)
            tokens.append([in_min, in_max, out_min, out_max])
        except ValueError:
            continue
    return tokens

def _parse_int_list(value):
    """Parse comma-separated integers into a list. Invalid entries are skipped.

    Examples:
    - "1,2,4" -> [1, 2, 4]
    - "8"     -> [8]
    - "1, x"  -> [1]
    """
    items = []
    if value is None:
        return items
    for part in str(value).split(','):
        p = part.strip()
        if not p:
            continue
        try:
            items.append(int(p))
        except ValueError:
            # skip malformed entries
            continue
    return items

def _parse_float_list(value):
    """Parse comma-separated floats into a list. Invalid entries are skipped."""
    items = []
    if value is None:
        return items
    for part in str(value).split(','):
        p = part.strip()
        if not p:
            continue
        try:
            items.append(float(p))
        except ValueError:
            continue
    return items


def _parse_bool(value, default=False):
    """Parse common boolean strings. Returns default when value is empty/unknown."""
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if s in {'0', 'false', 'no', 'n', 'off'}:
        return False
    return default

def load_env_config():
    """Load configuration from .env file and return a dict.

    Expected keys:
    - TOKENS_LIST: comma-separated pairs like "32:32,32:64"
    - TOKENS_LIST_PROPORTION: comma-separated weights, one per TOKENS_LIST entry
    - ADDITIVE: TRUE/FALSE, when TRUE run one mixed experiment across TOKENS_LIST
    - REQ_MIN_START: comma-separated integers (per-token-combo initial REQ_MIN)
    - REQ_MIN_INCREASE_MULTIPLIER: integer (multiplier for stage 1 success)
        - STOP_THRESHOLD: float (relative stop threshold used as
            M - m <= M * STOP_THRESHOLD in stage 2)
    """
    env_path = Path('.env')
    config = {
        'TOKENS_LIST': [],
        'TOKENS_LIST_PROPORTION': [],
        'ADDITIVE': False,
        'REQ_MIN_START': [1],
        'REQ_MIN_INCREASE_MULTIPLIER': 2.0,
        'STOP_THRESHOLD': 0.5,
        'EXPERIMENT_TYPE': 'MST',
        'DURATION': None,
    }

    if env_path.exists():
        with env_path.open('r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#') or '=' not in s:
                    continue
                key, val = s.split('=', 1)
                key = key.strip()
                val = val.strip()
                if key == 'TOKENS_LIST':
                    config['TOKENS_LIST'] = _parse_tokens_list(val)
                elif key == 'TOKENS_LIST_PROPORTION':
                    config['TOKENS_LIST_PROPORTION'] = _parse_float_list(val)
                elif key == 'ADDITIVE':
                    config['ADDITIVE'] = _parse_bool(val, default=False)
                elif key == 'REQ_MIN_START':
                    lst = _parse_int_list(val)
                    # Backwards compatibility: if parsing produced empty but val is a single int, wrap it
                    if not lst:
                        try:
                            lst = [int(val)]
                        except ValueError:
                            lst = config['REQ_MIN_START']
                    config['REQ_MIN_START'] = lst
                elif key == 'REQ_MIN_INCREASE_MULTIPLIER':
                    try:
                        config['REQ_MIN_INCREASE_MULTIPLIER'] = float(val)
                    except ValueError:
                        pass
                elif key == 'STOP_THRESHOLD':
                    try:
                        config['STOP_THRESHOLD'] = float(val)
                    except ValueError:
                        pass
                elif key == 'EXPERIMENT_TYPE':
                    config['EXPERIMENT_TYPE'] = val.strip() or 'MST'
                elif key == 'DURATION':
                    config['DURATION'] = val.strip()

    return config

CONFIG = load_env_config()

_DURATION_PATTERN = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>[smhdSMHD]?)$")
MIT_PLATEAU_REL_TOL = float(os.environ.get('MIT_PLATEAU_REL_TOL', '0.01'))
MIT_PLATEAU_ABS_TOL = float(os.environ.get('MIT_PLATEAU_ABS_TOL', '0.5'))
SUCCESS_RATE_THRESHOLD = float(os.environ.get('SUCCESS_RATE_THRESHOLD', '95.0'))


def _normalize_experiment_type(value):
    if not value:
        return 'MST'
    return value.strip().upper()


def get_experiment_type():
    """Return the experiment type from env or .env config."""
    env_val = os.environ.get('EXPERIMENT_TYPE')
    if env_val:
        return _normalize_experiment_type(env_val)
    return _normalize_experiment_type(CONFIG.get('EXPERIMENT_TYPE', 'MST'))


def is_additive_experiment():
    """Return whether additive mode is enabled via env or config."""
    env_val = os.environ.get('ADDITIVE')
    if env_val is not None:
        return _parse_bool(env_val, default=False)
    return bool(CONFIG.get('ADDITIVE', False))


def _resolve_tokens_list_proportion(tokens_list):
    """Return one non-negative weight per token interval pair."""
    env_val = os.environ.get('TOKENS_LIST_PROPORTION')
    if env_val is not None:
        weights = _parse_float_list(env_val)
    else:
        weights = list(CONFIG.get('TOKENS_LIST_PROPORTION', []))

    if not tokens_list:
        return []

    if not weights:
        return [1.0 for _ in tokens_list]

    cleaned = []
    for w in weights:
        try:
            value = float(w)
        except (TypeError, ValueError):
            value = 0.0
        cleaned.append(max(0.0, value))

    if len(cleaned) < len(tokens_list):
        print(
            "Warning: TOKENS_LIST_PROPORTION has fewer values than TOKENS_LIST. "
            "Missing weights default to 1.0."
        )
        cleaned.extend([1.0] * (len(tokens_list) - len(cleaned)))
    elif len(cleaned) > len(tokens_list):
        print(
            "Warning: TOKENS_LIST_PROPORTION has more values than TOKENS_LIST. "
            "Extra values are ignored."
        )
        cleaned = cleaned[:len(tokens_list)]

    if all(w == 0.0 for w in cleaned):
        print("Warning: all TOKENS_LIST_PROPORTION values are zero. Falling back to uniform weights.")
        return [1.0 for _ in tokens_list]

    return cleaned


def _parse_duration_seconds(value):
    """Parse duration strings like '1800s', '30m', '2h' into seconds."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    match = _DURATION_PATTERN.match(s)
    if not match:
        try:
            return float(s)
        except ValueError:
            return None
    number = float(match.group('value'))
    unit = match.group('unit').lower()
    if unit == 'm':
        number *= 60
    elif unit == 'h':
        number *= 3600
    elif unit == 'd':
        number *= 86400
    return number


def _get_duration_seconds():
    """Resolve the experiment duration in seconds from env or config."""
    env_val = os.environ.get('DURATION')
    seconds = _parse_duration_seconds(env_val)
    if seconds is not None:
        return seconds
    return _parse_duration_seconds(CONFIG.get('DURATION'))


def _compute_success_rate_from_results():
    """Compute overall success rate (%) directly from results payload."""
    results_path = Path(REQUESTS_DIR) / RESULTS_FILENAME
    if not results_path.exists():
        return None
    try:
        with results_path.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
    except Exception as exc:
        print(f"Warning: unable to read {results_path} for success rate check: {exc}")
        return None

    rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
    if not isinstance(rows, list) or not rows:
        return None

    request_groups = defaultdict(list)
    for item in rows:
        request_idx = item.get('request_idx')
        if request_idx is None:
            continue
        try:
            req_id = int(request_idx)
        except (TypeError, ValueError):
            continue
        worker_idx = item.get('worker_idx')
        key = (worker_idx if worker_idx is not None else -1, req_id)
        request_groups[key].append(item)

    total_requests = len(request_groups)
    if total_requests == 0:
        return None

    successful_requests = 0
    for group in request_groups.values():
        if all(entry.get('ok') and entry.get('error') == "None" for entry in group):
            successful_requests += 1

    return (successful_requests / total_requests) * 100.0


def _check_success_rate_threshold(threshold=None, prefer_results_json=False):
    """Check that success rates stay above threshold via CSVs or raw results."""
    effective_threshold = SUCCESS_RATE_THRESHOLD if threshold is None else threshold

    if prefer_results_json:
        success_rate = _compute_success_rate_from_results()
        if success_rate is not None:
            if success_rate < effective_threshold:
                print(
                    "Success rate below "
                    f"{effective_threshold}% detected in results payload: {success_rate}"
                )
                return False
            return True
        print("Warning: unable to compute success rate from results payload; falling back to CSV inspection.")

    csv_names = ("first_half.csv", "second_half.csv")
    base_dir = Path('requests')
    for name in csv_names:
        path = base_dir / name
        if not path.exists():
            continue
        try:
            with path.open('r', encoding='utf-8', newline='') as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or 'success_rate' not in reader.fieldnames:
                    continue
                for row in reader:
                    raw_val = row.get('success_rate')
                    if raw_val is None or raw_val == '':
                        continue
                    try:
                        val = float(raw_val)
                    except ValueError:
                        continue
                    if val < effective_threshold:
                        print(
                            f"Success rate below {effective_threshold}% detected in {name}: {val}"
                        )
                        return False
        except Exception as exc:
            print(f"Warning: unable to inspect success_rate in {path}: {exc}")
    return True

_TOKEN_SUFFIX_RE = re.compile(r"^(.*)_([0-9]+)-([0-9]+)$")

def _strip_token_suffix(filename):
    """Remove trailing _MIN-MAX suffix from filename if present."""
    name, ext = os.path.splitext(filename)
    match = _TOKEN_SUFFIX_RE.match(name)
    if match:
        return f"{match.group(1)}{ext}"
    return filename

def _get_requests_filename_base():
    base = os.environ.get('REQUESTS_FILENAME_BASE')
    if base:
        return base
    current = os.environ.get('REQUESTS_FILENAME', REQUESTS_FILENAME)
    normalized = _strip_token_suffix(current)
    os.environ['REQUESTS_FILENAME_BASE'] = normalized
    return normalized

def set_process_env_for_run(
    req_min_value,
    input_interval=None,
    output_interval=None,
    apply_output_bounds=True,
    requests_filename=None,
):
    """Set environment variables in-process for a run without modifying .env.

    input_interval/output_interval can be:
    - a single int, or
    - a tuple/list (min, max)
    """
    if req_min_value is not None:
        os.environ['REQ_MIN'] = str(req_min_value)
    if input_interval is not None:
        if isinstance(input_interval, (list, tuple)) and len(input_interval) >= 2:
            os.environ['MIN_INPUT_TOKENS'] = str(input_interval[0])
            os.environ['MAX_INPUT_TOKENS'] = str(input_interval[1])
        else:
            os.environ['MIN_INPUT_TOKENS'] = str(input_interval)
            os.environ['MAX_INPUT_TOKENS'] = str(input_interval)
    if apply_output_bounds and output_interval is not None:
        if isinstance(output_interval, (list, tuple)) and len(output_interval) >= 2:
            os.environ['MIN_OUTPUT_TOKENS'] = str(output_interval[0])
            os.environ['MAX_OUTPUT_TOKENS'] = str(output_interval[1])
        else:
            os.environ['MIN_OUTPUT_TOKENS'] = str(output_interval)
            os.environ['MAX_OUTPUT_TOKENS'] = str(output_interval)
    elif not apply_output_bounds:
        os.environ.pop('MIN_OUTPUT_TOKENS', None)
        os.environ.pop('MAX_OUTPUT_TOKENS', None)

    # Append min-max input interval to REQUESTS_FILENAME so downstream tools read the correct file
    if input_interval is not None:
        if isinstance(input_interval, (list, tuple)) and len(input_interval) >= 2:
            in_min, in_max = int(input_interval[0]), int(input_interval[1])
        else:
            in_min = in_max = int(input_interval)
        base_filename = _get_requests_filename_base()
        name, ext = os.path.splitext(base_filename)
        if not ext:
            ext = '.json'
        os.environ['REQUESTS_FILENAME'] = f"{name}_{in_min}-{in_max}{ext}"
    if requests_filename is not None:
        os.environ['REQUESTS_FILENAME'] = str(requests_filename)


def _weighted_counts(total_count, weights):
    """Distribute total_count across intervals according to weights."""
    if total_count <= 0 or not weights:
        return [0 for _ in weights]
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return [0 for _ in weights]

    raw = [(w / weight_sum) * total_count for w in weights]
    counts = [int(x) for x in raw]
    remainder = total_count - sum(counts)
    if remainder > 0:
        order = sorted(range(len(raw)), key=lambda i: (raw[i] - counts[i]), reverse=True)
        for idx in order[:remainder]:
            counts[idx] += 1
    return counts


def _load_cases(path):
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Workload file does not contain a list of cases: {path}")
    return payload


def _build_additive_workload(tokens_list, weights):
    """Create a single mixed requests file using weighted token-interval proportions."""
    if not tokens_list:
        raise ValueError("TOKENS_LIST is empty; additive experiment requires at least one interval.")

    prompts_path = REQUESTS_PROMPTS_FILE.resolve()
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts dataset missing: {prompts_path}")

    requests_dir = Path(REQUESTS_DIR)
    requests_dir.mkdir(parents=True, exist_ok=True)

    base_filename = _get_requests_filename_base()
    base_name, base_ext = os.path.splitext(base_filename)
    if not base_ext:
        base_ext = '.json'

    per_interval_cases = []
    interval_labels = []
    for tokens in tokens_list:
        if len(tokens) >= 4:
            in_min, in_max, out_min, out_max = tokens[0], tokens[1], tokens[2], tokens[3]
        else:
            in_min = in_max = tokens[0]
            out_min = out_max = tokens[1]

        label = f"{in_min}-{in_max}:{out_min}-{out_max}"
        interval_labels.append(label)
        per_interval_filename = f"{base_name}_{in_min}-{in_max}_{out_min}-{out_max}{base_ext}"
        per_interval_path = requests_dir / per_interval_filename

        if per_interval_path.is_file():
            print(f"Found cached additive workload component: {per_interval_path}")
        else:
            command = (
                f'python -u generate_requests.py {in_min} {in_max} '
                f'--prompts-file "{prompts_path}" --min-output {out_min} --max-output {out_max} '
                f'--output "{per_interval_path}"'
            )
            run_command(command, wait=True)
            if not per_interval_path.is_file():
                raise FileNotFoundError(
                    f"Workload generation failed; expected additive component not found: {per_interval_path}"
                )

        cases = _load_cases(per_interval_path)
        if not cases:
            raise ValueError(f"Generated additive component has no cases: {per_interval_path}")
        per_interval_cases.append(cases)

    total_cases = sum(len(cases) for cases in per_interval_cases)
    selected_counts = _weighted_counts(total_cases, weights)

    rng = random.Random(42)
    mixed_cases = []
    for idx, cases in enumerate(per_interval_cases):
        target_count = selected_counts[idx]
        if target_count <= 0:
            print(f"Additive mix -> interval {interval_labels[idx]} (weight {weights[idx]}): selected 0 cases")
            continue

        if target_count <= len(cases):
            selected = rng.sample(cases, target_count)
        else:
            selected = list(cases)
            missing = target_count - len(cases)
            for _ in range(missing):
                selected.append(rng.choice(cases))

        mixed_cases.extend(selected)
        print(
            f"Additive mix -> interval {interval_labels[idx]} (weight {weights[idx]}): "
            f"selected {len(selected)} cases"
        )

    if not mixed_cases:
        raise ValueError("Additive workload is empty after applying TOKENS_LIST_PROPORTION.")

    rng.shuffle(mixed_cases)
    mixed_filename = f"{base_name}_additive{base_ext}"
    mixed_path = requests_dir / mixed_filename
    with mixed_path.open('w', encoding='utf-8') as handle:
        json.dump(mixed_cases, handle)

    print(f"Created additive mixed workload at {mixed_path} with {len(mixed_cases)} cases")
    return mixed_path

def run_command(command, wait=True):
    """Run a shell command and wait for completion"""
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True)
    if wait:
        process.wait()
        if process.returncode != 0:
            print(f"Warning: Command '{command}' returned non-zero exit code: {process.returncode}")
    return process

def run_command_capture(command):
    """Run a shell command, wait, and capture stdout/stderr."""
    print(f"Running (capture): {command}")
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"Warning: Command '{command}' returned non-zero exit code: {result.returncode}")
    return result.returncode, result.stdout, result.stderr

def run_evaluation_pipeline(model, gpus, cpus, node, stage, parent_dir, experiment_type):
    """Run the evaluation pipeline steps 3-7 and return throughput metric."""
    # Step 3: Run loadgen
    run_command("python -u -m fmperf.loadgen.run")
    
    # Step 4: Change to requests directory
    original_dir = os.getcwd()
    os.chdir('requests')
    
    try:
        # Step 5: Convert to CSV
        run_command("python -u convert_to_csv.py")
        
        # Early metrics check before splitting results (invoke analyze_metrics.py as a script)
        early_fail = False
        median_resp_per_min = None
        try:
            code, out, err = run_command_capture("python -u analyze_metrics.py .")
            if code == 0:
                avg_resp = 0.0
                for line in out.splitlines():
                    if "Median responded requests per minute:" in line:
                        try:
                            avg_resp = float(line.split(":", 1)[1].strip())
                        except Exception:
                            pass
                        break
                median_resp_per_min = avg_resp
                req_min_env = os.environ.get('REQ_MIN', '0')
                try:
                    req_min_val = float(req_min_env)
                except ValueError:
                    req_min_val = 0.0
                if experiment_type != 'MIT' and avg_resp < (0.95 * req_min_val):
                    print(f"Early evaluation failure: median responded/min = {avg_resp:.3f} < 95% of REQ_MIN = {req_min_val}")
                    early_fail = True
                else:
                    print(
                        f"Early metrics check{' (MIT informational only)' if experiment_type == 'MIT' else ' passed'}: "
                        f"median responded/min = {avg_resp:.3f}, REQ_MIN = {req_min_val}"
                    )
            else:
                # Non-zero exit from analyze_metrics: log and continue normal flow
                print(f"Warning: analyze_metrics.py exited with code {code}. stderr: {err.strip()}")
        except Exception as e:
            # Don't block on metrics errors; proceed with normal flow
            print(f"Warning: analyze_metrics script invocation failed: {e}")
            early_fail = False
        
        # Step 6: Split results
        run_command("python -u split_results.py")
        
        # Step 7: Run evaluation (skip if early failure triggered)
        if early_fail:
            result = None
            evaluation_success = False
        elif experiment_type == 'MIT':
            # MIT experiments rely on throughput plateau detection later
            evaluation_success = True
        else:
            result = run_command("python -u evaluate.py", wait=True)
            # Evaluation.py returns 0 for success, non-zero for failure
            evaluation_success = (result.returncode == 0)
        
        return evaluation_success, median_resp_per_min
        
    finally:
        # Always return to original directory
        os.chdir(original_dir)

def start_stage_1():
    """Initialize stage 1"""
    return 1

def end_experiment(stage, M, m, M_0, m_0, evaluation):
    """Check termination condition for stage 2 using a relative threshold based on M.

    Stops when the gap (M - m) is less than or equal to M * STOP_THRESHOLD.
    Falls back to absolute STOP_THRESHOLD if M or m are not numbers.
    """
    stop_threshold = CONFIG.get('STOP_THRESHOLD', 0.5)
    if stage == 2 and (M is not None) and (m is not None):
        try:
            relative_threshold = float(M) * float(stop_threshold)
        except Exception:
            # Fallback: treat threshold as absolute if casting fails
            relative_threshold = float(stop_threshold)

        if (M - m) <= relative_threshold:
            if evaluation:
                return True, "REQ_MIN", None  # Return REQ_MIN as result
            else:
                return True, "m", m  # Return m as result
    return False, None, None

def update_stage_1(evaluation, current_req_min, retry_count_stage1, highest_true, lowest_false):
    """Update logic for stage 1 with a retry mechanism.

    Behavior:
    - Success (TRUE): track highest TRUE; if no confirmed FALSE yet, increase REQ_MIN by multiplier; if a confirmed FALSE exists, transition to stage 2.
    - Failure (FALSE): require a double-false at the SAME REQ_MIN to confirm the FALSE bound. A single FALSE followed by TRUE must NOT start stage 2.
    - If no TRUE yet and a value is confirmed FALSE (double-false), keep decreasing to find a TRUE bound.

    Stage 2 only starts when there is at least one TRUE (lower bound) and a CONFIRMED FALSE (upper bound via double-false).

    Returns: (stage, new_req_min, M_0, m_0, M, m, retry_count_stage1, highest_true, lowest_false)
    """
    multiplier = CONFIG.get('REQ_MIN_INCREASE_MULTIPLIER', 2.0)

    if evaluation:
        # Successful evaluation: record lower bound and clear any pending failure retry
        if highest_true is None or current_req_min > highest_true:
            highest_true = current_req_min

        # Clear failure retry on success
        retry_count_stage1 = 0

        if lowest_false is None:
            # No confirmed FALSE yet → keep increasing
            # Use float multiplier, keep REQ_MIN as integer via rounding
            new_req_min = max(1, int(round(current_req_min * multiplier)))
            return 1, new_req_min, None, None, None, None, retry_count_stage1, highest_true, lowest_false
        else:
            # We have a confirmed FALSE and at least one TRUE → transition to stage 2
            M_0 = lowest_false
            m_0 = highest_true
            stage = 2
            M = M_0
            m = m_0
            new_req_min = (M + m) / 2
            return stage, new_req_min, M_0, m_0, M, m, retry_count_stage1, highest_true, lowest_false
    else:
        # Failed evaluation: require double-false to confirm upper bound
        if retry_count_stage1 == 0:
            # First failure at this REQ_MIN → retry same value to confirm
            retry_count_stage1 = 1
            return 1, current_req_min, None, None, None, None, retry_count_stage1, highest_true, lowest_false
        else:
            # Second consecutive failure at same REQ_MIN → confirmed FALSE
            retry_count_stage1 = 0
            if lowest_false is None or current_req_min < lowest_false:
                lowest_false = current_req_min

            if highest_true is None:
                # No TRUE yet: keep decreasing to find a TRUE bound
                # Use float multiplier, keep REQ_MIN as integer via rounding
                new_req_min = max(1, int(round(current_req_min / multiplier)))
                return 1, new_req_min, None, None, None, None, retry_count_stage1, highest_true, lowest_false
            else:
                # Have TRUE and confirmed FALSE → transition to stage 2
                M_0 = lowest_false
                m_0 = highest_true
                stage = 2
                M = M_0
                m = m_0
                new_req_min = (M + m) / 2
                return stage, new_req_min, M_0, m_0, M, m, retry_count_stage1, highest_true, lowest_false

def update_stage_2(evaluation, current_req_min, M, m, retry_count_stage2):
    """Update logic for stage 2 with retry mechanism (uses retry_count_stage2)."""
    if evaluation:
        m = current_req_min  # Successful evaluation: move lower bound up
        retry_count_stage2 = 0  # Reset retry count on success
    else:
        if retry_count_stage2 == 0:
            # First failure: don't update M, just increment retry count
            retry_count_stage2 = 1
        else:
            # Second consecutive failure: update M and reset retry count
            M = current_req_min
            retry_count_stage2 = 0

    new_req_min = (M + m) / 2
    return new_req_min, M, m, retry_count_stage2

def run_experiment_for_tokens(tokens, initial_req_min=None, additive=False, additive_tokens=None, additive_weights=None):
    """Run the complete experiment for a specific token combination.

    initial_req_min: optional initial value for REQ_MIN specific to this
    token combination. If None, defaults to 1.
    """
    # Get environment variables at the start and store them as Python variables
    model = os.environ.get('MODEL', '')
    gpus = os.environ.get('GPUS', '')
    cpus = os.environ.get('CPUS', '')
    node = os.environ.get('NODE', '')
    experiment_type = get_experiment_type()
    is_mit = (experiment_type == 'MIT')
    print(f"Experiment type: {experiment_type}")
    duration_seconds = _get_duration_seconds()
    if duration_seconds is None and is_mit:
        print("Warning: Unable to determine experiment duration; MIT throughput checks may be unavailable.")
    
    print(f"Stored configuration - MODEL: {model}, GPUS: {gpus}, CPUS: {cpus}, NODE: {node}")
    
    # Configure experiment mode and token intervals
    if additive:
        parent_dir = "additive_experiment"
        interval_strs = ("MIXED", "MIXED")
        input_interval = None
        output_interval = None
        additive_components = additive_tokens if additive_tokens else [tokens]
        print(f"Running additive experiment with {len(additive_components)} token interval pairs")
    else:
        # tokens can be [in_min,in_max,out_min,out_max] or [in,out]
        if len(tokens) >= 4:
            in_min, in_max, out_min, out_max = tokens[0], tokens[1], tokens[2], tokens[3]
            parent_dir = f"{in_min}-{in_max}_{out_min}-{out_max}"
            input_interval = (in_min, in_max)
            output_interval = (out_min, out_max)
            interval_strs = (f"{in_min}-{in_max}", f"{out_min}-{out_max}")
        else:
            in_min = in_max = tokens[0]
            out_min = out_max = tokens[1]
            parent_dir = f"{tokens[0]}_{tokens[1]}"
            input_interval = tokens[0]
            output_interval = tokens[1]
            interval_strs = (str(tokens[0]), str(tokens[1]))
    os.makedirs(parent_dir, exist_ok=True)
    print(f"Created parent directory: {parent_dir}")
    
    # Step 1: Initialize stage 1
    stage = start_stage_1()
    req_min = initial_req_min if initial_req_min is not None else 1  # Per-combo initial value
    
    # Stage 2 variables (initialized when transitioning to stage 2)
    M_0, m_0, M, m = None, None, None, None
    
    # Bounds tracking for stage 1
    highest_true = None  # Highest req_min that yielded TRUE
    lowest_false = None  # Lowest req_min that yielded FALSE
    
    # Retry counters for stage 1 and stage 2
    retry_count_stage1 = 0
    retry_count_stage2 = 0
    mit_rpm_history = []
    
    max_iterations = 100  # Safety limit to prevent infinite loops
    iteration = 0

    # Set process env for request generation without modifying .env
    req_path = None
    if additive:
        weights = additive_weights if additive_weights is not None else []
        req_path = _build_additive_workload(additive_components, weights)
        set_process_env_for_run(
            req_min,
            input_interval=None,
            output_interval=None,
            apply_output_bounds=False,
            requests_filename=req_path.name,
        )
    else:
        set_process_env_for_run(req_min, input_interval=input_interval, output_interval=output_interval)
        requests_dir = Path('requests')
        os.chdir(requests_dir)
        sample_file = Path('sample_requests.json')
        if sample_file.exists():
            sample_file.unlink()
        os.chdir('..')
        # Skip generation if interval-specific file already exists (uses REQUESTS_FILENAME with input suffix)
        req_filename = os.environ.get('REQUESTS_FILENAME', REQUESTS_FILENAME)
        req_path = Path(REQUESTS_DIR) / req_filename
        if req_path.is_file():
            print(f"Found existing workload: {req_path}. Using cached file.")
        else:
            prompts_path = REQUESTS_PROMPTS_FILE.resolve()
            if not prompts_path.exists():
                raise FileNotFoundError(f"Prompts dataset missing: {prompts_path}")
            command = (
                f'python -u generate_requests.py {in_min} {in_max} '
                f'--prompts-file "{prompts_path}"'
            )
            if out_min is not None and out_max is not None:
                command += f" --min-output {out_min} --max-output {out_max}"
            run_command(command, wait=True)
            if req_path.is_file():
                print(f"Generated workload: {req_path}")
            else:
                raise FileNotFoundError(
                    f"Workload generation failed; expected file not found: {req_path}"
                )
    
    def _get_prompt_info():
        """Read prompt text and token count from generated requests file."""
        try:
            req_filename = os.environ.get("REQUESTS_FILENAME", REQUESTS_FILENAME)
            path = os.path.join(REQUESTS_DIR, req_filename)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                case = data[0]
                prompt_text = case.get("prompt_text")
                prompt_token_count = case.get("prompt_token_count")
                # Fallbacks
                if prompt_text is None:
                    req = case.get("request", {})
                    if isinstance(req, dict) and "request" in req and isinstance(req["request"], dict):
                        prompt_text = req["request"].get("text")
                if prompt_token_count is None:
                    req = case.get("request", {})
                    if isinstance(req, dict) and "prompt" in req and isinstance(req["prompt"], list):
                        prompt_token_count = len(req["prompt"])
                    elif isinstance(case.get("config"), dict) and "in_tokens" in case["config"]:
                        prompt_token_count = case["config"]["in_tokens"]
                return prompt_text, prompt_token_count
        except Exception as e:
            print(f"Warning: unable to read prompt info: {e}")
        return None, None

    def _compute_median_response_tokens():
        """Compute (median tokens per response, total completed requests)."""
        try:
            results_path = os.path.join(REQUESTS_DIR, RESULTS_FILENAME)
            with open(results_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
            if not isinstance(rows, list):
                return None, None
            totals = {}
            worker_idxs = set()
            for r in rows:
                rid = r.get("request_idx")
                n = r.get("n_tokens", 0)
                wid = r.get("worker_idx")
                if rid is None:
                    continue
                try:
                    n_val = float(n)
                except Exception:
                    n_val = 0.0
                totals[rid] = totals.get(rid, 0.0) + n_val
                if wid is not None:
                    worker_idxs.add(wid)
            worker_count = len(worker_idxs) if len(worker_idxs) > 0 else 1
            values = [v / worker_count for v in totals.values()]
            total_completed_requests = len(totals)
            if not values:
                return 0.0, total_completed_requests
            values.sort()
            mid = len(values) // 2
            if len(values) % 2 == 1:
                return float(values[mid]), total_completed_requests
            return (values[mid - 1] + values[mid]) / 2.0, total_completed_requests
        except Exception as e:
            print(f"Warning: unable to compute median response tokens: {e}")
            return None, None

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration}, Stage {stage}, INPUT_TOKENS={interval_strs[0]}, OUTPUT_TOKENS={interval_strs[1]}, REQ_MIN={req_min} ---")
        
        # Step 2: Update in-process environment for this iteration (no .env writes)
        set_process_env_for_run(
            req_min,
            apply_output_bounds=(not additive),
            requests_filename=(req_path.name if req_path is not None else None),
        )
        
        # Steps 3-7: Run evaluation pipeline with stored variables, passing current stage and parent_dir
        evaluation_result, responded_per_min = run_evaluation_pipeline(
            model, gpus, cpus, node, stage, parent_dir, experiment_type
        )
        # Capture the REQ_MIN used for this evaluation before any update logic
        req_min_used = req_min
        median_resp_tokens, total_completed_requests = _compute_median_response_tokens()
        requests_per_sec = None
        if duration_seconds and total_completed_requests is not None:
            try:
                requests_per_sec = total_completed_requests / float(duration_seconds)
            except Exception:
                requests_per_sec = None

        if is_mit:
            success_rate_ok = _check_success_rate_threshold(prefer_results_json=True)
            if not success_rate_ok:
                print(
                    f"MIT iteration failed due to success rate falling below {SUCCESS_RATE_THRESHOLD}%."
                )
                evaluation_result = False
            elif evaluation_result and stage == 1:
                responded_per_min = responded_per_min or (
                    requests_per_sec * 60.0 if requests_per_sec is not None else None
                )
                if responded_per_min is None:
                    print(
                        "Warning: Unable to compute responded requests per minute for MIT iteration; "
                        "skipping plateau detection for now."
                    )
                else:
                    mit_rpm_history.append(responded_per_min)
                    if len(mit_rpm_history) >= 3:
                        third_last = mit_rpm_history[-3]
                        second_last = mit_rpm_history[-2]
                        last = mit_rpm_history[-1]
                        prev_delta = second_last - third_last
                        curr_delta = last - second_last
                        threshold = max(abs(prev_delta) * MIT_PLATEAU_REL_TOL, MIT_PLATEAU_ABS_TOL)
                        if curr_delta < 0:
                            evaluation_result = False
                            print(
                                "Detected throughput regression: "
                                f"prev={second_last:.4f} rpm -> current={last:.4f} rpm (delta {curr_delta:.4f})."
                            )
                        elif abs(curr_delta) <= threshold:
                            evaluation_result = False
                            print(
                                "Detected MIT plateau using last three iterations: "
                                f"prev Δ={prev_delta:.4f}, current Δ={curr_delta:.4f}, "
                                f"threshold={threshold:.4f}."
                            )

        print(f"Evaluation result: {'Success' if evaluation_result else 'Failure'}")
        if stage == 2 and not evaluation_result:
            print(f"Retry count: {retry_count_stage2}")

        # When this is set, persist artifacts for the current iteration first,
        # then return the recorded value after store_results.py runs.
        stop_after_persist = False
        persist_return_value = None

        # Step 8: Check termination condition (for stage 2)
        if stage == 2:
            should_end, result_type, result_value = end_experiment(stage, M, m, M_0, m_0, evaluation_result)
            if should_end:
                if result_type == "REQ_MIN":
                    print(f"\nExperiment completed successfully! Optimal REQ_MIN = {req_min}")
                    stop_after_persist = True
                    persist_return_value = req_min
                else:
                    print(f"\nExperiment completed! Result m = {result_value}")
                    stop_after_persist = True
                    persist_return_value = result_value
        
        # Steps 9-10: Update stage variables
        if not stop_after_persist:
            if stage == 1:
                (stage, req_min, M_0, m_0, M, m, retry_count_stage1, highest_true, lowest_false) = update_stage_1(
                    evaluation_result, req_min, retry_count_stage1, highest_true, lowest_false
                )
                # If we transitioned to stage 2, reset stage-2 retry counter and log bounds
                if stage == 2:
                    retry_count_stage2 = 0
                    # Reset MIT history to avoid stage-1 trend checks leaking into stage-2 binary search.
                    if is_mit:
                        mit_rpm_history = []
                    print(f"Transitioned to Stage 2: highest TRUE = {highest_true}, lowest FALSE = {lowest_false}")
            elif stage == 2:
                req_min, M, m, retry_count_stage2 = update_stage_2(
                    evaluation_result, req_min, M, m, retry_count_stage2
                )

        # End-of-iteration logging: prompt and median response tokens
        prompt_text, prompt_token_count = _get_prompt_info()
        print("--- Iteration summary ---")
        if prompt_text is not None:
            display_text = prompt_text if len(str(prompt_text)) <= 400 else str(prompt_text)[:400] + "..."
            print(f"Prompt: {display_text}")
        if prompt_token_count is not None:
            print(f"Prompt token count: {prompt_token_count}")
        if median_resp_tokens is not None:
            print(f"Median tokens per response: {median_resp_tokens:.3f}")
        if total_completed_requests is not None:
            print(f"Completed requests: {total_completed_requests}")
        if responded_per_min is not None:
            print(f"Responded requests per minute (median): {responded_per_min:.3f}")
        if requests_per_sec is not None:
            print(f"Requests per second: {requests_per_sec:.4f}")

        # Persist results with explicit values, including the printed median
        try:
            original_dir2 = os.getcwd()
            os.chdir('requests')
            evaluation_flag = "TRUE" if evaluation_result else "FALSE"
            median_str = f"{median_resp_tokens:.3f}" if isinstance(median_resp_tokens, (int, float)) else (str(median_resp_tokens) if median_resp_tokens is not None else '')
            # Pass prompt details directly to store_results.py to avoid re-parsing
            store_args = [
                "python", "-u", "store_results.py",
                str(model), str(gpus), str(cpus), str(node), str(stage), str(parent_dir),
                str(interval_strs[0]), str(interval_strs[1]), str(req_min_used), str(evaluation_flag), str(median_str),
                str(prompt_token_count if prompt_token_count is not None else ''),
                str(prompt_text if prompt_text is not None else '')
            ]
            print("Running (args):", " ".join(store_args))
            subprocess.run(store_args)
        finally:
            os.chdir(original_dir2)
        
    # Small delay to avoid overwhelming the system
        time.sleep(1)

        if stop_after_persist:
            return persist_return_value
    
    print(f"\nReached maximum iterations ({max_iterations}). Stopping.")
    return None

def main():
    input_output_tokens = CONFIG.get('TOKENS_LIST', [])
    req_min_starts = CONFIG.get('REQ_MIN_START', [1])
    additive = is_additive_experiment()
    results = {}

    if additive:
        if not input_output_tokens:
            print("No token interval pairs found in TOKENS_LIST. Nothing to run.")
            return results

        additive_weights = _resolve_tokens_list_proportion(input_output_tokens)
        initial_req_min = req_min_starts[0] if req_min_starts else 1

        print(f"\n{'='*60}")
        print("Starting additive experiment with mixed token intervals")
        print(f"{'='*60}")

        result = run_experiment_for_tokens(
            input_output_tokens[0],
            initial_req_min,
            additive=True,
            additive_tokens=input_output_tokens,
            additive_weights=additive_weights,
        )
        results["ADDITIVE"] = result

        print("\nCompleted additive experiment")
        print(f"Result: {result}")
        print(f"\n{'='*60}")
        print("ALL EXPERIMENTS COMPLETED")
        print(f"{'='*60}")
        print(f"Tokens ADDITIVE: {result}")
        return results
    
    for idx, tokens in enumerate(input_output_tokens):
        print(f"\n{'='*60}")
        if len(tokens) >= 4:
            print(
                "Starting experiment for "
                f"INPUT_TOKENS={tokens[0]}-{tokens[1]}, OUTPUT_TOKENS={tokens[2]}-{tokens[3]}"
            )
        else:
            print(f"Starting experiment for INPUT_TOKENS={tokens[0]}, OUTPUT_TOKENS={tokens[1]}")
        print(f"{'='*60}")
        
        # Pick initial REQ_MIN by index; if not enough values, use the last one
        if req_min_starts:
            initial_req_min = req_min_starts[idx] if idx < len(req_min_starts) else req_min_starts[-1]
        else:
            initial_req_min = 1
        
        result = run_experiment_for_tokens(tokens, initial_req_min)
        if len(tokens) >= 4:
            result_key = f"{tokens[0]}-{tokens[1]}_{tokens[2]}-{tokens[3]}"
            print(
                "\nCompleted experiment for "
                f"INPUT_TOKENS={tokens[0]}-{tokens[1]}, OUTPUT_TOKENS={tokens[2]}-{tokens[3]}"
            )
        else:
            result_key = f"{tokens[0]}_{tokens[1]}"
            print(f"\nCompleted experiment for INPUT_TOKENS={tokens[0]}, OUTPUT_TOKENS={tokens[1]}")
        results[result_key] = result
        
        print(f"Result: {result}")
    
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
    for token_combo, result in results.items():
        print(f"Tokens {token_combo}: {result}")
    
    return results

if __name__ == "__main__":
    results = main()
    print(f"Final results: {results}")
