import os
import subprocess
import time
from pathlib import Path
import os
import json
import subprocess
import time
from fmperf.utils.constants import REQUESTS_DIR, REQUESTS_FILENAME, RESULTS_FILENAME

#
# Configuration loader: read values from .env without modifying the file.
#
def _parse_tokens_list(value):
    """Parse TOKENS_LIST env value: "a:b,a:b,..." -> [[a,b], ...]."""
    tokens = []
    if not value:
        return tokens
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            a_str, b_str = item.split(':', 1)
            tokens.append([int(a_str), int(b_str)])
        except ValueError:
            # Skip malformed entries
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

def load_env_config():
    """Load configuration from .env file and return a dict.

    Expected keys:
    - TOKENS_LIST: comma-separated pairs like "32:32,32:64"
    - REQ_MIN_START: comma-separated integers (per-token-combo initial REQ_MIN)
    - REQ_MIN_INCREASE_MULTIPLIER: integer (multiplier for stage 1 success)
    - STOP_THRESHOLD: float (used in end condition)
    """
    env_path = Path('.env')
    config = {
        'TOKENS_LIST': [],
        'REQ_MIN_START': [1],
        'REQ_MIN_INCREASE_MULTIPLIER': 2,
        'STOP_THRESHOLD': 0.5,
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
                        config['REQ_MIN_INCREASE_MULTIPLIER'] = int(val)
                    except ValueError:
                        pass
                elif key == 'STOP_THRESHOLD':
                    try:
                        config['STOP_THRESHOLD'] = float(val)
                    except ValueError:
                        pass

    return config

CONFIG = load_env_config()

def set_process_env_for_run(req_min_value, input_tokens=None, output_tokens=None):
    """Set environment variables in-process for a run without modifying .env."""
    if req_min_value is not None:
        os.environ['REQ_MIN'] = str(req_min_value)
    if input_tokens is not None:
        os.environ['MIN_INPUT_TOKENS'] = str(input_tokens)
        os.environ['MAX_INPUT_TOKENS'] = str(input_tokens)
    if output_tokens is not None:
        os.environ['MIN_OUTPUT_TOKENS'] = str(output_tokens)
        os.environ['MAX_OUTPUT_TOKENS'] = str(output_tokens)

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

def run_evaluation_pipeline(model, gpus, cpus, node, stage, parent_dir):
    """Run the evaluation pipeline steps 3-7"""
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
                req_min_env = os.environ.get('REQ_MIN', '0')
                try:
                    req_min_val = float(req_min_env)
                except ValueError:
                    req_min_val = 0.0
                if avg_resp < (0.95 * req_min_val):
                    print(f"Early evaluation failure: median responded/min = {avg_resp:.3f} < 95% of REQ_MIN = {req_min_val}")
                    early_fail = True
                else:
                    print(f"Early metrics check passed: median responded/min = {avg_resp:.3f}, REQ_MIN = {req_min_val}")
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
        else:
            result = run_command("python -u evaluate.py", wait=True)
            # Evaluation.py returns 0 for success, non-zero for failure
            evaluation_success = (result.returncode == 0)
        
        # Step 8: Store results with explicit tokens and REQ_MIN to avoid relying on .env
        min_in = os.environ.get('MIN_INPUT_TOKENS', '')
        min_out = os.environ.get('MIN_OUTPUT_TOKENS', '')
        req_min = os.environ.get('REQ_MIN', '')
        evaluation_flag = "TRUE" if evaluation_success else "FALSE"
        store_command = (
            f'python -u store_results.py '
            f'"{model}" "{gpus}" "{cpus}" "{node}" "{stage}" "{parent_dir}" '
            f'"{min_in}" "{min_out}" "{req_min}" "{evaluation_flag}"'
        )
        run_command(store_command)
        
        return evaluation_success
        
    finally:
        # Always return to original directory
        os.chdir(original_dir)

def start_stage_1():
    """Initialize stage 1"""
    return 1

def end_experiment(stage, M, m, M_0, m_0, evaluation):
    """Check termination condition for stage 2"""
    stop_threshold = CONFIG.get('STOP_THRESHOLD', 0.5)
    if stage == 2 and (M - m) <= stop_threshold:
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
    multiplier = CONFIG.get('REQ_MIN_INCREASE_MULTIPLIER', 2)

    if evaluation:
        # Successful evaluation: record lower bound and clear any pending failure retry
        if highest_true is None or current_req_min > highest_true:
            highest_true = current_req_min

        # Clear failure retry on success
        retry_count_stage1 = 0

        if lowest_false is None:
            # No confirmed FALSE yet → keep increasing
            new_req_min = current_req_min * multiplier
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
                new_req_min = current_req_min / multiplier
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

def run_experiment_for_tokens(tokens, initial_req_min=None):
    """Run the complete experiment for a specific token combination.

    initial_req_min: optional initial value for REQ_MIN specific to this
    token combination. If None, defaults to 1.
    """
    # Get environment variables at the start and store them as Python variables
    model = os.environ.get('MODEL', '')
    gpus = os.environ.get('GPUS', '')
    cpus = os.environ.get('CPUS', '')
    node = os.environ.get('NODE', '')
    
    print(f"Stored configuration - MODEL: {model}, GPUS: {gpus}, CPUS: {cpus}, NODE: {node}")
    
    # Create parent directory for this token pair
    parent_dir = f"{tokens[0]}_{tokens[1]}"
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
    
    max_iterations = 100  # Safety limit to prevent infinite loops
    iteration = 0

    # Set process env for the initial request generation without modifying .env
    set_process_env_for_run(req_min, input_tokens=tokens[0], output_tokens=tokens[1])
    requests_dir = Path('requests')
    os.chdir(requests_dir)
    sample_file = Path('sample_requests.json')
    if sample_file.exists():
        sample_file.unlink()
    os.chdir('..')
    run_command("python -u -m fmperf.loadgen.generate-input", wait=True)
    
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
        """Compute median tokens per full response from results file."""
        try:
            results_path = os.path.join(REQUESTS_DIR, RESULTS_FILENAME)
            with open(results_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            rows = payload["results"] if isinstance(payload, dict) and "results" in payload else payload
            if not isinstance(rows, list):
                return None
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
            if not values:
                return 0.0
            values.sort()
            mid = len(values) // 2
            if len(values) % 2 == 1:
                return float(values[mid])
            return (values[mid - 1] + values[mid]) / 2.0
        except Exception as e:
            print(f"Warning: unable to compute median response tokens: {e}")
            return None

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration}, Stage {stage}, INPUT_TOKENS={tokens[0]}, OUTPUT_TOKENS={tokens[1]}, REQ_MIN={req_min} ---")
        
        # Step 2: Update in-process environment for this iteration (no .env writes)
        set_process_env_for_run(req_min)
        
        # Steps 3-7: Run evaluation pipeline with stored variables, passing current stage and parent_dir
        evaluation_result = run_evaluation_pipeline(model, gpus, cpus, node, stage, parent_dir)
        print(f"Evaluation result: {'Success' if evaluation_result else 'Failure'}")
        if stage == 2 and not evaluation_result:
            print(f"Retry count: {retry_count_stage2}")
        
        # Step 8: Check termination condition (for stage 2)
        if stage == 2:
            should_end, result_type, result_value = end_experiment(stage, M, m, M_0, m_0, evaluation_result)
            if should_end:
                if result_type == "REQ_MIN":
                    print(f"\nExperiment completed successfully! Optimal REQ_MIN = {req_min}")
                    return req_min
                else:
                    print(f"\nExperiment completed! Result m = {result_value}")
                    return result_value
        
        # Steps 9-10: Update stage variables
        if stage == 1:
            (stage, req_min, M_0, m_0, M, m, retry_count_stage1, highest_true, lowest_false) = update_stage_1(
                evaluation_result, req_min, retry_count_stage1, highest_true, lowest_false
            )
            # If we transitioned to stage 2, reset stage-2 retry counter and log bounds
            if stage == 2:
                retry_count_stage2 = 0
                print(f"Transitioned to Stage 2: highest TRUE = {highest_true}, lowest FALSE = {lowest_false}")
        elif stage == 2:
            req_min, M, m, retry_count_stage2 = update_stage_2(
                evaluation_result, req_min, M, m, retry_count_stage2
            )

        # End-of-iteration logging: prompt and median response tokens
        prompt_text, prompt_token_count = _get_prompt_info()
        median_resp_tokens = _compute_median_response_tokens()
        print("--- Iteration summary ---")
        if prompt_text is not None:
            display_text = prompt_text if len(str(prompt_text)) <= 400 else str(prompt_text)[:400] + "..."
            print(f"Prompt: {display_text}")
        if prompt_token_count is not None:
            print(f"Prompt token count: {prompt_token_count}")
        if median_resp_tokens is not None:
            print(f"Median tokens per response: {median_resp_tokens:.3f}")
        
    # Small delay to avoid overwhelming the system
        time.sleep(1)
    
    print(f"\nReached maximum iterations ({max_iterations}). Stopping.")
    return None

def main():
    input_output_tokens = CONFIG.get('TOKENS_LIST', [])
    req_min_starts = CONFIG.get('REQ_MIN_START', [1])
    results = {}
    
    for idx, tokens in enumerate(input_output_tokens):
        print(f"\n{'='*60}")
        print(f"Starting experiment for INPUT_TOKENS={tokens[0]}, OUTPUT_TOKENS={tokens[1]}")
        print(f"{'='*60}")
        
        # Pick initial REQ_MIN by index; if not enough values, use the last one
        if req_min_starts:
            initial_req_min = req_min_starts[idx] if idx < len(req_min_starts) else req_min_starts[-1]
        else:
            initial_req_min = 1
        
        result = run_experiment_for_tokens(tokens, initial_req_min)
        results[f"{tokens[0]}_{tokens[1]}"] = result
        
        print(f"\nCompleted experiment for INPUT_TOKENS={tokens[0]}, OUTPUT_TOKENS={tokens[1]}")
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
