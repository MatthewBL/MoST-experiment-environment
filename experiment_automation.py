import os
import subprocess
import time
from pathlib import Path

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

def load_env_config():
    """Load configuration from .env file and return a dict.

    Expected keys:
    - TOKENS_LIST: comma-separated pairs like "32:32,32:64"
    - REQ_MIN_START: integer
    - REQ_MIN_INCREASE_MULTIPLIER: integer (multiplier for stage 1 success)
    - STOP_THRESHOLD: float (used in end condition)
    """
    env_path = Path('.env')
    config = {
        'TOKENS_LIST': [],
        'REQ_MIN_START': 1,
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
                    try:
                        config['REQ_MIN_START'] = int(val)
                    except ValueError:
                        pass
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
        
        # Early metrics check before splitting results
        early_fail = False
        try:
            import importlib
            am = importlib.import_module("requests/analyze_metrics")
            events = am.load_folder_events(".")
            metrics = am.compute_metrics(events)
            avg_resp = float(metrics.get("avg_responded_per_minute", 0.0))
            req_min_env = os.environ.get('REQ_MIN', '0')
            try:
                req_min_val = float(req_min_env)
            except ValueError:
                req_min_val = 0.0
            if avg_resp < (0.95 * req_min_val):
                print(f"Early evaluation failure: avg responded/min = {avg_resp:.3f} < 70% of REQ_MIN = {req_min_val}")
                early_fail = True
            else:
                print(f"Early metrics check passed: avg responded/min = {avg_resp:.3f}, REQ_MIN = {req_min_val}")
        except Exception as e:
            # Don't block on metrics errors; proceed with normal flow
            print(f"Warning: analyze_metrics check failed: {e}")
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
        
        # Step 8: Store results with environment variables as arguments, including stage and parent_dir
        store_command = f'python -u store_results.py "{model}" "{gpus}" "{cpus}" "{node}" "{stage}" "{parent_dir}"'
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

def update_stage_1(evaluation, current_req_min, retry_count_stage1):
    """Update logic for stage 1 with a retry mechanism.

    Behavior (reasonable assumption):
    - On success: double REQ_MIN and reset stage-1 retry counter.
    - On first failure: stay in stage 1 and increment retry counter (retry same REQ_MIN).
    - On second consecutive failure: transition to stage 2 (initialize M/m) and reset stage-1 retry counter.

    Returns: (stage, new_req_min, M_0, m_0, M, m, retry_count_stage1)
    """
    multiplier = CONFIG.get('REQ_MIN_INCREASE_MULTIPLIER', 2)
    if evaluation:
        # Successful evaluation: increase REQ_MIN by configured multiplier and continue
        new_req_min = current_req_min * multiplier
        retry_count_stage1 = 0
        return 1, new_req_min, None, None, None, None, retry_count_stage1
    else:
        # Failed evaluation: implement retry behavior for stage 1
        if retry_count_stage1 == 0:
            # First failure: don't transition yet, retry same req_min
            retry_count_stage1 = 1
            return 1, current_req_min, None, None, None, None, retry_count_stage1
        else:
            # Second consecutive failure: transition to stage 2
            M_0 = current_req_min
            m_0 = current_req_min / 2
            stage = 2
            M = M_0
            m = m_0
            new_req_min = (M + m) / 2
            retry_count_stage1 = 0
            return stage, new_req_min, M_0, m_0, M, m, retry_count_stage1

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

def run_experiment_for_tokens(tokens):
    """Run the complete experiment for a specific token combination"""
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
    req_min = CONFIG.get('REQ_MIN_START', 1)  # Initial value from .env
    
    # Stage 2 variables (initialized when transitioning to stage 2)
    M_0, m_0, M, m = None, None, None, None
    
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
            (stage, req_min, M_0, m_0, M, m, retry_count_stage1) = update_stage_1(
                evaluation_result, req_min, retry_count_stage1
            )
            # If we transitioned to stage 2, reset stage-2 retry counter
            if stage == 2:
                retry_count_stage2 = 0
        elif stage == 2:
            req_min, M, m, retry_count_stage2 = update_stage_2(
                evaluation_result, req_min, M, m, retry_count_stage2
            )
        
    # Small delay to avoid overwhelming the system
        time.sleep(1)
    
    print(f"\nReached maximum iterations ({max_iterations}). Stopping.")
    return None

def main():
    input_output_tokens = CONFIG.get('TOKENS_LIST', [])
    results = {}
    
    for tokens in input_output_tokens:
        print(f"\n{'='*60}")
        print(f"Starting experiment for INPUT_TOKENS={tokens[0]}, OUTPUT_TOKENS={tokens[1]}")
        print(f"{'='*60}")
        
        result = run_experiment_for_tokens(tokens)
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
