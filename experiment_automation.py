import os
import subprocess
import time
from pathlib import Path

def modify_env_file(req_min_value, input_tokens=None, output_tokens=None):
    """
    Modify the .env file to change REQ_MIN, MIN_INPUT_TOKENS, MAX_INPUT_TOKENS, 
    MIN_OUTPUT_TOKENS and MAX_OUTPUT_TOKENS values. input_tokens and output_tokens
    are optional; if None they won't be added/changed.
    """
    env_file = Path('.env')

    if not env_file.exists():
        raise FileNotFoundError(".env file not found")

    with env_file.open('r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    req_min_found = False
    input_found = False
    output_found = False

    for line in lines:
        stripped = line.strip()
        # Preserve blank lines and comments
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            new_lines.append(line)
            continue

        key, _ = stripped.split('=', 1)
        if key == 'REQ_MIN':
            new_lines.append(f'REQ_MIN={req_min_value}\n')
            req_min_found = True
        elif input_tokens is not None and key == 'MIN_INPUT_TOKENS':
            new_lines.append(f'MIN_INPUT_TOKENS={input_tokens}\n')
            input_found = True
        elif output_tokens is not None and key == 'MIN_OUTPUT_TOKENS':
            new_lines.append(f'MIN_OUTPUT_TOKENS={output_tokens}\n')
            output_found = True
        elif input_tokens is not None and key == 'MAX_INPUT_TOKENS':
            new_lines.append(f'MAX_INPUT_TOKENS={input_tokens}\n')
            input_found = True
        elif output_tokens is not None and key == 'MAX_OUTPUT_TOKENS':
            new_lines.append(f'MAX_OUTPUT_TOKENS={output_tokens}\n')
            output_found = True
        else:
            new_lines.append(line)

    # Append any missing variables (only if corresponding arg was provided for tokens)
    if not req_min_found:
        new_lines.append(f'REQ_MIN={req_min_value}\n')
    if input_tokens is not None and not input_found:
        new_lines.append(f'INPUT_TOKENS={input_tokens}\n')
    if output_tokens is not None and not output_found:
        new_lines.append(f'OUTPUT_TOKENS={output_tokens}\n')

    with env_file.open('w', encoding='utf-8') as f:
        f.writelines(new_lines)

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
        
        # Step 6: Split results
        run_command("python -u split_results.py")
        
        # Step 7: Run evaluation and capture result
        result = run_command("python -u evaluate.py", wait=True)
        
        # Step 8: Store results with environment variables as arguments, including stage and parent_dir
        store_command = f'python -u store_results.py "{model}" "{gpus}" "{cpus}" "{node}" "{stage}" "{parent_dir}"'
        run_command(store_command)
        
        # For this example, we'll assume evaluation.py returns 0 for success, non-zero for failure
        # You may need to modify this based on how your evaluation.py actually indicates success
        evaluation_success = (result.returncode == 0)
        
        return evaluation_success
        
    finally:
        # Always return to original directory
        os.chdir(original_dir)

def start_stage_1():
    """Initialize stage 1"""
    return 1

def end_experiment(stage, M, m, M_0, m_0, evaluation):
    """Check termination condition for stage 2"""
    if stage == 2 and (M - m) <= 0.5:
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
    if evaluation:
        # Successful evaluation: double REQ_MIN and continue
        new_req_min = current_req_min * 2
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
    req_min = 1  # Initial value
    
    # Stage 2 variables (initialized when transitioning to stage 2)
    M_0, m_0, M, m = None, None, None, None
    
    # Retry counters for stage 1 and stage 2
    retry_count_stage1 = 0
    retry_count_stage2 = 0
    
    max_iterations = 100  # Safety limit to prevent infinite loops
    iteration = 0

    modify_env_file(req_min, input_tokens=tokens[0], output_tokens=tokens[1])
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
        
        # Step 2: Modify .env file
        modify_env_file(req_min)
        
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
    input_output_tokens = [
        [32, 32],
        [32, 64],
	[32, 128],
	[64, 32],
	[64, 64],
	[64, 128],
	[128, 32],
	[128, 64],
	[128, 128]
        ]
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
