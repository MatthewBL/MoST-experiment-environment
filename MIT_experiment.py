import os
import subprocess
import time
from pathlib import Path

# Simpler experiment runner:
# - Iterates over a predefined list of INPUT/OUTPUT token pairs
# - For each pair, starts at REQ_MIN=1, runs loadgen
# - After each iteration, doubles REQ_MIN (exponential growth)
# - Stops when REQ_MIN reaches 128


def update_env(req_min: int, input_tokens: int, output_tokens: int) -> None:
    """Update or create .env with the required keys for this experiment.

    Preserves existing comments/keys where possible and updates only:
    - REQ_MIN
    - MIN_INPUT_TOKENS / MAX_INPUT_TOKENS
    - MIN_OUTPUT_TOKENS / MAX_OUTPUT_TOKENS
    """
    env_file = Path('.env')

    lines = []
    if env_file.exists():
        lines = env_file.read_text(encoding='utf-8').splitlines(True)  # keep line endings

    target_values = {
        'REQ_MIN': str(req_min),
        'MIN_INPUT_TOKENS': str(input_tokens),
        'MAX_INPUT_TOKENS': str(input_tokens),
        'MIN_OUTPUT_TOKENS': str(output_tokens),
        'MAX_OUTPUT_TOKENS': str(output_tokens),
    }

    found = {k: False for k in target_values.keys()}
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            new_lines.append(line)
            continue
        key, _ = stripped.split('=', 1)
        if key in target_values:
            new_lines.append(f"{key}={target_values[key]}\n")
            found[key] = True
        else:
            new_lines.append(line)

    # Append missing keys at the end if they were not found
    for key, value in target_values.items():
        if not found[key]:
            new_lines.append(f"{key}={value}\n")

    env_file.write_text(''.join(new_lines), encoding='utf-8')


def run_cmd(command: str, wait: bool = True) -> subprocess.Popen:
    print(f"Running: {command}")
    proc = subprocess.Popen(command, shell=True)
    if wait:
        proc.wait()
        if proc.returncode != 0:
            print(f"Warning: Command '{command}' returned non-zero exit code: {proc.returncode}")
    return proc


def run_single_iteration(req_min: int, input_tokens: int, output_tokens: int) -> None:
    """Update env for the iteration and run the load generator."""
    update_env(req_min, input_tokens, output_tokens)

    # Optionally regenerate inputs before running loadgen; keep it minimal.
    run_cmd("python -u -m fmperf.loadgen.generate-input", wait=True)

    # Send traffic per the current REQ_MIN
    run_cmd("python -u -m fmperf.loadgen.run", wait=True)

    date_str = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())

    run_cmd("mv requests/results.json requests/MIT/results_REQMIN{}_IN{}_OUT{}.json".format(req_min, input_tokens, output_tokens), wait=True)


def run_simple_experiments() -> dict:
    """Run simple experiments across token pairs, doubling REQ_MIN until 128."""
    input_output_tokens = [
        [128, 128],
        [128, 256],
        [128, 512],
        [128, 1024],
        [128, 2048],
        [256, 128],
        [256, 256],
        [256, 512],
        [256, 1024],
        [256, 2048],
        [512, 128],
        [512, 256],
        [512, 512],
        [512, 1024],
        [512, 2048],
        [1024, 128],
        [1024, 256],
        [1024, 512],
        [1024, 1024],
        [1024, 2048],
        [2048, 128],
        [2048, 256],
        [2048, 512],
        [2048, 1024],
        [2048, 2048],
    ]

    results = {}

    for tokens in input_output_tokens:
        input_tokens, output_tokens = tokens
        print("\n" + "=" * 60)
        print(f"Starting simple experiment for INPUT_TOKENS={input_tokens}, OUTPUT_TOKENS={output_tokens}")
        print("=" * 60)

        req_min = 128
        iterations = 0
        while True:
            iterations += 1
            print(f"\n--- Iteration {iterations}: REQ_MIN={req_min}, INPUT={input_tokens}, OUTPUT={output_tokens} ---")
            run_single_iteration(req_min, input_tokens, output_tokens)

            # End experiment once REQ_MIN reaches 128 (include this iteration)
            if req_min == 2048:
                print("Reached REQ_MIN=128. Ending experiment for this token pair.")
                break

            # Exponential increase (doubling)
            req_min *= 2

            # Small delay to avoid overwhelming the system
            time.sleep(1)

        results[f"{input_tokens}_{output_tokens}"] = 2048  # Final REQ_MIN reached
        print(f"Completed token pair {input_tokens}_{output_tokens} at REQ_MIN=128")

    print("\n" + "=" * 60)
    print("ALL SIMPLE EXPERIMENTS COMPLETED")
    print("=" * 60)
    for token_combo, result in results.items():
        print(f"Tokens {token_combo}: Final REQ_MIN {result}")

    return results


if __name__ == "__main__":
    final_results = run_simple_experiments()
    print(f"Final results: {final_results}")
