"""Resolve the number of GPUs used by a Slurm job serving an LLM.

Shared by the MoST experiment environment (to store GPU_COUNT in results.csv)
and the MoST API (spawned via `python GpuCount.py find --model ...`).
"""
import argparse
import csv
import json
import re
import subprocess
import sys

GPU_COUNT_FIELD = "GPU_COUNT"

_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9:_-]*$")
DEFAULT_TIMEOUT_SECONDS = 10.0


class GpuCountError(Exception):
    """Raised when the GPU count cannot be resolved from Slurm."""

    def __init__(self, message, code):
        super().__init__(message)
        self.message = message
        self.code = code


def run_command_capture_stdout(command, args, timeout_seconds=DEFAULT_TIMEOUT_SECONDS):
    """Run a command and return its stdout, or None on any failure."""
    try:
        result = subprocess.run(
            [command] + list(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def parse_scontrol_output(stdout):
    """Parse `scontrol show job` output into a flat dict, mirroring the MoST API."""
    fields = {}
    current_key = None
    if stdout is None:
        return fields
    for token in str(stdout).split():
        eq_index = token.find("=")
        if eq_index > 0:
            key = token[:eq_index]
            if _KEY_RE.fullmatch(key):
                current_key = key
                fields[current_key] = token[eq_index + 1:]
                continue
        if current_key is not None:
            fields[current_key] = fields.get(current_key, "") + " " + token
    return fields


def node_range_includes_node(candidate, target_node):
    """Return True when target_node falls inside a Slurm range like gpu[07-08]."""
    match = re.match(r"^(.*?)\[([^\]]+)\](.*)$", candidate)
    if not match:
        return False

    prefix, body, suffix = match.groups()
    if not target_node.startswith(prefix) or not target_node.endswith(suffix):
        return False

    numeric_part = target_node[len(prefix):len(target_node) - len(suffix)]
    if not numeric_part.isdigit():
        return False

    value = int(numeric_part)
    width = len(numeric_part)

    for segment in body.split(","):
        trimmed = segment.strip()
        range_match = re.fullmatch(r"(\d+)-(\d+)", trimmed)
        if not range_match:
            if trimmed == numeric_part:
                return True
            continue
        start, end = int(range_match.group(1)), int(range_match.group(2))
        if value < start or value > end:
            continue
        # Zero-padded ranges (e.g. gpu[07-08]) only match same-width names.
        if len(range_match.group(1)) != len(range_match.group(2)) or width == len(range_match.group(1)):
            return True

    return False


def node_list_includes_node(node_list, target_node):
    """Return True when target_node appears in a node list (plain or ranged)."""
    for candidate in re.split(r"[\s,]+", str(node_list or "")):
        if not candidate:
            continue
        if candidate == target_node or node_range_includes_node(candidate, target_node):
            return True
    return False
def get_running_jobs_on_node(node):
    """Return the ids of running Slurm jobs allocated on `node` (None if squeue fails)."""
    stdout = run_command_capture_stdout(
        "squeue", ["--noheader", "-t", "RUNNING", "--format=%i %N"]
    )
    if stdout is None:
        return None

    job_ids = []
    for line in stdout.splitlines():
        match = re.match(r"^\s*(\d+)\s+(.+?)\s*$", line)
        if not match or not node_list_includes_node(match.group(2).strip(), node):
            continue
        job_ids.append(int(match.group(1)))
    return job_ids


def get_job_info(job_id):
    """Return scontrol job info for `job_id` as a dict (None if unavailable)."""
    stdout = run_command_capture_stdout("scontrol", ["show", "job", str(job_id)])
    return None if stdout is None else parse_scontrol_output(stdout)


def extract_gpu_count_from_tres_per_job(tres_per_job):
    """Extract the gpu:N count from a TresPerJob value like 'gpu:4'."""
    parts = str(tres_per_job or "").split(":")
    if not parts:
        return None
    try:
        count = float(parts[-1])
    except (TypeError, ValueError):
        return None
    if count.is_integer() and count >= 0:
        return int(count)
    return None


def command_model_matches(command_model, model_id):
    """Match a model id against the model token of a job Command line."""
    expected = str(model_id or "").strip()
    actual = str(command_model or "").strip()
    if not expected or not actual:
        return False
    if actual == expected:
        return True
    # Accept the last path segment of the model id, e.g. "Llama-3.3-70B-Instruct".
    return actual.split("/")[-1] == expected.split("/")[-1]


def find_model_job_gpu_count(model_id, node, port):
    """Find the running Slurm job serving model_id on node:port.

    Returns a dict with jobId/node/model/port/gpuCount/source, or raises
    GpuCountError with code SQUEUE_UNAVAILABLE / JOB_NOT_FOUND.
    """
    running_job_ids = get_running_jobs_on_node(node)
    if running_job_ids is None:
        raise GpuCountError("squeue is unavailable.", "SQUEUE_UNAVAILABLE")

    if not running_job_ids:
        raise GpuCountError('No running jobs found on node "%s".' % node, "JOB_NOT_FOUND")

    port = str(port)
    for job_id in running_job_ids:
        info = get_job_info(job_id)
        if not info or info.get("JobState") != "RUNNING":
            continue

        command_parts = [part for part in str(info.get("Command") or "").split() if part]
        if len(command_parts) < 3:
            # Not a model-serving job: the command has no model and port.
            continue

        command_model = command_parts[1]
        command_port = command_parts[2]
        if not command_model_matches(command_model, model_id):
            continue
        if command_port != port:
            continue

        gpu_count = extract_gpu_count_from_tres_per_job(info.get("TresPerJob"))
        if gpu_count is None:
            continue

        return {
            "jobId": job_id,
            "node": node,
            "model": command_model,
            "port": command_port,
            "gpuCount": gpu_count,
            "source": "slurm",
        }

    raise GpuCountError(
        'No running job found on node "%s" serving model "%s" on port "%s".'
        % (node, model_id, port),
        "JOB_NOT_FOUND",
    )

def read_gpu_count_from_results_csv(csv_path):
    """Read GPU_COUNT from the first row of a results.csv (None when missing)."""
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for key in (GPU_COUNT_FIELD, "gpu_count"):
                    value = row.get(key)
                    if value is not None and str(value).strip() != "":
                        try:
                            return int(float(str(value).strip()))
                        except (TypeError, ValueError):
                            return str(value).strip()
                break
    except Exception:
        return None
    return None

def _main(argv=None):
    parser = argparse.ArgumentParser(
        prog="GpuCount",
        description="Resolve the number of GPUs used by a Slurm job serving an LLM.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    find_parser = subparsers.add_parser(
        "find", help="Find the GPU count of the running job serving a model on a node and port."
    )
    find_parser.add_argument("--model", required=True)
    find_parser.add_argument("--node", required=True)
    find_parser.add_argument("--port", required=True)

    csv_parser = subparsers.add_parser(
        "from-csv", help="Read the GPU count stored in a results.csv file."
    )
    csv_parser.add_argument("--path", required=True)

    args = parser.parse_args(argv)

    if args.command == "find":
        try:
            result = find_model_job_gpu_count(args.model, args.node, args.port)
        except GpuCountError as error:
            print(json.dumps({"error": error.message, "code": error.code}))
            return 1
        except Exception as error:
            print(json.dumps({"error": str(error), "code": "UNKNOWN_ERROR"}))
            return 1
        print(json.dumps(result))
        return 0

    count = read_gpu_count_from_results_csv(args.path)
    print(json.dumps({"gpuCount": count, "source": args.path, "field": GPU_COUNT_FIELD}))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
