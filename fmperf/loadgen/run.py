import time
import copy
import requests
from typing import Iterable, List
import json
import pandas as pd
import os
from durations import Duration
import numpy as np
from fmperf.utils.approx import approx
import grpc
from google.protobuf import json_format
from fmperf.utils import parse_results
from datetime import datetime
from .collect_energy import collect_metrics, summarize_energy
from fmperf.utils.constants import REQUESTS_DIR, REQUESTS_FILENAME, RESULTS_FILENAME
import threading
import itertools
import math
import random


def run(result_filename=None):
    if result_filename is None:
        result_filename = RESULTS_FILENAME

    def get_streaming_response_tgis(response, request_timeout, ttft_timeout, tpot_timeout):
        stop = False
        generated_tokens = 0
        request_start_time = time.time_ns()
        first_token_received = False
        
        while not stop:
            try:
                # Check request timeout
                current_time = time.time_ns()
                if (current_time - request_start_time) / 1e9 > request_timeout:
                    yield None, 0, current_time, False, TimeoutError(f"Request timeout: {request_timeout}s exceeded")
                    return
                
                x = next(response)
                timestamp = time.time_ns()
                data = json_format.MessageToDict(x)
                # skip first response (tokenizer output only)
                if "inputTokenCount" not in data:
                    n_tokens = data["generatedTokenCount"] - generated_tokens
                    generated_tokens = data["generatedTokenCount"]
                    
                    # Check TTFT timeout for first token
                    if not first_token_received:
                        ttft = (timestamp - request_start_time) / 1e9
                        if ttft > ttft_timeout:
                            yield None, 0, timestamp, False, TimeoutError(f"TTFT timeout: {ttft_timeout}s exceeded (TTFT: {ttft:.3f}s)")
                            return
                        first_token_received = True
                    
                    yield data, n_tokens, timestamp, True, None
            except Exception as e:
                timestamp = time.time_ns()
                yield None, 0, timestamp, False, e

    def get_streaming_response_vllm(response, request_timeout, ttft_timeout, tpot_timeout):
        response_iter = response.iter_lines(
            chunk_size=8192,
            decode_unicode=False,
            delimiter=b"\n",
        )

        stop = False
        prev_completion_tokens = 0
        request_start_time = time.time_ns()
        first_token_received = False
        last_token_time = request_start_time
        
        while not stop:
            try:
                # Check request timeout
                current_time = time.time_ns()
                if (current_time - request_start_time) / 1e9 > request_timeout:
                    yield None, 0, current_time, False, TimeoutError(f"Request timeout: {request_timeout}s exceeded")
                    return
                
                chunk = next(response_iter)
                timestamp = time.time_ns()
                
                # Check TPOT timeout for subsequent tokens
                if first_token_received and (timestamp - last_token_time) / 1e9 > tpot_timeout:
                    yield None, 0, timestamp, False, TimeoutError(f"TPOT timeout: {tpot_timeout}s exceeded")
                    return
                
                if chunk and not stop:
                    data = chunk.decode("utf-8").strip().split("data: ")[1]
                    out = json.loads(data)["choices"][0]
                    stop = out["finish_reason"] is not None
                    usage = json.loads(data)["usage"]
                    token_count = usage["completion_tokens"] - prev_completion_tokens
                    prev_completion_tokens = usage["completion_tokens"]
                    
                    # Check TTFT timeout for first token
                    if not first_token_received:
                        ttft = (timestamp - request_start_time) / 1e9
                        if ttft > ttft_timeout:
                            yield None, 0, timestamp, False, TimeoutError(f"TTFT timeout: {ttft_timeout}s exceeded (TTFT: {ttft:.3f}s)")
                            return
                        first_token_received = True
                    
                    for i in range(token_count):
                        yield {
                            "index": out["index"],
                            "text": "" if (i < token_count - 1) else out["text"],
                            "logprobs": None,
                            "finish_reason": (
                                None if (i < token_count - 1) else out["finish_reason"]
                            ),
                            "stop_reason": (
                                None if (i < token_count - 1) else out["stop_reason"]
                            ),
                        }, 1, timestamp, True, None
                    
                    last_token_time = timestamp
            except Exception as e:
                timestamp = time.time_ns()
                yield None, 0, timestamp, False, e

        # we have stopped
        yield None, 0, time.time_ns(), False, StopIteration()

    def _parse_int_env(key):
        value = os.environ.get(key)
        if value is None or value == "":
            return None
        try:
            return int(float(value))
        except ValueError:
            return None

    def _parse_bool_env(key, default=False):
        value = os.environ.get(key)
        if value is None:
            return default
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        return default

    def _parse_tokens_list_env(value):
        tokens = []
        if not value:
            return tokens
        for item in str(value).split(','):
            raw = item.strip()
            if not raw or ':' not in raw:
                continue
            in_part, out_part = raw.split(':', 1)
            try:
                if '-' in in_part:
                    in_min_str, in_max_str = in_part.split('-', 1)
                    in_min = int(in_min_str.strip())
                    in_max = int(in_max_str.strip())
                else:
                    in_min = in_max = int(in_part.strip())
                if '-' in out_part:
                    out_min_str, out_max_str = out_part.split('-', 1)
                    out_min = int(out_min_str.strip())
                    out_max = int(out_max_str.strip())
                else:
                    out_min = out_max = int(out_part.strip())
            except ValueError:
                continue
            if in_min > in_max:
                in_min, in_max = in_max, in_min
            if out_min > out_max:
                out_min, out_max = out_max, out_min
            if in_min <= 0 or in_max <= 0 or out_min <= 0 or out_max <= 0:
                continue
            tokens.append((in_min, in_max, out_min, out_max))
        return tokens

    def _parse_float_list_env(value):
        values = []
        if value is None:
            return values
        for part in str(value).split(','):
            piece = part.strip()
            if not piece:
                continue
            try:
                values.append(float(piece))
            except ValueError:
                continue
        return values

    def _weighted_counts(total_count, weights):
        if total_count <= 0 or not weights:
            return [0 for _ in weights]
        weight_sum = sum(weights)
        if weight_sum <= 0:
            return [0 for _ in weights]
        raw = [(w / weight_sum) * total_count for w in weights]
        counts = [int(x) for x in raw]
        remaining = total_count - sum(counts)
        if remaining > 0:
            order = sorted(range(len(raw)), key=lambda idx: (raw[idx] - counts[idx]), reverse=True)
            for idx in order[:remaining]:
                counts[idx] += 1
        return counts

    def _get_output_token_override_bounds():
        min_value = _parse_int_env("MIN_OUTPUT_TOKENS")
        max_value = _parse_int_env("MAX_OUTPUT_TOKENS")
        if min_value is None and max_value is None:
            return None
        if min_value is None:
            min_value = max_value
        if max_value is None:
            max_value = min_value
        if min_value is None or max_value is None:
            return None
        if min_value <= 0 or max_value <= 0:
            raise ValueError("Output token bounds must be positive")
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        return (min_value, max_value)

    def _build_request_payload(template_request, target, rng, override_bounds, exact_output_tokens=True):
        payload = copy.deepcopy(template_request)
        if not override_bounds:
            return payload, None
        min_tokens, max_tokens = override_bounds
        if not exact_output_tokens:
            if target == "vllm":
                payload["min_tokens"] = int(min_tokens)
                payload["max_tokens"] = int(max_tokens)
            elif target == "tgis":
                params = payload.setdefault("params", {})
                stopping = params.setdefault("stopping", {})
                stopping["minNewTokens"] = int(min_tokens)
                stopping["maxNewTokens"] = int(max_tokens)
            return payload, (int(min_tokens), int(max_tokens))
        if int(max_tokens) == int(min_tokens):
            desired_tokens = int(min_tokens)
        else:
            desired_tokens = int(rng.randint(low=min_tokens, high=max_tokens + 1))
        if target == "vllm":
            payload["min_tokens"] = desired_tokens
            payload["max_tokens"] = desired_tokens
        elif target == "tgis":
            params = payload.setdefault("params", {})
            stopping = params.setdefault("stopping", {})
            stopping["minNewTokens"] = desired_tokens
            stopping["maxNewTokens"] = desired_tokens
        return payload, desired_tokens

    infile = os.path.join(REQUESTS_DIR, REQUESTS_FILENAME)
    outfile = os.path.join(REQUESTS_DIR, result_filename)
    target = os.environ["TARGET"]
    api_url = os.environ["URL"]
    req_min = float(os.environ["REQ_MIN"])  # Changed from int() to float() to allow non-integer values
    duration = Duration(os.environ["DURATION"])
    backoff = Duration(os.environ["BACKOFF"])
    grace_period = Duration(os.environ["GRACE_PERIOD"])
    
    # Get timeout values from environment variables with defaults
    request_timeout = float(os.environ.get("REQUEST_TIMEOUT", "300"))  # 5 minutes default
    ttft_timeout = float(os.environ.get("TTFT_TIMEOUT", "60"))  # 60 seconds default
    tpot_timeout = float(os.environ.get("TPOT_TIMEOUT", "30"))  # 30 seconds default

    with open(infile, "rb") as f:
        sample_requests = json.load(f)

    output_token_override = _get_output_token_override_bounds()
    additive_enabled = _parse_bool_env("ADDITIVE", default=False)
    additive_tokens = _parse_tokens_list_env(os.environ.get("TOKENS_LIST", ""))

    use_additive_scheduler = additive_enabled and len(additive_tokens) > 0
    additive_schedule = []
    additive_interval_case_indices = []
    additive_case_cursors = []
    additive_output_bounds = []
    additive_lock = threading.Lock()
    additive_schedule_cursor = itertools.count()

    if use_additive_scheduler:
        additive_weights = _parse_float_list_env(os.environ.get("TOKENS_LIST_PROPORTION"))
        cleaned_weights = [max(0.0, float(x)) for x in additive_weights]
        if len(cleaned_weights) < len(additive_tokens):
            cleaned_weights.extend([1.0] * (len(additive_tokens) - len(cleaned_weights)))
        elif len(cleaned_weights) > len(additive_tokens):
            cleaned_weights = cleaned_weights[:len(additive_tokens)]
        if not cleaned_weights or all(weight == 0.0 for weight in cleaned_weights):
            cleaned_weights = [1.0 for _ in additive_tokens]

        estimated_requests = int(math.ceil((req_min * duration.to_seconds() * 1.1) / 60.0))
        estimated_requests = max(1, estimated_requests)
        additive_counts = _weighted_counts(estimated_requests, cleaned_weights)
        for interval_idx, interval_count in enumerate(additive_counts):
            additive_schedule.extend([interval_idx] * interval_count)
        random.Random(42).shuffle(additive_schedule)

        additive_output_bounds = [(tokens[2], tokens[3]) for tokens in additive_tokens]
        additive_interval_case_indices = [[] for _ in additive_tokens]
        additive_case_cursors = [0 for _ in additive_tokens]
        for sample_idx, sample in enumerate(sample_requests):
            config = sample.get("config", {}) if isinstance(sample, dict) else {}
            in_tokens = config.get("in_tokens")
            if in_tokens is None:
                request_obj = sample.get("request", {}) if isinstance(sample, dict) else {}
                prompt_ids = request_obj.get("prompt") if isinstance(request_obj, dict) else None
                if isinstance(prompt_ids, list):
                    in_tokens = len(prompt_ids)
            if in_tokens is None:
                continue
            try:
                in_tokens = int(in_tokens)
            except (TypeError, ValueError):
                continue
            for interval_idx, interval in enumerate(additive_tokens):
                in_min, in_max, _, _ = interval
                if in_min <= in_tokens <= in_max:
                    additive_interval_case_indices[interval_idx].append(sample_idx)

        empty_intervals = [
            idx for idx, indices in enumerate(additive_interval_case_indices) if len(indices) == 0
        ]
        if empty_intervals:
            print(
                "Warning: additive scheduler found no prompt candidates for intervals: "
                + ", ".join(str(additive_tokens[idx]) for idx in empty_intervals)
            )

        print(
            f">> Additive scheduler enabled with {len(additive_tokens)} intervals, "
            f"{len(additive_schedule)} scheduled slots (10% margin)."
        )
    else:
        if additive_enabled:
            print(
                "Warning: ADDITIVE is enabled but TOKENS_LIST is empty/invalid; "
                "falling back to random request sampling."
            )

    all_sample_indices = list(range(len(sample_requests)))

    def worker(wid, channel, worker_req_per_sec, exp_num_users):
        rs = np.random.RandomState(seed=wid)
        rs_lock = threading.Lock()
        stub = None
        if target == "tgis":
            from text_generation_tests.pb import generation_pb2_grpc as gpb2
            stub = gpb2.GenerationServiceStub(channel)
        
        # Calculate requests per second for this worker with some randomness
        # worker_req_per_sec is the target per worker (REQ_MIN split by num_workers)
        variation = rs.uniform(0.8, 1.2)
        worker_req_per_sec = worker_req_per_sec * variation
        
        # Calculate interval between requests for this worker
        if worker_req_per_sec > 0:
            base_interval = 1.0 / worker_req_per_sec
            jitter_range = 0.3  # ±30% jitter
        else:
            base_interval = float('inf')
            jitter_range = 0.0

        t_start = time.time_ns()

        output = []
        output_lock = threading.Lock()
        request_counter = itertools.count()
        requests_scheduled = 0

        # Track in-flight requests for cleanup; agnostic to expected duration
        inflight = set()  # set[threading.Thread]

        # progress logging: print remaining every LOG_INTERVAL seconds
        LOG_INTERVAL = 5.0
        last_log_time = t_start
        
        # Driftless scheduler: schedule first request immediately and then at fixed intervals with jitter
        # This avoids skipping the first interval window which caused under-sending at higher RPMs.
        next_request_time = t_start  # first request goes out immediately
        if worker_req_per_sec > 0:
            interval_base_ns = (1.0 / worker_req_per_sec) * 1e9
        else:
            interval_base_ns = float('inf')

        def process_request(req_idx):
            # For additive mode, pick from a precomputed global schedule and wrap around as needed.
            per_request_override = output_token_override
            exact_output_tokens = True
            if use_additive_scheduler:
                with additive_lock:
                    schedule_pos = next(additive_schedule_cursor)
                    schedule_idx = schedule_pos % len(additive_schedule)
                    interval_idx = additive_schedule[schedule_idx]
                    interval_candidates = additive_interval_case_indices[interval_idx]
                    if interval_candidates:
                        cursor = additive_case_cursors[interval_idx]
                        sample_idx = interval_candidates[cursor % len(interval_candidates)]
                        additive_case_cursors[interval_idx] = cursor + 1
                    else:
                        sample_idx = all_sample_indices[schedule_pos % len(all_sample_indices)]
                    per_request_override = additive_output_bounds[interval_idx]
                    exact_output_tokens = False
            else:
                # Pick a sample request (thread-safe selection)
                with rs_lock:
                    sample_idx = rs.randint(low=0, high=len(sample_requests))
            template_request = sample_requests[sample_idx]["request"]
            with rs_lock:
                request_payload, _ = _build_request_payload(
                    template_request,
                    target,
                    rs,
                    per_request_override,
                    exact_output_tokens=exact_output_tokens,
                )

            if target == "vllm":
                headers = {"User-Agent": "fmaas-load-test"}
                t0 = time.time_ns()
                try:
                    response = requests.post(
                        "http://%s/v1/completions" % (api_url),
                        headers=headers,
                        json=request_payload,
                        stream=True,
                        timeout=request_timeout
                    )
                except requests.exceptions.Timeout:
                    timestamp = time.time_ns()
                    record = {
                        "response": None,
                        "ok": False,
                        "error": f"Request timeout: {request_timeout}s exceeded",
                        "timestamp": timestamp,
                        "exp_req_min": req_min,
                        "exp_duration": duration.to_seconds(),
                        "duration_ms": (timestamp - t0) / 1000.0 / 1000.0,
                        "exclude": (timestamp - t_start) / 1000.0 / 1000.0 / 1000.0
                        > (duration.to_seconds() + grace_period.to_seconds()),
                        "worker_idx": wid,
                        "request_idx": req_idx,
                        "sample_idx": sample_idx,
                        "response_idx": 0,
                        "n_tokens": 0,
                        "exp_num_users": exp_num_users,
                    }
                    with output_lock:
                        output.append(record)
                    time.sleep(backoff.to_seconds())
                    return True
            elif target == "tgis":
                from text_generation_tests.pb import generation_pb2 as pb2
                message = json_format.ParseDict(request_payload, pb2.SingleGenerationRequest())
                t0 = time.time_ns()
                response = stub.GenerateStream(message)
            else:
                raise ValueError(f"Invalid target: {target}")

            stop = False
            response_idx = 0

            if target == "vllm":
                response_generator = get_streaming_response_vllm(response, request_timeout, ttft_timeout, tpot_timeout)
            elif target == "tgis":
                response_generator = get_streaming_response_tgis(response, request_timeout, ttft_timeout, tpot_timeout)
            else:
                raise ValueError(f"Invalid target: {target}")

            apply_backoff = False

            while not stop:
                r, n_tokens, t, ok, err = next(response_generator)

                if not ok:
                    stop = True
                    # check if we have reached end of stream
                    if type(err) is StopIteration:
                        continue
                    else:
                        apply_backoff = True

                record = {
                    "response": r,
                    "ok": ok,
                    "error": str(err),
                    "timestamp": t,
                    "exp_req_min": req_min,
                    "exp_duration": duration.to_seconds(),
                    "duration_ms": (t - t0) / 1000.0 / 1000.0,
                    "exclude": (t - t_start) / 1000.0 / 1000.0 / 1000.0
                    > (duration.to_seconds() + grace_period.to_seconds()),
                    "worker_idx": wid,
                    "request_idx": req_idx,
                    "sample_idx": sample_idx,
                    "response_idx": response_idx,
                    "n_tokens": n_tokens,
                    "exp_num_users": exp_num_users,
                }

                with output_lock:
                    output.append(record)
                response_idx += 1
                t0 = t

            if apply_backoff:
                time.sleep(backoff.to_seconds())

            return True

        # Scheduler loop: use next_request_time to avoid drift and ensure the first request is immediate
        # Only schedule when we have a positive target rate
        while worker_req_per_sec > 0 and (next_request_time - t_start) < duration.to_seconds() * 1e9:
            current_time = time.time_ns()
            if current_time < next_request_time:
                time.sleep((next_request_time - current_time) / 1e9)

            # Prune finished threads
            finished = [t for t in inflight if not t.is_alive()]
            for t in finished:
                inflight.discard(t)

            # Schedule a new request by starting a dedicated thread
            req_idx = next(request_counter)
            th = threading.Thread(target=process_request, args=(req_idx,), daemon=True)
            th.start()
            inflight.add(th)
            requests_scheduled += 1
            
            # Compute next schedule time with jitter around the base interval
            jitter = rs.uniform(1 - jitter_range, 1 + jitter_range)
            next_request_time += int(interval_base_ns * jitter)

            # progress logging: print remaining time every LOG_INTERVAL seconds
            now_ns = time.time_ns()
            elapsed_s = (now_ns - t_start) / 1e9
            remaining_s = duration.to_seconds() - elapsed_s
            if remaining_s < 0:
                remaining_s = 0.0
            if (now_ns - last_log_time) / 1e9 >= LOG_INTERVAL:
                print(f"[worker {wid}] remaining: {remaining_s:.1f}s (elapsed: {elapsed_s:.1f}s, reqs scheduled: {requests_scheduled}, inflight: {len(inflight)})")
                last_log_time = now_ns

        # Wait for all in-flight request threads to finish
        for th in list(inflight):
            th.join()

        with open("results_wid%d" % (wid), "w") as f:
            json.dump(output, f)

        return True

    from datetime import datetime
    import concurrent.futures

    energy_start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    channel = grpc.insecure_channel(api_url) if target == "tgis" else None

    # Determine number of workers dynamically based on target RPM and per-worker capacity
    per_worker_rpm_capacity = int(os.environ.get("WORKER_RPM_CAPACITY", "15"))
    max_workers = int(os.environ.get("MAX_WORKERS", "64"))
    num_workers = max(1, min(max_workers, int(math.ceil(req_min / max(1, per_worker_rpm_capacity)))))
    print(f">> Using {num_workers} workers (capacity ~{per_worker_rpm_capacity} rpm/worker) to achieve {req_min} requests per minute")
    # Split the global REQ_MIN target across workers
    per_worker_req_per_sec = (req_min / max(1, num_workers)) / 60.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            futures.append(
                executor.submit(
                    worker,
                    wid=i,
                    channel=channel,
                    worker_req_per_sec=per_worker_req_per_sec,
                    exp_num_users=num_workers,
                )
            )

        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    energy_stop_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    all_outputs = []
    for i in range(num_workers):
        with open("results_wid%d" % (i), "rb") as f:
            tmp = json.load(f)
        all_outputs.extend(tmp)

    def check_consistent(row):
        if not row["ok"]:
            return False
        case = sample_requests[row["sample_idx"]]
        expected = case.get("expected")
        if not expected:
            return False
        if row["response_idx"] >= len(expected):
            return False
        tmp = expected[row["response_idx"]]
        consistent = row["response"] == approx(tmp)
        return consistent

    for row in all_outputs:
        row["consistent"] = check_consistent(row)

    # collect and summarize energy metrics
    energy = {}
    if os.environ.get("PROM_URL") is None:
        print(
            ">> skipped collecting energy metrics because prometheus is not available."
        )
    else:
        step = os.environ.get("NUM_PROM_STEPS", "30")
        ns = os.environ["NAMESPACE"]
        collect_metrics(energy_start_time, energy_stop_time, step, ns)
        all_energy_metrics = summarize_energy(energy_start_time)
        print(all_energy_metrics)
        energy = all_energy_metrics[["num_users", "energy"]].to_dict()

    merged_data = {"results": all_outputs, "energy": energy}

    print(">> writing results to file: %s" % (outfile))
    with open(outfile, "w") as f:
        json.dump(merged_data, f)

    return all_outputs


if __name__ == "__main__":
    parse_results(run(), print_df=True)