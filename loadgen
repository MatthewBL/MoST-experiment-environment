import time
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

    def worker(wid, channel):
        rs = np.random.RandomState(seed=wid)
        
        # Calculate requests per second for this worker with some randomness
        base_req_per_sec = req_min / 60.0
        # Add ±20% variation to create uneven distribution
        variation = rs.uniform(0.8, 1.2)
        worker_req_per_sec = base_req_per_sec * variation
        
        # Calculate interval between requests for this worker
        if worker_req_per_sec > 0:
            base_interval = 1.0 / worker_req_per_sec
            # Add some jitter (±30%) to make distribution more uneven
            jitter_range = 0.3
        else:
            base_interval = float('inf')
            jitter_range = 0

        t_start = time.time_ns()

        output = []
        request_idx = 0

        # progress logging: print remaining every LOG_INTERVAL seconds
        LOG_INTERVAL = 5.0
        last_log_time = t_start
        last_request_time = t_start

        while (
            time.time_ns() - t_start < duration.to_seconds() * 1000.0 * 1000.0 * 1000.0
        ):
            # Calculate next request time with jitter
            if worker_req_per_sec > 0:
                jitter = rs.uniform(1 - jitter_range, 1 + jitter_range)
                next_interval_ns = base_interval * jitter * 1e9
                next_request_time = last_request_time + next_interval_ns
                
                # Sleep until next request time
                current_time = time.time_ns()
                if current_time < next_request_time:
                    sleep_time = (next_request_time - current_time) / 1e9
                    time.sleep(sleep_time)

            sample_idx = rs.randint(low=0, high=len(sample_requests))
            sample_request = sample_requests[sample_idx]["request"]

            if target == "vllm":  # StackSpec will also use this
                headers = {"User-Agent": "fmaas-load-test"}
                t0 = time.time_ns()
                try:
                    response = requests.post(
                        "http://%s/v1/completions" % (api_url),
                        headers=headers,
                        json=sample_request,
                        stream=True,
                        timeout=request_timeout  # Add request timeout
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
                        "request_idx": request_idx,
                        "sample_idx": sample_idx,
                        "response_idx": 0,
                        "n_tokens": 0,
                    }
                    output.append(record)
                    request_idx += 1
                    last_request_time = timestamp
                    time.sleep(backoff.to_seconds())
                    continue
            elif target == "tgis":
                from text_generation_tests.pb import generation_pb2 as pb2

                message = json_format.ParseDict(
                    sample_request, pb2.SingleGenerationRequest()
                )
                t0 = time.time_ns()
                response = stub.GenerateStream(message)
            else:
                raise ValueError(f"Invalid target: {target}")

            stop = False
            response_idx = 0

            if target == "vllm":  # StackSpec will also use this
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
                    "request_idx": request_idx,
                    "sample_idx": sample_idx,
                    "response_idx": response_idx,
                    "n_tokens": n_tokens,
                }

                output.append(record)
                response_idx += 1
                t0 = t

            if apply_backoff:
                time.sleep(backoff.to_seconds())

            request_idx += 1
            last_request_time = time.time_ns()

            # progress logging: print remaining time every LOG_INTERVAL seconds
            now_ns = time.time_ns()
            elapsed_s = (now_ns - t_start) / 1e9
            remaining_s = duration.to_seconds() - elapsed_s
            if remaining_s < 0:
                remaining_s = 0.0
            if (now_ns - last_log_time) / 1e9 >= LOG_INTERVAL:
                print(f"[worker {wid}] remaining: {remaining_s:.1f}s (elapsed: {elapsed_s:.1f}s, reqs: {request_idx})")
                last_log_time = now_ns

        with open("results_wid%d" % (wid), "w") as f:
            json.dump(output, f)

        return True

    from datetime import datetime
    import concurrent.futures

    energy_start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    channel = grpc.insecure_channel(api_url) if target == "tgis" else None

    # Use a fixed number of workers (you can adjust this based on your needs)
    # Using 4 workers as a reasonable default for distributing requests
    num_workers = min(4, max(1, int(req_min // 15)))  # Changed to int() for worker count
    print(f">> Using {num_workers} workers to achieve {req_min} requests per minute")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i in range(num_workers):
            futures.append(executor.submit(worker, wid=i, channel=channel))

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
        if row["ok"]:
            tmp = sample_requests[row["sample_idx"]]["expected"]
            tmp = tmp[row["response_idx"]]
            consistent = row["response"] == approx(tmp)
            return consistent
        else:
            return False

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