import argparse
import time
import numpy as np
import sys
import json
import os
import grpc
import pickle
from google.protobuf import json_format
import requests
from typing import Iterable, List
from importlib import resources as impresources
import fmperf.data
import traceback
from transformers import AutoTokenizer
from fmperf.utils.constants import REQUESTS_DIR
from dotenv import load_dotenv

load_dotenv()

code = os.getenv("CODE", "false").lower() != "false"

if code:
    import fmperf

    tmp = fmperf.__file__.split("/")[:-1]
    tmp.append("Cluster.py")
    seed_text_file = "/".join(tmp)
else:
    seed_text_file = impresources.files(fmperf.data) / "ai.txt"

with open(seed_text_file, "r") as f:
    text = f.read()

parser = argparse.ArgumentParser()
parser.add_argument("--import-text", help="json file name of input texts")
parser.add_argument(
    "--from-model",
    help="generate requests according to requests model",
    action="store_true",
)
args = parser.parse_args()


def get_streaming_response(response: requests.Response, request_timeout: float, ttft_timeout: float, tpot_timeout: float):
    response_iter = response.iter_lines(
        chunk_size=8192,
        decode_unicode=False,
        delimiter=b"\n",
    )

    finished = False
    prev_completion_tokens = 0
    request_start_time = time.time_ns()
    first_token_received = False
    last_token_time = request_start_time

    while not finished:
        # request-level timeout
        current_time = time.time_ns()
        if (current_time - request_start_time) / 1e9 > request_timeout:
            raise TimeoutError(f"Request timeout: {request_timeout}s exceeded")

        chunk = next(response_iter)
        timestamp = time.time_ns()

        # time per output token (after first)
        if first_token_received and (timestamp - last_token_time) / 1e9 > tpot_timeout:
            raise TimeoutError(f"TPOT timeout: {tpot_timeout}s exceeded")

        if chunk and not finished:
            data = chunk.decode("utf-8").strip().split("data: ")[1]
            data_parsed = json.loads(data)
            out = data_parsed["choices"][0]
            finished = out["finish_reason"] is not None

            if ("usage" in data_parsed) and (data_parsed["usage"] is not None):
                usage = data_parsed["usage"]
                token_count = usage["completion_tokens"] - prev_completion_tokens
                prev_completion_tokens = usage["completion_tokens"]

                # first-token timeout (TTFT)
                if not first_token_received:
                    ttft = (timestamp - request_start_time) / 1e9
                    if ttft > ttft_timeout:
                        raise TimeoutError(
                            f"TTFT timeout: {ttft_timeout}s exceeded (TTFT: {ttft:.3f}s)"
                        )
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
                    }

                last_token_time = timestamp
            else:
                raise RuntimeError("No usage data in server response")


def get_text():
    if args.import_text:
        texts = json.load(open(args.import_text, "r"))
        n = len(texts)
        i = 0
        while True:
            t = texts[i]
            i = (i + 1) % n
            yield t
    else:
        while True:
            yield text


text_generator = get_text()


def generate_vllm_request(config, url):
    # Remove http:// prefix if present to avoid duplication
    url_no_prefix = url.replace("http://", "")

    model = requests.get("http://%s/v1/models" % (url_no_prefix)).json()["data"][0][
        "id"
    ]

    # Set Hugging Face token if available in environment
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    tokenizer_kwargs = {}
    if hf_token:
        tokenizer_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)

    source_text = next(text_generator)
    prompt_ids = tokenizer(source_text).input_ids[-config["in_tokens"] :]

    request = {
        "model": model,
        "prompt": prompt_ids,
        "ignore_eos": True,
        "min_tokens": config["out_tokens"],
        "max_tokens": config["out_tokens"],
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "stop": [],
        "ignore_eos": True,
    }

    if not args.from_model:
        request["temperature"] = 0.0 if config["is_greedy"] else 1.0
    else:
        if config["is_greedy"] or config["temperature"] == 0.0:
            request["temperature"] = 0.0
            request["top_k"] = -1
            request["top_p"] = 1
        else:
            request["temperature"] = config["temperature"]
            request["top_k"] = config["top_k"] if config["top_k"] > 0 else -1
            request["top_p"] = config["top_p"]

    headers = {"User-Agent": "Test Client"}

    # Timeouts configuration (can be overridden via env)
    request_timeout = float(os.environ.get("REQUEST_TIMEOUT", "300"))
    ttft_timeout = float(os.environ.get("TTFT_TIMEOUT", "60"))
    tpot_timeout = float(os.environ.get("TPOT_TIMEOUT", "30"))

    response = requests.post(
        "http://%s/v1/completions" % (url_no_prefix),
        headers=headers,
        json=request,
        stream=True,
        timeout=request_timeout,
    )

    if response.status_code != 200:
        raise RuntimeError(response.text)

    expected = []
    for r in get_streaming_response(response, request_timeout, ttft_timeout, tpot_timeout):
        expected.append(r)

    # let's check if we get one output per token (not the case for TGIS)
    assert len(expected) == config["out_tokens"]

    # Derive human-readable prompt text via tokenizer decode for logging
    try:
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    except Exception:
        # Fallback: if decode fails, store token IDs string
        prompt_text = f"<token_ids:{len(prompt_ids)}>"

    return request, expected, prompt_text, len(prompt_ids)


def generate_tgis_request(config, url):
    """
    Generate (streaming) gRPC request and expected response
    """

    from text_generation_tests.pb import (
        generation_pb2_grpc as gpb2,
        generation_pb2 as pb2,
    )

    channel = grpc.insecure_channel(url)
    stub = gpb2.GenerationServiceStub(channel)

    params = {
        "method": "GREEDY" if config["is_greedy"] else "SAMPLE",
        "stopping": {
            "minNewTokens": config["out_tokens"],
            "maxNewTokens": config["out_tokens"],
        },
        "sampling": {
            "seed": 42,
        },
        "truncate_input_tokens": config["in_tokens"],
    }

    request = {
        "model_id": "null",
        "params": params,
        "request": {
            "text": next(text_generator),
        },
    }

    if args.from_model:
        params["sampling"]["temperature"] = config["temperature"]
        params["sampling"]["top_k"] = config["top_k"]
        params["sampling"]["top_p"] = config["top_p"]

    message = json_format.ParseDict(request, pb2.SingleGenerationRequest())

    # Timeouts configuration (can be overridden via env)
    request_timeout = float(os.environ.get("REQUEST_TIMEOUT", "300"))
    ttft_timeout = float(os.environ.get("TTFT_TIMEOUT", "60"))
    tpot_timeout = float(os.environ.get("TPOT_TIMEOUT", "30"))

    response = []
    request_start_time = time.time_ns()
    first_token_received = False
    last_token_time = request_start_time

    for x in stub.GenerateStream(message):
        # request-level timeout
        current_time = time.time_ns()
        if (current_time - request_start_time) / 1e9 > request_timeout:
            raise TimeoutError(f"Request timeout: {request_timeout}s exceeded")

        tmp = json_format.MessageToDict(x)
        if "inputTokenCount" not in tmp:
            timestamp = time.time_ns()
            # TPOT timeout after first token
            if first_token_received and (timestamp - last_token_time) / 1e9 > tpot_timeout:
                raise TimeoutError(f"TPOT timeout: {tpot_timeout}s exceeded")

            # TTFT for first token
            if not first_token_received:
                ttft = (timestamp - request_start_time) / 1e9
                if ttft > ttft_timeout:
                    raise TimeoutError(
                        f"TTFT timeout: {ttft_timeout}s exceeded (TTFT: {ttft:.3f}s)"
                    )
                first_token_received = True

            response.append(tmp)
            last_token_time = timestamp

    # Prompt text is sent in the request; token count equals configured input budget
    prompt_text = request["request"]["text"]
    prompt_token_count = config["in_tokens"]
    return request, response, prompt_text, prompt_token_count


np.random.seed(42)

# Get sample size
sample_size = int(os.environ["SAMPLE_SIZE"])

if not args.from_model:
    # Get input size distribution info
    min_in_tokens = int(os.environ["MIN_INPUT_TOKENS"])
    max_in_tokens = int(os.environ["MAX_INPUT_TOKENS"])

    # Get output size distribution info
    min_out_tokens = int(os.environ["MIN_OUTPUT_TOKENS"])
    max_out_tokens = int(os.environ["MAX_OUTPUT_TOKENS"])

    # Get greedy
    frac_greedy = float(os.environ["FRAC_GREEDY"])

# output file
filename = os.environ["REQUESTS_FILENAME"]

# target
target = os.environ["TARGET"]

# url
url = os.environ["URL"]

# overwrite
overwrite = os.getenv("OVERWRITE", "false").lower() != "false"

if os.path.isfile(os.path.join(REQUESTS_DIR, filename)) and not overwrite:
    print("File %s already exists; skipping workload generation" % (filename))
    sys.exit()


print(">> ---------------------------------")
print(">> Generating heterogeneous requests")
print(">> ---------------------------------")
print(">> sample_size    = %d" % (sample_size))

if not args.from_model:
    print(">> min_in_tokens  = %d" % (min_in_tokens))
    print(">> max_in_tokens  = %d" % (max_in_tokens))
    print(">> min_out_tokens = %d" % (min_out_tokens))
    print(">> max_out_tokens = %d" % (max_out_tokens))
    print(">> frac_greedy    = %.2f" % (frac_greedy))

print(">> filename       = %s" % (filename))
print(">> target         = %s" % (target))
print(">> url            = %s" % (url))


cases = []

if args.from_model:
    requests_model_file = impresources.files(fmperf.data) / "all_nbins_64.pkl"
    print(">> loading requests model from file: ", requests_model_file)

    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "CustomHistogram":
                from .custom_histogram_model import CustomHistogram

                return CustomHistogram
            return super().find_class(module, name)

    requests_model = CustomUnpickler(open(requests_model_file, "rb")).load()

    samples = requests_model.sample(sample_size)
    print(samples)

attempts_per_sample = int(os.environ.get("MAX_GENERATE_ATTEMPTS", "3"))
retry_backoff = float(os.environ.get("RETRY_BACKOFF_SECONDS", "1.0"))

for sample_idx in range(sample_size):
    if args.from_model:
        sample = samples.iloc[sample_idx]
        config = {
            "in_tokens": sample["input_token_count"],
            "out_tokens": sample["generated_token_count"],
            "is_greedy": sample["is_greedy"],
            "temperature": sample["params.temperature"],
            "top_k": sample["params.top_k"],
            "top_p": sample["params.top_p"],
        }
    else:
        config = {
            "in_tokens": np.random.randint(low=min_in_tokens, high=max_in_tokens + 1),
            "out_tokens": np.random.randint(
                low=min_out_tokens, high=max_out_tokens + 1
            ),
            "is_greedy": np.random.uniform() < frac_greedy,
        }

    case = {
        "config": config,
    }

    success = False
    for attempt in range(1, attempts_per_sample + 1):
        try:
            if target == "tgis":
                req, expected, prompt_text, prompt_token_count = generate_tgis_request(config, url)
                case["request"], case["expected"] = req, expected
                case["prompt_text"], case["prompt_token_count"] = prompt_text, prompt_token_count
            elif target == "vllm":  # StackSpec will also use this
                req, expected, prompt_text, prompt_token_count = generate_vllm_request(config, url)
                case["request"], case["expected"] = req, expected
                case["prompt_text"], case["prompt_token_count"] = prompt_text, prompt_token_count
            else:
                raise ValueError(f"Invalid target: {target}")

            # verify expected token count to avoid downstream mismatches
            if len(case["expected"]) != config["out_tokens"]:
                raise RuntimeError(
                    f"Expected {config['out_tokens']} tokens, got {len(case['expected'])}"
                )

            print(json.dumps(case, indent=4))
            cases.append(case)
            success = True
            break
        except Exception:
            print(f"[sample {sample_idx}] attempt {attempt} failed:\n{traceback.format_exc()}")
            time.sleep(retry_backoff)

    if not success:
        print(f"[sample {sample_idx}] giving up after {attempts_per_sample} attempts; skipping sample")


if len(cases) > 0:
    print(">> Writing %d requests to %s" % (len(cases), filename))
    with open(os.path.join(REQUESTS_DIR, filename), "w") as f:
        json.dump(cases, f)
