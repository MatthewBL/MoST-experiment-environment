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
from typing import Iterable, List, Optional
from importlib import resources as impresources
import fmperf.data
import traceback
from transformers import AutoTokenizer
from pathlib import Path

from fmperf.utils.constants import REQUESTS_DIR, REQUESTS_FILENAME
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPTS_FILE = REPO_ROOT / "oasst_roots_en_max1000_tokens.jsonl"
TOKEN_FIELD = "gpt2_token_count"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--from-model",
    help="generate requests according to requests model",
    action="store_true",
)
parser.add_argument("--prompts-file", type=Path, help="Path to JSONL prompts file (overrides PROMPTS_FILE)")
parser.add_argument("--min-input", dest="min_input", type=int, help="Minimum allowed prompt tokens", default=None)
parser.add_argument("--max-input", dest="max_input", type=int, help="Maximum allowed prompt tokens", default=None)
parser.add_argument(
    "--min-output",
    dest="min_output",
    type=int,
    help="Minimum generated tokens",
    default=None,
)
parser.add_argument(
    "--max-output",
    dest="max_output",
    type=int,
    help="Maximum generated tokens",
    default=None,
)
parser.add_argument(
    "--frac-greedy",
    dest="frac_greedy",
    type=float,
    help="Override FRAC_GREEDY probability",
    default=None,
)
parser.add_argument(
    "--output",
    dest="output",
    type=Path,
    help="Destination file for generated workload",
    default=None,
)
args = parser.parse_args()

PROMPTS_FILE = args.prompts_file or Path(os.environ.get("PROMPTS_FILE", DEFAULT_PROMPTS_FILE))
if not PROMPTS_FILE.exists():
    raise FileNotFoundError(f"Prompts file not found: {PROMPTS_FILE}")


def _resolve_int(value: Optional[int], env_key: str, fallback: Optional[int] = None) -> int:
    if value is not None:
        return value
    env_value = os.environ.get(env_key)
    if env_value is not None:
        return int(env_value)
    if fallback is not None:
        return fallback
    raise ValueError(f"Missing required integer for {env_key}")


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
        current_time = time.time_ns()
        if (current_time - request_start_time) / 1e9 > request_timeout:
            raise TimeoutError(f"Request timeout: {request_timeout}s exceeded")

        chunk = next(response_iter)
        timestamp = time.time_ns()

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


def _extract_text_and_tokens(obj: dict):
    text_keys = ["text", "prompt", "content", "message", "input"]
    token_keys = [TOKEN_FIELD, "n_tokens", "tokens", "input_token_count", "token_count"]
    prompt_text = None
    for k in text_keys:
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            prompt_text = obj[k].strip()
            break
    prompt_tokens = None
    for k in token_keys:
        if k in obj:
            try:
                prompt_tokens = int(obj[k])
                break
            except Exception:
                continue
    return prompt_text, prompt_tokens


def _load_prompts_from_jsonl(path: Path):
    prompts = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                txt, toks = _extract_text_and_tokens(obj)
                if txt is None:
                    continue
                prompts.append({"text": txt, "tokens": toks})
    except FileNotFoundError:
        print(f"Warning: prompts file not found: {path}")
    except Exception as e:
        print(f"Warning: failed to load prompts from {path}: {e}")
    return prompts


def generate_vllm_request(config, url, source_text):
    url_no_prefix = url.replace("http://", "")

    model = requests.get("http://%s/v1/models" % (url_no_prefix)).json()["data"][0]["id"]

    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    tokenizer_kwargs = {}
    if hf_token:
        tokenizer_kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)
    prompt_ids = tokenizer(source_text).input_ids

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

    assert len(expected) == config["out_tokens"]

    try:
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
    except Exception:
        prompt_text = source_text

    return request, expected, prompt_text, len(prompt_ids)


def generate_tgis_request(config, url, source_text):
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
            "text": source_text,
        },
    }

    if args.from_model:
        params["sampling"]["temperature"] = config["temperature"]
        params["sampling"]["top_k"] = config["top_k"]
        params["sampling"]["top_p"] = config["top_p"]

    message = json_format.ParseDict(request, pb2.SingleGenerationRequest())

    request_timeout = float(os.environ.get("REQUEST_TIMEOUT", "300"))
    ttft_timeout = float(os.environ.get("TTFT_TIMEOUT", "60"))
    tpot_timeout = float(os.environ.get("TPOT_TIMEOUT", "30"))

    response = []
    request_start_time = time.time_ns()
    first_token_received = False
    last_token_time = request_start_time

    for x in stub.GenerateStream(message):
        current_time = time.time_ns()
        if (current_time - request_start_time) / 1e9 > request_timeout:
            raise TimeoutError(f"Request timeout: {request_timeout}s exceeded")

        tmp = json_format.MessageToDict(x)
        if "inputTokenCount" not in tmp:
            timestamp = time.time_ns()
            if first_token_received and (timestamp - last_token_time) / 1e9 > tpot_timeout:
                raise TimeoutError(f"TPOT timeout: {tpot_timeout}s exceeded")

            if not first_token_received:
                ttft = (timestamp - request_start_time) / 1e9
                if ttft > ttft_timeout:
                    raise TimeoutError(
                        f"TTFT timeout: {ttft_timeout}s exceeded (TTFT: {ttft:.3f}s)"
                    )
                first_token_received = True

            response.append(tmp)
            last_token_time = timestamp

    prompt_text = request["request"]["text"]
    prompt_token_count = config["in_tokens"]
    return request, response, prompt_text, prompt_token_count


np.random.seed(42)

min_in_tokens = _resolve_int(args.min_input, "MIN_INPUT_TOKENS")
max_in_tokens = _resolve_int(args.max_input, "MAX_INPUT_TOKENS")
if min_in_tokens <= 0 or max_in_tokens <= 0:
    raise ValueError("Input token bounds must be positive")
if min_in_tokens > max_in_tokens:
    raise ValueError("MIN_INPUT_TOKENS cannot exceed MAX_INPUT_TOKENS")

min_out_tokens = _resolve_int(args.min_output, "MIN_OUTPUT_TOKENS", fallback=min_in_tokens)
max_out_tokens = _resolve_int(args.max_output, "MAX_OUTPUT_TOKENS", fallback=max_in_tokens)
if min_out_tokens <= 0 or max_out_tokens <= 0:
    raise ValueError("Output token bounds must be positive")
if min_out_tokens > max_out_tokens:
    raise ValueError("MIN_OUTPUT_TOKENS cannot exceed MAX_OUTPUT_TOKENS")

frac_greedy = args.frac_greedy if args.frac_greedy is not None else float(os.environ.get("FRAC_GREEDY", "1.0"))

base_filename = os.environ.get("REQUESTS_FILENAME", REQUESTS_FILENAME)
name_root = Path(base_filename).stem
suffix = Path(base_filename).suffix or ".json"
filename = f"{name_root}_{min_in_tokens}-{max_in_tokens}{suffix}"
output_path = args.output if args.output is not None else Path(REQUESTS_DIR) / filename
output_path.parent.mkdir(parents=True, exist_ok=True)

target = os.environ["TARGET"]
url = os.environ["URL"]

overwrite = os.getenv("OVERWRITE", "false").lower() != "false"
if output_path.exists() and not overwrite:
    print(f"File {output_path} already exists; skipping workload generation")
    sys.exit()

print(">> ---------------------------------")
print(">> Generating requests from JSONL prompts")
print(">> ---------------------------------")
if args.from_model:
    sample_size = int(os.environ.get("SAMPLE_SIZE", "0"))
    print(">> sample_size    = %d" % (sample_size))

if not args.from_model:
    print(">> min_in_tokens  = %d" % (min_in_tokens))
    print(">> max_in_tokens  = %d" % (max_in_tokens))
    print(">> min_out_tokens = %d" % (min_out_tokens))
    print(">> max_out_tokens = %d" % (max_out_tokens))
    print(">> frac_greedy    = %.2f" % (frac_greedy))

print(">> filename       = %s" % (output_path.name))
print(">> target         = %s" % (target))
print(">> url            = %s" % (url))
print(">> prompts_file   = %s" % (PROMPTS_FILE))

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
    samples = requests_model.sample(sample_size) if sample_size > 0 else requests_model.sample(0)
    print(samples)

attempts_per_sample = int(os.environ.get("MAX_GENERATE_ATTEMPTS", "3"))
retry_backoff = float(os.environ.get("RETRY_BACKOFF_SECONDS", "1.0"))

prompts_pool = _load_prompts_from_jsonl(PROMPTS_FILE)
if not prompts_pool and not args.from_model:
    print("Warning: no prompts available in prompts file; aborting generation")
    sys.exit(1)

if args.from_model:
    for sample_idx in range(sample_size):
        sample = samples.iloc[sample_idx]
        config = {
            "in_tokens": sample["input_token_count"],
            "out_tokens": sample["generated_token_count"],
            "is_greedy": sample["is_greedy"],
            "temperature": sample["params.temperature"],
            "top_k": sample["params.top_k"],
            "top_p": sample["params.top_p"],
        }
        case = {
            "config": config,
        }

        success = False
        source_text = ""
        for attempt in range(1, attempts_per_sample + 1):
            try:
                if target == "tgis":
                    req, expected, prompt_text, prompt_token_count = generate_tgis_request(config, url, source_text)
                    case["request"], case["expected"] = req, expected
                    case["prompt_text"], case["prompt_token_count"] = prompt_text, prompt_token_count
                elif target == "vllm":
                    req, expected, prompt_text, prompt_token_count = generate_vllm_request(config, url, source_text)
                    case["request"], case["expected"] = req, expected
                    case["prompt_text"], case["prompt_token_count"] = prompt_text, prompt_token_count
                else:
                    raise ValueError(f"Invalid target: {target}")

                case["config"]["in_tokens"] = prompt_token_count

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
else:
    for sample_idx, p in enumerate(prompts_pool):
        source_text = p["text"]
        in_tok = p.get("tokens")
        if in_tok is None:
            raise ValueError(
                "Prompt is missing a token count; ensure generate_requests.py filters only prompts with gpt2_token_count"
            )
        in_tok = int(in_tok)

        config = {
            "in_tokens": in_tok,
            "out_tokens": int(np.random.randint(low=min_out_tokens, high=max_out_tokens + 1)),
            "is_greedy": np.random.uniform() < frac_greedy,
        }

        case = {
            "config": config,
        }

        success = False
        for attempt in range(1, attempts_per_sample + 1):
            try:
                if target == "tgis":
                    req, expected, prompt_text, prompt_token_count = generate_tgis_request(config, url, source_text)
                    case["request"], case["expected"] = req, expected
                    case["prompt_text"], case["prompt_token_count"] = prompt_text, prompt_token_count
                elif target == "vllm":
                    req, expected, prompt_text, prompt_token_count = generate_vllm_request(config, url, source_text)
                    case["request"], case["expected"] = req, expected
                    case["prompt_text"], case["prompt_token_count"] = prompt_text, prompt_token_count
                else:
                    raise ValueError(f"Invalid target: {target}")

                case["config"]["in_tokens"] = prompt_token_count

                if len(case["expected"]) != config["out_tokens"]:
                    raise RuntimeError(
                        f"Expected {config['out_tokens']} tokens, got {len(case['expected'])}"
                    )

                print(json.dumps(case, indent=4))
                cases.append(case)
                success = True
                break
            except Exception:
                print(f"[prompt {sample_idx}] attempt {attempt} failed:\n{traceback.format_exc()}")
                time.sleep(retry_backoff)

        if not success:
            print(f"[prompt {sample_idx}] giving up after {attempts_per_sample} attempts; skipping sample")


if len(cases) > 0:
    print(">> Writing %d requests to %s" % (len(cases), output_path))
    with output_path.open("w") as f:
        json.dump(cases, f)
else:
    print("Warning: no cases generated; nothing written")
