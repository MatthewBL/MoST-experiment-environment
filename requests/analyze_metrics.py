import os
import sys
import json
import csv
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
from statistics import median as stats_median

# ------------------------------------------------------------
# Metrics computed per folder:
# - Number of tokens generated per request (by response_idx)
# - Maximum number of requests started per minute
#   (start = first token timestamp for a response_idx)
# - Maximum number of requests responded per minute
#   (responded = last token timestamp for a response_idx)
#
# Assumptions about schema (adjust mapping if needed):
# - results.json: either
#   a) a top-level object with key "results" that is a list of token events, each having
#      at least "response_idx" and a timestamp field like "timestamp" or "created_at"; or
#   b) a list of response objects with "response_idx" and a nested list of tokens under
#      keys like "tokens" or "chunks" where each token has a timestamp.
# - results.csv (fallback): rows contain at least "response_idx" and a timestamp column
#   such as "timestamp" or "created_at". Rows representing individual streamed tokens
#   will be counted toward token totals.
#
# A "started" request is the minute of its first token. A "responded" request is the
# minute of its last token.
# ------------------------------------------------------------

TimestampKeys = ("timestamp", "created_at", "time", "ts")
TokenListKeys = ("tokens", "chunks", "deltas")


def parse_timestamp(ts_val) -> Optional[datetime]:
    """Parse a timestamp value into a timezone-aware datetime (UTC).
    Robustly supports ISO 8601 strings and epoch seconds/milliseconds.
    Avoids platform OSError by trying plausible interpretations.
    """
    if ts_val is None:
        return None

    # Numeric epoch (seconds or milliseconds)
    if isinstance(ts_val, (int, float)):
        # Try as seconds, milliseconds, microseconds, nanoseconds
        candidates: List[float] = [float(ts_val)]
        # Milliseconds
        candidates.append(float(ts_val) / 1000.0)
        # Microseconds
        candidates.append(float(ts_val) / 1_000_000.0)
        # Nanoseconds
        candidates.append(float(ts_val) / 1_000_000_000.0)
        for v in candidates:
            try:
                return datetime.fromtimestamp(v, tz=timezone.utc)
            except (OSError, OverflowError, ValueError):
                continue
        return None

    if isinstance(ts_val, str):
        s = ts_val.strip()
        # Normalize trailing 'Z'
        if s.endswith("Z"):
            s = s[:-1]
        # Try ISO 8601
        try:
            dt = datetime.fromisoformat(s)
            # If naive, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
        # Try common datetime string formats
        dt_patterns = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y%m%dT%H%M%S",
            "%Y%m%dT%H%M%S.%f",
            "%Y%m%d%H%M%S",
            "%Y%m%d%H%M%S.%f",
        ]
        for pat in dt_patterns:
            try:
                dt = datetime.strptime(s, pat).replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                continue
        # Try numeric epoch encoded as string (seconds or ms)
        try:
            num = float(s)
            return parse_timestamp(num)
        except Exception:
            return None

    return None


def minute_bucket(dt: datetime) -> datetime:
    """Round down to the minute bucket in UTC."""
    return dt.replace(second=0, microsecond=0, tzinfo=timezone.utc)


def extract_events_from_json_item(item) -> List[Dict[str, Optional[datetime]]]:
    """Extract token events from a JSON item.
    Returns a list of dicts with keys: 'request_idx', 'response_idx', 'timestamp'.
    Attempts multiple shapes: flat token event with timestamp, or nested token lists.
    """
    events: List[Dict[str, Optional[datetime]]] = []

    response_idx = item.get("response_idx")
    request_idx = item.get("request_idx")

    # Case 1: flat event contains a timestamp directly
    for key in TimestampKeys:
        if key in item:
            ts = parse_timestamp(item.get(key))
            if ts is not None and response_idx is not None:
                events.append({
                    "request_idx": request_idx if request_idx is not None else None,
                    "response_idx": int(response_idx),
                    "timestamp": ts,
                })
                return events

    # Case 2: nested token lists (tokens/chunks/deltas)
    for list_key in TokenListKeys:
        if list_key in item and isinstance(item[list_key], list):
            for token in item[list_key]:
                ts = None
                if isinstance(token, dict):
                    for tk in TimestampKeys:
                        if tk in token:
                            ts = parse_timestamp(token.get(tk))
                            if ts is not None:
                                break
                if ts is not None and response_idx is not None:
                    events.append({
                        "request_idx": request_idx if request_idx is not None else None,
                        "response_idx": int(response_idx),
                        "timestamp": ts,
                    })
            return events

    return events


def load_events_from_results_json(fp: str) -> List[Dict[str, Optional[datetime]]]:
    events: List[Dict[str, Optional[datetime]]] = []
    with open(fp, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If top-level contains 'results' list
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        for item in data["results"]:
            if isinstance(item, dict):
                events.extend(extract_events_from_json_item(item))
        return events

    # If data is a list of items
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                events.extend(extract_events_from_json_item(item))
        return events

    # Otherwise, attempt any nested reasonable structures
    return events


def load_events_from_results_csv(fp: str) -> List[Dict[str, Optional[datetime]]]:
    events: List[Dict[str, Optional[datetime]]] = []
    with open(fp, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            response_idx = row.get("response_idx")
            request_idx = row.get("request_idx")
            if response_idx is None and request_idx is None:
                # If neither identifier is present, skip
                continue
            ts_val = None
            for key in TimestampKeys:
                if key in row and row[key]:
                    ts_val = row[key]
                    break
            ts = parse_timestamp(ts_val)
            if ts is not None:
                events.append({
                    "request_idx": int(request_idx) if request_idx is not None and request_idx != "" else None,
                    "response_idx": int(response_idx) if response_idx is not None and response_idx != "" else None,
                    "timestamp": ts,
                })
    return events


def load_folder_events(folder: str) -> List[Dict[str, Optional[datetime]]]:
    """Load token events from a folder, preferring results.json then results.csv."""
    # Prefer results.json
    json_path = os.path.join(folder, "results.json")
    if os.path.exists(json_path):
        ev = load_events_from_results_json(json_path)
        if ev:
            return ev
    # Try output.csv
    output_csv = os.path.join(folder, "output.csv")
    if os.path.exists(output_csv):
        ev = load_events_from_results_csv(output_csv)
        if ev:
            return ev
    # Try results.csv
    results_csv = os.path.join(folder, "results.csv")
    if os.path.exists(results_csv):
        ev = load_events_from_results_csv(results_csv)
        if ev:
            return ev
    # Try first_half.csv and second_half.csv if present
    first_half_csv = os.path.join(folder, "first_half.csv")
    if os.path.exists(first_half_csv):
        ev = load_events_from_results_csv(first_half_csv)
        if ev:
            return ev
    second_half_csv = os.path.join(folder, "second_half.csv")
    if os.path.exists(second_half_csv):
        ev = load_events_from_results_csv(second_half_csv)
        if ev:
            return ev
    return []


def compute_metrics(events: List[Dict[str, Optional[datetime]]]):
    """Compute metrics from token events.
    Events must have 'timestamp' and either 'request_idx' or 'response_idx'.
    Requests are grouped by 'request_idx' when available; otherwise, derived
    sequentially by treating any event with response_idx==0 as a new request.
    """
    # Sort by timestamp to ensure sequential grouping when deriving requests
    events_sorted = [e for e in events if e.get("timestamp") is not None]
    events_sorted.sort(key=lambda e: e["timestamp"])  # type: ignore

    # Determine request IDs
    use_request_idx = any(e.get("request_idx") is not None for e in events_sorted)
    request_id_of_event: List[int] = []
    # Track max response_idx per request for accurate token counts
    max_resp_idx: Dict[int, int] = {}
    first_ts: Dict[int, datetime] = {}
    last_ts: Dict[int, datetime] = {}

    if use_request_idx:
        for e in events_sorted:
            rid = int(e["request_idx"])  # type: ignore
            ts = e["timestamp"]  # type: ignore
            request_id_of_event.append(rid)
            # Update max response_idx if available
            rsi = e.get("response_idx")
            if rsi is not None:
                rsi_int = int(rsi)
                max_resp_idx[rid] = max(rsi_int, max_resp_idx.get(rid, -1))
            if rid not in first_ts or ts < first_ts[rid]:
                first_ts[rid] = ts
            if rid not in last_ts or ts > last_ts[rid]:
                last_ts[rid] = ts
    else:
        # Derive sequential request IDs from response_idx resets
        current_rid = -1
        for e in events_sorted:
            ri = e.get("response_idx")
            ts = e["timestamp"]  # type: ignore
            if ri is not None and int(ri) == 0:
                current_rid += 1
            if current_rid < 0:
                # If stream doesn't start at 0, start now
                current_rid = 0
            request_id_of_event.append(current_rid)
            rsi = e.get("response_idx")
            if rsi is not None:
                rsi_int = int(rsi)
                max_resp_idx[current_rid] = max(rsi_int, max_resp_idx.get(current_rid, -1))
            if current_rid not in first_ts or ts < first_ts[current_rid]:
                first_ts[current_rid] = ts
            if current_rid not in last_ts or ts > last_ts[current_rid]:
                last_ts[current_rid] = ts

    # Compute responded requests per minute (based on last token time of each request)
    responded_per_minute: Dict[datetime, int] = {}
    for rid, end_ts in last_ts.items():
        mb = minute_bucket(end_ts)
        responded_per_minute[mb] = responded_per_minute.get(mb, 0) + 1

    counts = list(responded_per_minute.values())
    if counts:
        median_responded_requests_per_minute = float(stats_median(counts))
    else:
        median_responded_requests_per_minute = 0.0

    if counts:
        average_responded_requests_per_minute = sum(counts) / len(counts)
        min_responded_requests_per_minute = min(counts)
        max_responded_requests_per_minute = max(counts)
    else:
        average_responded_requests_per_minute = 0.0
        min_responded_requests_per_minute = 0
        max_responded_requests_per_minute = 0

    return {
        "median_responded_requests_per_minute": median_responded_requests_per_minute,
        "average_responded_requests_per_minute": average_responded_requests_per_minute,
        "min_responded_requests_per_minute": min_responded_requests_per_minute,
        "max_responded_requests_per_minute": max_responded_requests_per_minute,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_metrics.py <folder_path>")
        print("Example: python analyze_metrics.py \"C:\\Users\\Hola Isa\\Desktop\\New folder (4)\\sent_data\\2025-12-04_10-56-09\"")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Error: Not a directory: {folder}")
        sys.exit(2)

    events = load_folder_events(folder)
    if not events:
        print("No events found. Ensure results.json or results.csv contains token-level entries with 'response_idx' and timestamps.")
        sys.exit(3)

    metrics = compute_metrics(events)

    print("Folder:", folder)
    print("Median responded requests per minute:", f"{metrics['median_responded_requests_per_minute']:.3f}")
    print("Average responded requests per minute:", f"{metrics['average_responded_requests_per_minute']:.3f}")
    print("Minimum responded requests per minute:", metrics['min_responded_requests_per_minute'])
    print("Maximum responded requests per minute:", metrics['max_responded_requests_per_minute'])


if __name__ == "__main__":
    main()
