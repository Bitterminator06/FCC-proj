"""
Cloud Computing Load Testing & Elasticity Benchmarking Tool.
Simulates concurrent edge clients sending hand landmark payloads to the Cloud Inference API.
Demonstrates:
  - Throughput (Requests Per Second - RPS)
  - Latency percentiles (P50, P90, P99)
  - Edge vs Cloud Bandwidth Savings
  - Horizontal Scalability and Cloud Sizing recommendations for academic reports.
"""

import time
import requests
import concurrent.futures
import numpy as np

API_URL = "http://127.0.0.1:8000/predict"
METRICS_URL = "http://127.0.0.1:8000/metrics"

def send_single_inference(dummy_payload):
    t_start = time.perf_counter()
    try:
        resp = requests.post(API_URL, json={"landmarks": dummy_payload}, timeout=2.0)
        dur_ms = (time.perf_counter() - t_start) * 1000
        return resp.status_code == 200, dur_ms
    except Exception:
        return False, 0.0

def run_cloud_benchmark(total_requests=200, concurrency=10):
    print("=" * 65)
    print("CLOUD COMPUTING ELASTICITY & CONCURRENCY BENCHMARK")
    print(f"Target Endpoint: {API_URL}")
    print(f"Total Requests: {total_requests} | Concurrent Virtual Clients: {concurrency}")
    print("=" * 65)

    dummy_landmarks = [0.05] * 42
    latencies = []
    successes = 0

    benchmark_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_single_inference, dummy_landmarks) for _ in range(total_requests)]
        for f in concurrent.futures.as_completed(futures):
            ok, lat = f.result()
            if ok:
                successes += 1
                latencies.append(lat)

    total_time = time.perf_counter() - benchmark_start
    rps = total_requests / total_time

    # Latency percentiles
    p50 = np.percentile(latencies, 50) if latencies else 0
    p90 = np.percentile(latencies, 90) if latencies else 0
    p99 = np.percentile(latencies, 99) if latencies else 0

    print("\nBENCHMARK RESULTS (Cloud Metrics):")
    print(f"  * Total Time Elapsed       : {total_time:.2f} seconds")
    print(f"  * Throughput (RPS)         : {rps:.1f} req/sec")
    print(f"  * Success Rate             : {(successes/total_requests)*100:.1f}% ({successes}/{total_requests})")
    print(f"  * P50 Latency (Median)     : {p50:.2f} ms")
    print(f"  * P90 Latency              : {p90:.2f} ms")
    print(f"  * P99 Latency              : {p99:.2f} ms")

    # Fetch server side observability telemetry
    try:
        telemetry = requests.get(METRICS_URL).json()
        bw = telemetry.get("bandwidth_analysis", {})
        print("\nEDGE-CLOUD BANDWIDTH SAVINGS TELEMETRY:")
        print(f"  * Actual Landmark Data Ingress : {bw.get('actual_landmarks_received_kb')} KB")
        print(f"  * Equivalent Raw Video Ingress : {bw.get('estimated_raw_video_mb')} MB")
        print(f"  * Network Bandwidth Saved      : {bw.get('bandwidth_reduction_pct')}")
    except Exception:
        pass

    print("=" * 65)

if __name__ == "__main__":
    run_cloud_benchmark(total_requests=250, concurrency=12)
