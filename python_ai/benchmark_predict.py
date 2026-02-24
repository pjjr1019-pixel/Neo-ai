"""
FastAPI HTTP Endpoint Benchmark Script

This script benchmarks the /predict endpoint using httpx and asyncio.
Run this script while your FastAPI server is running (default: http://127.0.0.1:8000).
"""
import asyncio
import httpx
import time

URL = "http://127.0.0.1:8000/predict"
PAYLOAD = {"price": 123.45, "volume": 1000}
CONCURRENCY = 20
REQUESTS = 200

async def worker(client, n):
    for _ in range(n):
        response = await client.post(URL, json=PAYLOAD)
        assert response.status_code == 200

async def main():
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=10) as client:
        tasks = [worker(client, REQUESTS // CONCURRENCY) for _ in range(CONCURRENCY)]
        await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start
    print(f"Sent {REQUESTS} requests in {elapsed:.2f} seconds.")
    print(f"Throughput: {REQUESTS/elapsed:.2f} req/sec")

if __name__ == "__main__":
    asyncio.run(main())
