from fastapi import FastAPI, Depends
from typing import Callable
from pydantic import BaseModel
from typing import Optional
import logging

app = FastAPI()
logging.basicConfig(
    filename='fastapi_service.log',
    level=logging.INFO
)


class FeatureInput(BaseModel):
    price: float
    volume: float
    # Add more features as needed


class LearnInput(BaseModel):
    features: list
    target: float


class PredictionOutput(BaseModel):
    action: str
    confidence: float
    risk: Optional[float]


@app.post("/predict", response_model=PredictionOutput)
def predict(features: FeatureInput):
    """Predict trading action based on input features."""
    try:
        logging.info(f"Received /predict request: {features}")
        action = "buy" if features.price > 100 else "hold"
        confidence = 0.95
        risk = 0.1
        result = PredictionOutput(
            action=action,
            confidence=confidence,
            risk=risk
        )
        return result
    except Exception as e:
        logging.error(f"Error in /predict: {e}")
        return PredictionOutput(action="error", confidence=0.0, risk=None)


def learning_logic(data: LearnInput) -> dict:
    """
    Placeholder for learning logic.
    Returns a dict with status and received data.
    """
    # Use model_dump for Pydantic v2+ compliance
    return {"status": "learning triggered", "received": data.model_dump()}


def get_learning_logic() -> Callable[[LearnInput], dict]:
    return learning_logic

# --- Metrics and resource monitoring ---
import time
import psutil
class Metrics:
    latency = []
    throughput = 0
    start_time = time.perf_counter()
    request_count = 0

def resource_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / 1024 ** 2
    cpu = process.cpu_percent(interval=0.1)
    return {'memory_mb': mem, 'cpu_percent': cpu}


@app.post("/learn")
def learn(
    data: LearnInput,
    logic: Callable[[LearnInput], dict] = Depends(get_learning_logic)
) -> dict:
    """
    Trigger learning process with provided data.
    Returns a dict with status and received data or error.
    Uses dependency injection for learning_logic for testability.
    Always returns a serializable dict, even if dependency is broken.
    """
    try:
        logging.info(f"Received /learn request: {data}")
        if not callable(logic):
            raise TypeError("Injected learning logic is not callable")
        result = logic(data)
        if callable(result):
            raise TypeError(
                "Learning logic returned a function instead of a result"
            )
        logging.info(f"Learning result: {result}")
        if not isinstance(result, dict):
            raise TypeError("Learning logic did not return a dict")
        return result
    except Exception as e:
        logging.error(f"Error in /learn: {e}")
        return {
            "status": "error",
            "received": (
                data.model_dump() if hasattr(data, 'model_dump')
                else str(data)
            )
        }


@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    Metrics.latency.append(elapsed)
    Metrics.request_count += 1
    Metrics.throughput = Metrics.request_count / (time.perf_counter() - Metrics.start_time)
    response.headers["X-Latency"] = str(elapsed)
    response.headers["X-Throughput"] = str(Metrics.throughput)
    usage = resource_usage()
    response.headers["X-Memory-MB"] = str(usage['memory_mb'])
    response.headers["X-CPU-Percent"] = str(usage['cpu_percent'])
    return response

@app.get("/metrics")
def get_metrics():
    usage = resource_usage()
    return {
        "avg_latency": sum(Metrics.latency) / len(Metrics.latency) if Metrics.latency else 0,
        "throughput": Metrics.throughput,
        "memory_mb": usage['memory_mb'],
        "cpu_percent": usage['cpu_percent']
    }
