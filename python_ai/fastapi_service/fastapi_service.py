# Metrics tracker for middleware and /metrics endpoint
class Metrics:
	latency = []
	request_count = 0
	throughput = 0.0
	start_time = None

def reset_metrics():
	Metrics.latency = []
	Metrics.request_count = 0
	Metrics.throughput = 0.0
	Metrics.start_time = time.perf_counter()



"""
FastAPI service for NEO Hybrid AI system.
Includes endpoints for learning, prediction, and resource monitoring.
Flake8 and best-practice compliant.
"""

import logging
import time
import psutil
from fastapi import FastAPI, Depends
from typing import Callable, List
from pydantic import BaseModel


app = FastAPI()


class PredictInput(BaseModel):
	"""Input model for /predict endpoint."""
	price: float
	volume: int



class PredictionOutput(BaseModel):
	"""Output model for /predict endpoint."""
	action: str
	confidence: float
	risk: float

def predict_logic(data: PredictInput) -> PredictionOutput:
	"""Dummy prediction logic for demonstration."""
	return PredictionOutput(
		action="buy",
		confidence=0.95,
		risk=0.05
	)

@app.post("/predict")
def predict(data: PredictInput) -> dict:
	"""
	Predict endpoint for NEO Hybrid AI system.
	Returns action, confidence, and risk.
	Always returns a serializable dict, even if logic fails.
	"""
	try:
		result = predict_logic(data)
		return result.model_dump()
	except Exception as e:
		import traceback
		logging.error(f"Error in /predict: {e}\n{traceback.format_exc()}")
		return {
			"action": "error",
			"confidence": 0.0,
			"risk": 1.0,
			"detail": str(e)
		}


class LearnInput(BaseModel):
	"""Input model for /learn endpoint."""
	features: List[float]
	target: float


def learning_logic(data: LearnInput) -> dict:
	"""Dummy learning logic for demonstration."""
	return {"status": "learning triggered", "received": data.model_dump()}


def get_learning_logic() -> Callable[[LearnInput], dict]:
	"""Dependency injector for learning logic."""
	return learning_logic







def resource_usage() -> dict:
	"""Returns current process memory and CPU usage."""
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
	if Metrics.start_time is None:
		Metrics.start_time = time.perf_counter()
	start = time.perf_counter()
	response = await call_next(request)
	elapsed = time.perf_counter() - start
	Metrics.latency.append(elapsed)
	Metrics.request_count += 1
	Metrics.throughput = Metrics.request_count / (
		max(time.perf_counter() - Metrics.start_time, 1e-6)
	)
	response.headers["X-Latency"] = str(elapsed)
	response.headers["X-Throughput"] = str(Metrics.throughput)
	usage = resource_usage()
	response.headers["X-Memory-MB"] = str(usage['memory_mb'])
	response.headers["X-CPU-Percent"] = str(usage['cpu_percent'])
	return response

@app.get("/metrics")
def get_metrics():
	"""Returns current metrics and resource usage."""
	usage = resource_usage()
	return {
		"avg_latency": sum(Metrics.latency) / len(Metrics.latency)
		if Metrics.latency else 0,
		"throughput": Metrics.throughput,
		"memory_mb": usage["memory_mb"],
		"cpu_percent": usage["cpu_percent"]
	}

class PredictInput(BaseModel):
	"""Input model for /predict endpoint."""
	price: float
	volume: int

class PredictionOutput(BaseModel):
	"""Output model for /predict endpoint."""
	action: str
	confidence: float
	risk: float

def predict_logic(data: PredictInput) -> PredictionOutput:
	"""Dummy prediction logic for demonstration."""
	return PredictionOutput(
		action="buy",
		confidence=0.95,
		risk=0.05
	)

@app.post("/predict")
def predict(data: PredictInput) -> dict:
	"""
	Predict endpoint for NEO Hybrid AI system.
	Returns action, confidence, and risk.
	Always returns a serializable dict, even if logic fails.
	"""
	try:
		result = predict_logic(data)
		return result.model_dump()
	except Exception as e:
		logging.error(f"Error in /predict: {e}")
		return {
			"action": "error",
			"confidence": 0.0,
			"risk": 1.0
		}
