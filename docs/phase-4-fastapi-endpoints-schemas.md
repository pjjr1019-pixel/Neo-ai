# NEO Hybrid AI — FastAPI Endpoints & Schemas

## Endpoints
- /predict: Accepts feature input, returns action, confidence, risk
- /learn: Accepts training data, triggers learning

## Input Schema (FeatureInput)
- price: float
- volume: float
- ... (extend as needed)

## Output Schema (PredictionOutput)
- action: str
- confidence: float
- risk: float (optional)

## Example Request
POST /predict
{
  "price": 120.0,
  "volume": 210.0
}

## Example Response
{
  "action": "buy",
  "confidence": 0.85,
  "risk": 0.1
}

---
Update this file as endpoints and schemas evolve.