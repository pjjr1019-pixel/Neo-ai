# API Usage

## Authentication
- Endpoints are protected by JWT/API key middleware.
- Use `Authorization: Bearer <token>` or configured API-key flow.

## `POST /compute-features`
Request:
```json
{
  "symbol": "BTC/USD",
  "ohlcv_data": {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}
}
```

## `POST /predict`
Request:
```json
{"features": {"sma_14": 100.0, "rsi_14": 50.0}}
```
Response:
```json
{"prediction": 0.12, "confidence": 0.82, "signal": "BUY"}
```

## `POST /learn`
Request:
```json
{"features": [0.1, 0.2, 0.3], "target": 0.01}
```

## `GET /explain`
- Returns global feature importances and explanation method.
