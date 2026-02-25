# API Usage Examples

## /predict Endpoint

**Request:**
```
POST /predict
Content-Type: application/json
{
  "price": 123.45,
  "volume": 1000
}
```

**Response:**
```
{
  "action": "buy",
  "confidence": 0.95,
  "risk": 0.1
}
```

## /learn Endpoint

**Request:**
```
POST /learn
Content-Type: application/json
{
  "features": [1, 2, 3],
  "target": 1.0
}
```

**Response:**
```
{
  "status": "learning triggered",
  "received": {"features": [1, 2, 3], "target": 1.0}
}
```

## /metrics Endpoint

**Request:**
```
GET /metrics
```

**Response:**
```
{
  "avg_latency": 0.002,
  "throughput": 100.0,
  "memory_mb": 50.0,
  "cpu_percent": 2.5
}
```
