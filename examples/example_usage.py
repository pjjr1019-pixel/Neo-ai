# Example: Using NEO FastAPI endpoints

import requests

# Predict endpoint
response = requests.post(
    'http://localhost:8000/predict',
    json={'price': 150, 'volume': 10}
)
print('Predict:', response.json())

# Learn endpoint
response = requests.post(
    'http://localhost:8000/learn',
    json={'features': [1, 2, 3], 'target': 1.0}
)
print('Learn:', response.json())
