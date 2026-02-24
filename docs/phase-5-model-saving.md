# NEO Hybrid AI — Model Saving for Java Inference

## Overview
- Save models in PyTorch and ONNX formats
- Use pruning/distillation for efficiency
- Document model saving and loading

## Example (Python, PyTorch/ONNX)
import torch
import torch.nn as nn
import onnx

# Dummy model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(2, 1)
    def forward(self, x):
        return self.fc(x)

model = SimpleModel()
torch.save(model.state_dict(), 'model.pth')

# Export to ONNX
x = torch.randn(1, 2)
torch.onnx.export(model, x, 'model.onnx')

---
## Logging
- Log model saving, loading, and versioning
- Update this file as saving logic evolves