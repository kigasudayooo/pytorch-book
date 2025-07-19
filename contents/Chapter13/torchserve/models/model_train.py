# model.py
import torch
import torch.nn as nn
import torch.optim as optim
from model_def import SimpleLinearModel

def train_model():
    model = SimpleLinearModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    xs = torch.tensor([[-1.0], [0.0], [1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
    ys = torch.tensor([[-3.0], [-1.0], [1.0], [3.0], [5.0], [7.0]], dtype=torch.float32)

    for _ in range(500):
        optimizer.zero_grad()
        outputs = model(xs)
        loss = criterion(outputs, ys)
        loss.backward()
        optimizer.step()

    return model
# Save model
model = train_model()
torch.save(model.state_dict(), "model.pth")