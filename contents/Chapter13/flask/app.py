from flask import Flask, request, jsonify
import torch
from model_def import SimpleLinearModel

app = Flask(__name__)

# Load the trained model
model = SimpleLinearModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    value = float(request.form.get('value', 0))
    input_tensor = torch.tensor([[value]], dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor)
    
    return jsonify({
        "input": value,
        "prediction": prediction.item()
    })

if __name__ == "__main__":
    app.run(port=5001)