import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load a pretrained ResNet model
model = models.resnet18(pretrained=True)
model.eval()

# ImageNet class labels (simplified - just a few examples)
class_names = [
    'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead shark',
    'electric ray', 'stingray', 'rooster', 'hen', 'ostrich', 'brambling',
    'goldfinch', 'house finch', 'junco', 'indigo bunting', 'robin', 'bulbul',
    'jay', 'magpie', 'chickadee'
]

# Simulate model output for demonstration
# This would normally come from model(input_tensor)
example_output = torch.tensor([
    [ 1.2,  4.5, -0.8,  2.1,  0.3,  # First image predictions
     -1.5,  0.9,  3.2, -0.4,  1.1,
      0.5, -0.2,  1.8,  0.7, -1.0,
      2.8,  1.6, -0.6,  0.4,  1.3],
    [-0.5,  5.2,  0.3,  1.4, -0.8,  # Second image predictions
      0.9,  1.2,  2.8,  0.6,  1.5,
     -1.1,  0.4,  2.1,  0.2, -0.7,
      1.9,  0.8, -0.3,  1.6,  0.5]
])

def interpret_output(output_tensor, top_k=5):
    # Apply softmax to convert logits to probabilities
    probabilities = torch.nn.functional.softmax(output_tensor, dim=1)
    
    # Get top k probabilities and class indices
    top_probs, top_indices = torch.topk(probabilities, k=top_k)
    
    # Convert to numpy for easier handling
    top_probs = top_probs.numpy()
    top_indices = top_indices.numpy()
    
    return top_probs, top_indices

# Interpret the outputs
top_probs, top_indices = interpret_output(example_output)

# Print results for each image
for img_idx in range(len(example_output)):
    print(f"\nImage {img_idx + 1} Predictions:")
    print("------------------------")
    print(f"Raw logits (first 5): {example_output[img_idx, :5].tolist()}")
    
    # Print top predictions
    print("\nTop 5 Predictions:")
    for rank, (prob, class_idx) in enumerate(zip(top_probs[img_idx], top_indices[img_idx])):
        class_name = class_names[class_idx]
        print(f"{rank + 1}. {class_name}: {prob:.1%}")

# Different ways to access the predictions
print("\nDifferent ways to work with the output tensor:")
print("---------------------------------------------")

# 1. Get the single highest confidence prediction
max_conf, predicted_class = torch.max(example_output, dim=1)
print(f"\n1. Highest confidence prediction (first image):")
print(f"   Class index: {predicted_class[0]}")
print(f"   Class name: {class_names[predicted_class[0]]}")
print(f"   Raw logit value: {max_conf[0]:.2f}")

# 2. Get all predictions above a threshold
threshold = 2.0
above_threshold = torch.where(example_output[0] > threshold)[0]
print(f"\n2. Predictions above threshold {threshold}:")
for idx in above_threshold:
    print(f"   {class_names[idx]}: {example_output[0][idx]:.2f}")

# 3. Get prediction differences (useful for analyzing model uncertainty)
sorted_values, _ = torch.sort(example_output[0], descending=True)
top_difference = sorted_values[0] - sorted_values[1]
print(f"\n3. Confidence margin (difference between top predictions): {top_difference:.2f}")

# 4. Convert to probabilities and get confidence
probabilities = torch.nn.functional.softmax(example_output, dim=1)
confidence = torch.max(probabilities, dim=1).values
print(f"\n4. Model confidence (probability) for top prediction: {confidence[0]:.1%}")

# 5. Categorical entropy (measure of prediction uncertainty)
entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
print(f"\n5. Prediction entropy (higher means more uncertain): {entropy[0]:.2f}")