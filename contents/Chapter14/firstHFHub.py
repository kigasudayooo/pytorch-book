from transformers import pipeline

# Load a small sentiment analysis model
classifier = pipeline("sentiment-analysis", model="prajjwal1/bert-tiny")

# Test the model
text = "I love programming with PyTorch!"
result = classifier(text)
print(result)  # Output: [{'label': 'POSITIVE', 'score': 0.9998}]
