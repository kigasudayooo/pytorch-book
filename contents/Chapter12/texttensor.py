import torch
from transformers import BertTokenizer, BertModel

def text_to_embeddings(texts):
    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()  # Set to evaluation mode

    # Tokenize the input texts
    encoded = tokenizer(
        texts,
        padding=True,      # Pad shorter sequences to match longest
        truncation=True,   # Truncate sequences that are too long
        return_tensors='pt'  # Return PyTorch tensors
    )

    # Generate embeddings
    with torch.no_grad():  # No need to calculate gradients
        outputs = model(**encoded)
        embeddings = outputs.last_hidden_state

    return embeddings, encoded['input_ids']

# Example usage
texts = [
    "I love my dog",
    "The manatee became a doctor"
]

# Get embeddings
embeddings, token_ids = text_to_embeddings(texts)

# Print information about the embeddings
print(f"Input texts: {texts}")
print(f"Encodings: {token_ids}")
print(f"\nEmbedding tensor shape: {embeddings.shape}")
print("Shape explanation:")
print(f"- Number of sentences: {embeddings.shape[0]}")
print(f"- Words per sentence: {embeddings.shape[1]}")
print(f"- Embedding dimensions: {embeddings.shape[2]}")

# Look at one word's embedding
first_word_embedding = embeddings[0, 0]  # First sentence, first word
print(f"\nFirst word embedding (first 5 values): {first_word_embedding[:5]}")

# Decode tokens back to words
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
decoded = tokenizer.decode(token_ids[0])
print(f"\nDecoded tokens from first sentence: {decoded}")