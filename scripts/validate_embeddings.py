"""
Task 24: Validate C forward pass against PyTorch.
Runs the reference MiniLM model in Python and dumps the embedding for a test
sentence to a binary file that the C validation test can compare against.
"""
import struct
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
TEST_SENTENCE = "The quick brown fox jumps over the lazy dog."
OUT_EMBEDDING = "data/ref_embedding.bin"
OUT_TOKENS    = "data/ref_tokens.bin"

def mean_pool(token_embeddings, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

def main():
    print(f"Model : {MODEL_ID}")
    print(f"Input : {TEST_SENTENCE!r}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModel.from_pretrained(MODEL_ID)
    model.eval()

    inputs = tokenizer(TEST_SENTENCE, return_tensors="pt", padding=False, truncation=True)
    token_ids = inputs["input_ids"][0].tolist()
    seq_len   = len(token_ids)

    print(f"Tokens: {seq_len}  ids: {token_ids}")

    with torch.no_grad():
        out = model(**inputs)

    embedding = mean_pool(out.last_hidden_state, inputs["attention_mask"])
    embedding = embedding[0].numpy()

    import os
    os.makedirs("data", exist_ok=True)

    with open(OUT_TOKENS, "wb") as f:
        f.write(struct.pack("I", seq_len))
        for tid in token_ids:
            f.write(struct.pack("i", tid))

    with open(OUT_EMBEDDING, "wb") as f:
        f.write(struct.pack("I", len(embedding)))
        f.write(embedding.astype(np.float32).tobytes())

    print(f"Saved {seq_len} token ids  -> {OUT_TOKENS}")
    print(f"Saved {len(embedding)}-dim embedding -> {OUT_EMBEDDING}")
    print(f"L2 norm: {np.linalg.norm(embedding):.6f}")
    print(f"First 8 values: {embedding[:8]}")

if __name__ == "__main__":
    main()

