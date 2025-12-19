import numpy as np
from src.attention.single_head import SingleHead

np.random.seed(42)


# === consts ===

EMBED_DIMS = 4
SEQUENCE_LENGTH = 3

inputs = np.random.randn(EMBED_DIMS, SEQUENCE_LENGTH)

attn = SingleHead(EMBED_DIMS)

output, attn_weights = attn.forward(inputs)


# === basic checks ===

print("output shape: ", output.shape) 
print("input shape: ", inputs.shape)
# print("attn_weights: \n", attn_weights)
