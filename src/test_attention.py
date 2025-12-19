import numpy as np
from attention.single_head import SingleHead

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

sums = np.sum(attn_weights, axis=-1)
is_normalized = np.allclose(sums, 1.0)

print(f"Softmax sums to 1.0: {is_normalized}")
if not is_normalized:
    print("Actual sums:", sums)
    
    
# 1. Shuffle the input columns
perm = np.random.permutation(SEQUENCE_LENGTH)
shuffled_inputs = inputs[:, perm]

# 2. Get output for shuffled input
shuffled_output, _ = attn.forward(shuffled_inputs)

# 3. Check if the shuffled original output matches the new output
# We compare output[:, perm] (original output shuffled) with shuffled_output
is_invariant = np.allclose(output[:, perm], shuffled_output)

print(f"Permutation Invariance Check: {is_invariant}")