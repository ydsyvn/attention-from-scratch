import numpy as np
import torch
import torch.nn as nn
from attention.base import BaseAttention

np.random.seed(42)


# consts
EMBED_DIMS = 4
SEQUENCE_LENGTH = 3

inputs = np.random.randn(EMBED_DIMS, SEQUENCE_LENGTH)

attn = BaseAttention(EMBED_DIMS)

output, attn_weights = attn.forward(inputs)

# basic checks
print("output shape: ", output.shape) 
print("input shape: ", inputs.shape)
# print("attn_weights: \n", attn_weights)

# compare against pytorch
torch.manual_seed(42)

pytorch_attn = nn.MultiheadAttention(EMBED_DIMS, num_heads=1) # single head attention
pytorch_output = pytorch_attn.forward(inputs)

print("output: \n", output)
print("pytorch_output: \n", pytorch_output)

print("outputs are the same. The implementation is correct!" if output == pytorch_output else "outputs aren't the same. The implementation is incorrect.")