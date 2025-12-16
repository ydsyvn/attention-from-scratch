import numpy as np
from attention.math_utils import softmax

class BaseAttention:
    def __init__(self, d_model=128):
        self.d_model = d_model
        self.scale = 1.0 / np.sqrt(d_model)
        
        self.W_q = np.random.randn(d_model, d_model) * self.scale # (d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model) * self.scale # (d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model) * self.scale # (d_model, d_model)
    
    def forward(self, x):
        """
        forward
        
        :param x: The inputs -> (seq_len, d_model)
        """
        # extract sequence length
        seq_len = x.shape[0] # if x is shape (batch_size, seq_len, d_model) this will not work
        
        # compute Q, K and V
        Q = self.W_q @ x # (seq_len, d_model)
        K = self.W_k @ x # (seq_len, d_model)
        V = self.W_v @ x # (seq_len, d_model)
        
        # 
        mask = np.tril(np.ones((seq_len, seq_len))) # (seq_len, seq_len)

        attn_scores = Q @ K.transpose() / self.scale # (seq_len, seq_len)
        attn_scores = np.where(mask == 0, -1e9, attn_scores) # (seq_len, seq_len)
        
        # apply softmax
        attn_weights = softmax(attn_scores) # (seq_len, seq_len)
        
        output = attn_weights @ V # (seq_len, d_model)
        
        return output, attn_weights
