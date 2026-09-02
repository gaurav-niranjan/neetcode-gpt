import torch
import torch.nn as nn
from torchtyping import TensorType
import math

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.W_K = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_Q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.W_V = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.attention_dim = attention_dim


    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        B, S, D = embedded.shape
        K = self.W_K(embedded)  #(B, S, attention_dim)
        Q = self.W_Q(embedded)  #(B, S, attention_dim)
        V = self.W_V(embedded)  #(B, S, attention_dim)

        scores = (Q@K.transpose(-1,-2))/math.sqrt(self.attention_dim) #(B, S, S)
        mask = torch.tril(torch.ones((S, S), dtype=torch.bool))
        #print(mask)
        scores = scores.masked_fill(~mask, float('-inf')) #(B, S, S)
        #print(scores)
        attention = torch.softmax(scores, dim=2)

        return attention @ V
