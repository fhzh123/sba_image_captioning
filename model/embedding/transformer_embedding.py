import torch.nn as nn
from .token import TokenEmbedding
from .positional import PositionalEmbedding

class TransformerEmbedding(nn.Module):
    """
    Embedding which is consisted with under features
    1. TokenEmbedding : normal embedding matrix
    2. PositionalEmbedding : adding positional information using sin, cos
    sum of all these features are output of Embedding
    """

    def __init__(self, vocab_size, d_model, pad_idx=0, max_len=512, embedding_dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=d_model, pad_idx=pad_idx)
        self.position = PositionalEmbedding(d_model=d_model, max_len=max_len)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=embedding_dropout)
        
    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        x = self.norm(self.dropout(x))
        return self.dropout(x)