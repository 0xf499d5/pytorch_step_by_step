from torch import nn
import numpy as np
import torch


class Transformer(nn.Module):
    def __init__(
        self, 
        n_layers: int,
        input_dim: int, 
        embed_dim: int,
        ffn_hidden_dim: int,
        n_heads: int, 
        max_seq_len: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.encoders = Encoders(
            n_layers=n_layers,
            input_dim=input_dim,
            embed_dim=embed_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate
        )
        self.decoders = Decoders(
            n_layers=n_layers,
            input_dim=input_dim,
            embed_dim=embed_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoders(x)
        y = self.decoders(x, h, is_masked=True)
        return y

class Encoders(nn.Module):
    def __init__(
        self, 
        n_layers: int,
        input_dim: int, 
        embed_dim: int,
        ffn_hidden_dim: int,
        n_heads: int, 
        max_seq_len: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.word_emb = WordEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim
        )
        self.pos_emb = PositionEmbedding(
            max_seq_len=max_seq_len,
            model_dim=embed_dim
        )
        self.encoders = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=ffn_hidden_dim,
                dropout=dropout_rate,
                batch_first=True,
                norm_first=True
            ),
            num_layers=n_layers,
            norm=nn.LayerNorm(embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.word_emb(x)
        x = self.pos_emb(x)
        y = self.encoders(x)
        return y


class Decoders(nn.Module):
    def __init__(
        self, 
        n_layers: int,
        input_dim: int, 
        embed_dim: int,
        ffn_hidden_dim: int,
        n_heads: int, 
        max_seq_len: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.word_emb = WordEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim
        )
        self.pos_emb = PositionEmbedding(
            max_seq_len=max_seq_len,
            model_dim=embed_dim
        )
        self.decoders = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=n_heads,
                dim_feedforward=ffn_hidden_dim,
                dropout=dropout_rate,
                batch_first=True,
                norm_first=True
            ),
            num_layers=n_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        self.ol = nn.Linear(embed_dim, input_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        h: torch.Tensor, 
        is_masked: bool = True
    ) -> torch.Tensor:
        _, S, _ = x.shape
        x = self.word_emb(x)
        x = self.pos_emb(x)
        m = self.generate_square_subsequent_mask(S)
        y = self.decoders(
            tgt=x,
            memory=h,
            tgt_is_causal=is_masked,
            tgt_mask=m
        )
        y = self.ol(y)
        return y
    
    def generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask.masked_fill_(mask.bool(), -torch.inf)
        return mask


class WordEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.we = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, I = x.shape
        assert I == self.input_dim, "ShapeError"
        xe = self.we(x)
        return xe


class PositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, model_dim: int):
        super().__init__()
        assert model_dim % 2 == 0, "model_dim must be a even number"
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim
        self._init_position_embedding_vectors()

    def _init_position_embedding_vectors(self):
        """
        pe(i, j, e) = sin(10000 ** (-j / e)), if j % 2 == 0
        pe(i, j, e) = cos(10000 ** (-j / e)), if j % 2 == 1
        i := position, j := feature dimention index, e := model_dim

          10000 ** (-j / e))
        = e ** [(-j/e) * ln(10000)]
        """
        i = torch.arange(0, self.max_seq_len).float().unsqueeze(1)  # (S, 1)
        e = self.model_dim
        j = torch.arange(0, e, 2).float()  # (S,)

        pe = torch.zeros(self.max_seq_len, self.model_dim)
        w = i * torch.exp((-j/e) * np.log(10000))
        pe[:, 0::2] = torch.sin(w)
        pe[:, 1::2] = torch.cos(w)
        pe = pe.unsqueeze(0)  # (B, S, E)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        assert E == self.model_dim and S <= self.max_seq_len, "Error!"
        xpe = self.pe[:, :S, :] + x / np.sqrt(self.model_dim)
        return xpe