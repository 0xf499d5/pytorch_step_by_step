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
        self.encoders = nn.ModuleList([
            Encoder(
                input_dim=embed_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                dropout_rate=dropout_rate
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.word_emb(x)
        x = self.pos_emb(x)
        y = x
        for encoder in self.encoders:
            y = encoder(y)
        y = self.norm(y)
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
        self.decoders = nn.ModuleList([
            Decoder(
                input_dim=embed_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                dropout_rate=dropout_rate
            ) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.ol = nn.Linear(embed_dim, input_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        h: torch.Tensor, 
        is_masked: bool = True
    ) -> torch.Tensor:
        x = self.word_emb(x)
        x = self.pos_emb(x)
        y = x
        for decoder in self.decoders:
            y = decoder(y, h, is_masked=is_masked)
        y = self.norm(y)
        y = self.ol(y)
        return y



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


class Encoder(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        ffn_hidden_dim: int,
        n_heads: int, 
        max_seq_len: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            SelfAttentionBlock(
                input_dim=input_dim,
                n_heads=n_heads,
                max_seq_len=max_seq_len,
                dropout_rate=dropout_rate
            ),
            FeedForwardBlock(
                input_dim=input_dim,
                hidden_dim=ffn_hidden_dim,
                output_dim=input_dim,
                dropout_rate=dropout_rate
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: add source mask
        y = self.blocks(x)
        return y


class Decoder(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        ffn_hidden_dim: int,
        n_heads: int, 
        max_seq_len: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.self_attn = SelfAttentionBlock(
            input_dim=input_dim,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate
        )
        self.cross_attn = CrossAttentionBlock(
            input_dim=input_dim,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate
        )
        self.ffn = FeedForwardBlock(
            input_dim=input_dim,
            hidden_dim=ffn_hidden_dim,
            output_dim=input_dim,
            dropout_rate=dropout_rate
        )

    def forward(
        self, 
        x: torch.Tensor, 
        h: torch.Tensor, 
        is_masked: bool = True
    ) -> torch.Tensor:
        y = self.self_attn(x, is_masked=is_masked)
        y = self.cross_attn(x, h)
        y = self.ffn(y)
        return y
    

class SelfAttentionBlock(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        n_heads: int, 
        max_seq_len: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim

        self.norm = nn.LayerNorm(input_dim)
        self.mha = MultiHeadAttention(
            input_dim=input_dim,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate
        )
        self.dp = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, is_masked: bool = False) -> torch.Tensor:
        B, S, E = x.shape
        assert E == self.input_dim, "Error!"

        y = self.norm(x)
        y = self.mha(y, is_masked=is_masked)
        y = self.dp(y)
        return x + y
    

class CrossAttentionBlock(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        n_heads: int, 
        max_seq_len: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim

        self.norm = nn.LayerNorm(input_dim)
        self.mha = MultiHeadAttention(
            input_dim=input_dim,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate
        )
        self.dp = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        assert E == self.input_dim, "Error!"

        y = self.norm(x)
        self.mha.init_kv(hidden_states=h)  # TODO
        y = self.mha(y, is_masked=False)
        y = self.dp(y)
        return x + y


class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        n_heads: int, 
        max_seq_len: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        assert input_dim % n_heads == 0, "Error!"

        self.input_dim = input_dim  # embed_dim or model_dim
        self.n_heads = n_heads
        self.head_dim = input_dim // n_heads
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate

        self.wq = nn.Linear(input_dim, input_dim)  # query weights
        self.wk = nn.Linear(input_dim, input_dim)  # key weights
        self.wv = nn.Linear(input_dim, input_dim)  # value weights
        self.dp = nn.Dropout(dropout_rate)         # dropout
        self.ol = nn.Linear(input_dim, input_dim)  # output linear
        
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("mk", mask)  # target mask matrix

        self.k = None
        self.v = None

    def init_kv(self, hidden_states: torch.Tensor):
        B, S, E = hidden_states.shape
        try:
            assert E == self.input_dim
        except AssertionError as e:
            print("-" * 50)
            print(f"E: {E}, self.input_dim: {self.input_dim}")
            print("-" * 50)
            raise e

        self.k = self.wk(hidden_states).view(B, S, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        self.v = self.wv(hidden_states).view(B, S, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

    def clear_kv(self):
        self.k = None
        self.v = None

    def forward(self, x: torch.Tensor, is_masked: bool = False) -> torch.Tensor:
        B, S, E = x.shape  # (Batch size, Seq len, Embedding dim)
        assert E == self.input_dim, "Error!"

        # q = self.wq(x)
        # q = q.view(B, S, self.n_heads, self.head_dim)  # (B, S, N, H)
        # q = q.contiguous()
        # q = q.transpose(1, 2)  # (B, N, S, H)
        q = self.wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

        if self.k is None and self.v is None:
            self.init_kv(x)
        k, v = self.k, self.v

        a = q @ k.transpose(2, 3)
        a /= np.sqrt(self.input_dim)
        if is_masked:
            a.masked_fill_(self.mk[:S, :S], -torch.inf)
        a = torch.softmax(a, dim=-1)
        a = self.dp(a)
        a = a.detach()
        c = a @ v
        # c = c.transpose(1, 2).view(B, S, -1).contiguous()
        c = c.transpose(1, 2).contiguous().view(B, S, -1)
        o = self.ol(c)
        return o


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim

        self.norm = nn.LayerNorm(input_dim)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.ffn(y)
        return x + y