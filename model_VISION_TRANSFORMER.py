from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat
from torch import Tensor
import torch
import numpy as np

def PositionEmbedding(seq_len, emb_size):
    embeddings = torch.ones(seq_len, emb_size)
    for i in range(seq_len):
        for j in range(emb_size):
            if j % 2 == 0:
                embeddings[i][j] = np.sin(i / (pow(10000, j / emb_size)))
            else:
                embeddings[i][j] = np.cos(i / (pow(10000, (j - 1) / emb_size)))
    return embeddings

class Patch_embedding(nn.Module):
    '''

    train_features torch.Size([64, 3, 48, 48])
    Patch_embedding torch.Size([64, 36, 128]) [batch_size, patches, em_size]


    '''
    def __init__(self, C_channels, patch_size, emb_size, img_size=48):
        super().__init__()

        self.len_seq = (img_size // patch_size)**2

        self.patch_size = patch_size
        # self.projection = nn.Sequential(
        #     # weź zdjęcię podziel na h1 * w1 pakiety i spłaszcz je
        #     Rearrange('b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=patch_size, w1=patch_size),
        #     nn.Linear(patch_size*patch_size*C_channels, emb_size)
        # )
        self.projection = nn.Sequential(
            nn.Conv2d(C_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        # embeddings
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_size)) # (1,1,emb_size)
        self.pos_embed = nn.Parameter(PositionEmbedding(self.len_seq + 1, emb_size)) # [36 + 1, 128]

    def forward(self, x) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Musi być typu [batch, C_in, H, W] (4-wymiarowy), a mamy {x.ndim} wymiary")
        N, C, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError("Rozmiar obrazu (H, W) musi być podzielny przez patch_size")
        x = self.projection(x) # ([64, 36, 128]) [batch_size, patches, em_size]

        #cls token i positional encoding
        cls_token = repeat(self.cls_token, ' () s e -> n s e', n=N) # (1,1,emb_size) -> (batch size, 1, emb_size)

        x = torch.cat([cls_token, x], dim=1) # (batch, patches + 1, emb_size) ([64, 37, 128])

        x = x + self.pos_embed

        return x



class EncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, num_hidden_layers, dropout):
        super().__init__()

        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers

        self.norm = nn.LayerNorm(self.emb_size)

        self.multihead = nn.MultiheadAttention(
            embed_dim=self.emb_size,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )

        self.MLP = nn.Sequential(
            nn.Linear(self.emb_size, self.num_hidden_layers), # patrz strona 5 tabela 1
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.num_hidden_layers, self.emb_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, x_patches):
        x_norm1 = self.norm(x_patches)
        x_aten, _ = self.multihead(x_norm1, x_norm1, x_norm1) #
        x_first_added = x_aten + x_patches

        x_norm2 = self.norm(x_first_added)
        x_mlp = self.MLP(x_norm2)

        return x_mlp + x_first_added

class Vit(nn.Module):
    def __init__(self, emb_size, num_heads, num_leyers, num_hidden_layers, C_channels, patch_size,num_class, dropout, img_size=48):
        super().__init__()
        self.attention = nn.Sequential(*[EncoderBlock(emb_size, num_heads,num_hidden_layers, dropout) for _ in range(num_leyers)])
        self.patchemb = Patch_embedding(C_channels =C_channels,patch_size=patch_size,emb_size=emb_size, img_size=img_size)
        self.ff = nn.Linear(emb_size, num_class)

    def forward(self, x):  # x -> (b, c, h, w)
        embeddings = self.patchemb(x)
        x = self.attention(embeddings)  # embeddings -> (b, (h/patch_size * w/patch_size) + 1, emb_dim)
        # # Pobieramy pierwszy token z sekwencji, czyli token CLS, który agreguje informacje z całego obrazu
        cls = x[:, 0, :]
        # Przekazujemy token CLS przez końcową warstwę liniową, aby uzyskać predykcję klasyfikacyjną
        x = self.ff(cls)
        return x