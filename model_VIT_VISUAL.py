from torch import nn
from einops.layers.torch import Rearrange
from einops import repeat
from torch import Tensor
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2



def overlay_attention(image, attention_map, alpha=0.6):
    """
    Nakłada mapę uwagi na obraz wejściowy.

    :param image: Tensor obrazu wejściowego (C, H, W)
    :param attention_map: Tensor mapy uwagi (H, W)
    :param alpha: Współczynnik przezroczystości dla heatmapy
    :return: Obraz z nałożoną mapą uwagi
    """

    # Konwersja tensora obrazu na NumPy i normalizacja do [0, 255]
    image = image.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    image = (image - image.min()) / (image.max() - image.min()) * 255
    image = image.astype(np.uint8)


    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min()) * 255
    attention_map = attention_map.astype(np.uint8)

    # Konwersja mapy uwagi do heatmapy
    heatmap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)

    # Nałożenie heatmapy na obraz
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay


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
            nn.Linear(self.emb_size, self.num_hidden_layers),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.num_hidden_layers, self.emb_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, x_patches):
        x_norm1 = self.norm(x_patches)
        x_aten, attn_weights = self.multihead(x_norm1, x_norm1, x_norm1)  # attn_weights -> (B, num_heads, N, N)
        x_first_added = x_aten + x_patches

        x_norm2 = self.norm(x_first_added)
        x_mlp = self.MLP(x_norm2)

        return x_mlp + x_first_added, attn_weights


class ViT(nn.Module):
    def __init__(self, emb_size, num_heads, num_layers, num_hidden_layers, C_channels, patch_size, num_class, dropout, img_size=48):
        super().__init__()
        self.attention = nn.ModuleList([EncoderBlock(emb_size, num_heads, num_hidden_layers, dropout) for _ in range(num_layers)])
        self.patchemb = Patch_embedding(C_channels=C_channels, patch_size=patch_size, emb_size=emb_size, img_size=img_size)
        self.ff = nn.Linear(emb_size, num_class)

    def forward(self, x):
        embeddings = self.patchemb(x)
        attn_maps = []  # Lista do przechowywania map uwagi

        for layer in self.attention:
            embeddings, attn_weights = layer(embeddings)
            attn_maps.append(attn_weights)  # Zapisujemy mapy uwagi


        cls = embeddings[:, 0, :]
        x = self.ff(cls)

        return x, attn_maps  # Zwracamy predykcję i mapy uwagi


# Tworzymy losowy obraz o wymiarach (1, C, H, W)
image = torch.rand(1, 3, 48, 48)  # (batch=1, channels=3, height=48, width=48)

# Inicjalizujemy model
vit = ViT(emb_size=64, num_heads=8, num_layers=6, num_hidden_layers=128,
          C_channels=3, patch_size=8, num_class=10, dropout=0.1)

# Wykonujemy predykcję, aby otrzymać mapy uwagi
_, attn_maps = vit(image)

# Wybieramy mapę uwagi z ostatniej warstwy
last_layer_attention = attn_maps[-1]  # (B, num_heads, N, N)


# Wybieramy pierwszą głowicę
head_idx = 0
attention_map = last_layer_attention[0, head_idx].detach().numpy()
print("attention_map",attention_map)
print("attention_map",attention_map.shape)

attention_map = attention_map[1:37]  # Usuwamy token CLS
attention_map = attention_map.reshape(6, 6)  # Teraz jest kwadratowe

# Nałożenie mapy uwagi na obraz
overlayed_image = overlay_attention(image[0], attention_map)  # X[0] to pojedynczy obraz

# Wyświetlenie wyniku
plt.figure(figsize=(6, 6))
plt.imshow(overlayed_image)
plt.axis("off")
plt.title("Mapa uwagi nałożona na obraz")
plt.show()