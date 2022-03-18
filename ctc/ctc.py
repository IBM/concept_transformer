# Cross-attention code based on https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/cct.py by Phil Wang
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# Pre-defined CTC Models
__all__ = ["mnist_ctc"]


def mnist_ctc(*args, **kwargs):
    return _cct(
        num_layers=2,
        num_heads=2,
        mlp_ratio=1,
        embedding_dim=128,
        img_size=28,
        n_input_channels=1,
        num_classes=2,
        n_unsup_concepts=0,
        n_concepts=10,
        n_spatial_concepts=0,
        *args,
        **kwargs,
    )


def tmnist_ctc(*args, **kwargs):
    return _cct(
        num_layers=2,
        n_conv_layers=2,
        num_heads=2,
        mlp_ratio=1,
        embedding_dim=256,
        n_input_channels=1,
        num_classes=10,
        n_concepts=0,
        n_spatial_concepts=11,
        pooling_kernel_size=4,
        pooling_stride=3,
        pooling_padding=2,
        max_seqlen=4,
        *args,
        **kwargs,
    )




def _cct(
    num_layers,
    num_heads,
    mlp_ratio,
    embedding_dim,
    kernel_size=3,
    stride=None,
    padding=None,
    *args,
    **kwargs,
):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    return CTC(
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        embedding_dim=embedding_dim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        *args,
        **kwargs,
    )


# Modules
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self, dim, n_outputs=None, num_heads=8, attention_dropout=0.1, projection_dropout=0.0
    ):
        super().__init__()
        n_outputs = n_outputs if n_outputs else dim
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)

        self.proj = nn.Linear(dim, n_outputs)
        self.proj_drop = nn.Dropout(projection_dropout)

    def forward(self, x, y):
        B, Nx, C = x.shape
        By, Ny, Cy = y.shape

        assert C == Cy, "Feature size of x and y must be the same"

        q = self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = (
            self.kv(y)
            .reshape(By, Ny, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q = q[0]
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class TransformerEncoderLayer(nn.Module):
    """
    Inspired by torch.nn.TransformerEncoderLayer and
    rwightman's timm package.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        attention_dropout=0.1,
        drop_path_rate=0.1,
    ):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = Attention(
            dim=d_model,
            num_heads=nhead,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        self.activation = F.gelu

    def forward(self, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        src = src + self.drop_path(self.self_attn(self.pre_norm(src)))
        src = self.norm1(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.drop_path(self.dropout2(src2))
        return src


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PretrainedTokenizer(nn.Module):
    def __init__(
        self, name="resnet50", embedding_dim=768, flatten=False, freeze=True, *args, **kwargs
    ):
        """
        Note: if img_size=448, number of tokens is 14 x 14
        """
        super().__init__()

        self.flatten = flatten
        self.embedding_dim = embedding_dim

        ver = torchvision.__version__.split("a")[0]
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load("pytorch/vision:v" + ver, name, pretrained=True)
        model.eval()

        # Remove final pooler and classifier
        self.n_features = model.fc.in_features
        model.avgpool = nn.Sequential()
        model.fc = nn.Sequential()

        # Freeze model
        self.model = model
        if freeze:
            self.model.requires_grad_(False)

        self.fc = nn.Conv2d(self.n_features, embedding_dim, kernel_size=1)

    def forward(self, x):
        out = self.model(x)

        width = int(math.sqrt(out.shape[1] / self.n_features))
        out = out.unflatten(-1, (self.n_features, width, width))
        out = self.fc(out)

        if self.flatten:
            return out.flatten(-2, -1).transpose(-2, -1)
        else:
            return out


class Tokenizer(nn.Module):
    """Applies strided convolutions to the input and then tokenizes (creates patches)"""

    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        n_conv_layers=1,
        n_input_channels=3,
        n_output_channels=64,
        in_planes=64,
        activation=None,
        max_pool=True,
        conv_bias=False,
        flatten=True,
    ):
        super().__init__()

        n_filter_list = (
            [n_input_channels]
            + [in_planes for _ in range(n_conv_layers - 1)]
            + [n_output_channels]
        )

        self.conv_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        n_filter_list[i],
                        n_filter_list[i + 1],
                        kernel_size=(kernel_size, kernel_size),
                        stride=(stride, stride),
                        padding=(padding, padding),
                        bias=conv_bias,
                    ),
                    nn.Identity() if activation is None else activation(),
                    nn.MaxPool2d(
                        kernel_size=pooling_kernel_size,
                        stride=pooling_stride,
                        padding=pooling_padding,
                    )
                    if max_pool
                    else nn.Identity(),
                )
                for i in range(n_conv_layers)
            ]
        )

        self.flatten = flatten
        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        out = self.forward(torch.zeros((1, n_channels, height, width)))
        if self.flatten:
            return out.shape[1]
        else:
            return self.flattener(out).transpose(-2, -1).shape[1]

    def forward(self, x):
        out = self.conv_layers(x)
        if self.flatten:
            return self.flattener(out).transpose(-2, -1)
        else:
            return out

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class FactorizedPositionEncoding(nn.Module):
    def __init__(self, max_seqlen, dim, embedding_dim):
        super().__init__()
        self.dim = dim
        self.max_seqlen = max_seqlen

        # Create factorized position embeddings
        self.positional_emb = nn.ParameterList()
        for i in range(dim):
            self.positional_emb.append(
                nn.Parameter(
                    torch.zeros(1, self.max_seqlen, *(dim - 1) * [1], embedding_dim).transpose(
                        1, i + 1
                    ),
                    requires_grad=True,
                )
            )

    def forward(self, x):
        x = F.pad(
            x,
            [j for i in x.shape[-1:1:-1] for j in [0, self.max_seqlen - i]],
            mode="constant",
            value=0,
        )
        x = x.transpose(-1, 1)
        x = x + sum(self.positional_emb)
        return x


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout_rate=0.1,
        attention_dropout=0.1,
        stochastic_depth_rate=0.1,
        positional_embedding="sine",
        sequence_length=None,
        *args,
        **kwargs,
    ):
        super().__init__()

        positional_embedding = (
            positional_embedding
            if positional_embedding in ["sine", "learnable", "none"]
            else "sine"
        )
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length

        assert sequence_length is not None or positional_embedding == "none", (
            f"Positional embedding is set to {positional_embedding} and"
            f" the sequence length was not specified."
        )

        if positional_embedding != "none":
            if positional_embedding == "learnable":
                self.positional_emb = nn.Parameter(
                    torch.zeros(1, sequence_length, embedding_dim), requires_grad=True
                )
                nn.init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = nn.Parameter(
                    self.sinusoidal_embedding(sequence_length, embedding_dim), requires_grad=False
                )
        else:
            self.positional_emb = None

        self.dropout = nn.Dropout(p=dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth_rate, num_layers)]
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout_rate,
                    attention_dropout=attention_dropout,
                    drop_path_rate=dpr[i],
                )
                for i in range(num_layers)
            ]
        )

        self.apply(self.init_weight)

    def forward(self, x):
        if self.positional_emb is not None:
            x += self.positional_emb
        elif self.sequence_length is not None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode="constant", value=0)

        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        return x

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor(
            [[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)] for p in range(n_channels)]
        )
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)


class ConceptTransformer(nn.Module):
    """
    Processes spatial and non-spatial concepts in parallel and aggregates the log-probabilities at the end
    """

    def __init__(
        self,
        embedding_dim=768,
        num_classes=10,
        num_heads=2,
        attention_dropout=0.1,
        projection_dropout=0.1,
        n_unsup_concepts=0,
        n_concepts=10,
        n_spatial_concepts=0,
        *args,
        **kwargs,
    ):
        super().__init__()

        # Unsupervised concepts
        self.n_unsup_concepts = n_unsup_concepts
        self.unsup_concepts = nn.Parameter(
            torch.zeros(1, n_unsup_concepts, embedding_dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.unsup_concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_unsup_concepts > 0:
            self.unsup_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

        # Non-spatial concepts
        self.n_concepts = n_concepts
        self.concepts = nn.Parameter(torch.zeros(1, n_concepts, embedding_dim), requires_grad=True)
        nn.init.trunc_normal_(self.concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_concepts > 0:
            self.concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

        # Sequence pooling for both non-spatial and unsupervised concepts
        if n_concepts > 0 or n_unsup_concepts > 0:
            self.token_attention_pool = nn.Linear(embedding_dim, 1)

        # Spatial Concepts
        self.n_spatial_concepts = n_spatial_concepts
        self.spatial_concepts = nn.Parameter(
            torch.zeros(1, n_spatial_concepts, embedding_dim), requires_grad=True
        )
        nn.init.trunc_normal_(self.spatial_concepts, std=1.0 / math.sqrt(embedding_dim))
        if n_spatial_concepts > 0:
            self.spatial_concept_tranformer = CrossAttention(
                dim=embedding_dim,
                n_outputs=num_classes,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
                projection_dropout=projection_dropout,
            )

    def forward(self, x):
        unsup_concept_attn, concept_attn, spatial_concept_attn = None, None, None

        out = 0
        if self.n_unsup_concepts > 0 or self.n_concepts > 0:
            token_attn = F.softmax(self.token_attention_pool(x), dim=1).transpose(-1, -2)
            x_pooled = torch.matmul(token_attn, x)

        if self.n_unsup_concepts > 0:  # unsupervised stream
            out_unsup, unsup_concept_attn = self.concept_tranformer(x_pooled, self.unsup_concepts)
            unsup_concept_attn = unsup_concept_attn.mean(1)  # average over heads
            out = out + out_unsup.squeeze(1)  # squeeze token dimension

        if self.n_concepts > 0:  # Non-spatial stream
            out_n, concept_attn = self.concept_tranformer(x_pooled, self.concepts)
            concept_attn = concept_attn.mean(1)  # average over heads
            out = out + out_n.squeeze(1)  # squeeze token dimension

        if self.n_spatial_concepts > 0:  # Spatial stream
            out_s, spatial_concept_attn = self.spatial_concept_tranformer(x, self.spatial_concepts)
            spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads
            out = out + out_s.mean(1)  # pool tokens sequence

        return out, unsup_concept_attn, concept_attn, spatial_concept_attn


# Shared methods
def ent_loss(probs):
    """Entropy loss"""
    ent = -probs * torch.log(probs + 1e-8)
    return ent.mean()


def concepts_sparsity_cost(concept_attn, spatial_concept_attn):
    cost = ent_loss(concept_attn) if concept_attn is not None else 0.0
    if spatial_concept_attn is not None:
        cost = cost + ent_loss(spatial_concept_attn)
    return cost


def concepts_cost(concept_attn, attn_targets):
    """Non-spatial concepts cost
        Attention targets are normalized to sum to 1,
        but cost is normalized to be O(1)

    Args:
        attn_targets, torch.tensor of size (batch_size, n_concepts): one-hot attention targets
    """
    if concept_attn is None:
        return 0.0
    if attn_targets.dim() < 3:
        attn_targets = attn_targets.unsqueeze(1)
    norm = attn_targets.sum(-1, keepdims=True)
    idx = ~torch.isnan(norm).squeeze()
    if not torch.any(idx):
        return 0.0
    # MSE requires both floats to be of the same type
    norm_attn_targets = (attn_targets[idx] / norm[idx]).float()
    n_concepts = norm_attn_targets.shape[-1]
    return n_concepts * F.mse_loss(concept_attn[idx], norm_attn_targets, reduction="mean")


def spatial_concepts_cost(spatial_concept_attn, attn_targets):
    """Spatial concepts cost
        Attention targets are normalized to sum to 1

    Args:
        attn_targets, torch.tensor of size (batch_size, n_patches, n_concepts):
            one-hot attention targets

    Note:
        If one patch contains a `np.nan` the whole patch is ignored
    """
    if spatial_concept_attn is None:
        return 0.0
    norm = attn_targets.sum(-1, keepdims=True)
    idx = ~torch.isnan(norm).squeeze()
    if not torch.any(idx):
        return 0.0
    norm_attn_targets = (attn_targets[idx] / norm[idx]).float()
    n_concepts = norm_attn_targets.shape[-1]
    return n_concepts * F.mse_loss(spatial_concept_attn[idx], norm_attn_targets, reduction="mean")


class CTC(nn.Module):
    """Concept Transformer Classifier"""

    def __init__(
        self,
        img_size=224,
        embedding_dim=768,
        n_input_channels=3,
        n_conv_layers=1,
        kernel_size=7,
        stride=2,
        padding=3,
        pooling_kernel_size=3,
        pooling_stride=2,
        pooling_padding=1,
        n_concepts=0,
        n_spatial_concepts=0,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.n_concepts = n_concepts
        self.n_spatial_concepts = n_spatial_concepts

        self.tokenizer = Tokenizer(
            n_input_channels=n_input_channels,
            n_output_channels=embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,
            pooling_padding=pooling_padding,
            max_pool=True,
            activation=nn.ReLU,
            n_conv_layers=n_conv_layers,
            conv_bias=False,
        )

        sequence_length = self.tokenizer.sequence_length(
            n_channels=n_input_channels, height=img_size, width=img_size
        )
        self.transformer_classifier = TransformerLayer(
            embedding_dim=embedding_dim, sequence_length=sequence_length, *args, **kwargs
        )
        self.norm = nn.LayerNorm(embedding_dim)

        self.concept_transformer = ConceptTransformer(
            embedding_dim=embedding_dim,
            attention_dropout=0.1,
            projection_dropout=0.1,
            n_concepts=n_concepts,
            n_spatial_concepts=n_spatial_concepts,
            *args,
            **kwargs,
        )

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.transformer_classifier(x)
        x = self.norm(x)
        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.concept_transformer(x)
        return out.squeeze(-2), unsup_concept_attn, concept_attn, spatial_concept_attn


class CTC_ADAPT(nn.Module):
    """Concept Transformer Classifier with pretrained tokenizer and adaptive number of tokens

    Args:
        max_seqlen (int): Maximal sequence length per dimension, which means that if max_seqlen=20 and
            inputs are 2D (images) then the number of tokens is 20*20=400
    """

    def __init__(
        self, embedding_dim=768, n_concepts=0, n_spatial_concepts=0, max_seqlen=14, *args, **kwargs
    ):
        super().__init__()

        self.n_concepts = n_concepts
        self.n_spatial_concepts = n_spatial_concepts

        self.tokenizer = PretrainedTokenizer(
            embedding_dim=embedding_dim, flatten=False, *args, **kwargs
        )
        self.position_embedding = FactorizedPositionEncoding(max_seqlen, 2, embedding_dim)
        self.flattener = nn.Flatten(1, 2)
        self.transformer_classifier = TransformerLayer(
            embedding_dim=embedding_dim, positional_embedding="none", *args, **kwargs
        )
        self.norm = nn.LayerNorm(embedding_dim)

        self.concept_transformer = ConceptTransformer(
            embedding_dim=embedding_dim,
            attention_dropout=0.1,
            projection_dropout=0.1,
            n_concepts=n_concepts,
            n_spatial_concepts=n_spatial_concepts,
            *args,
            **kwargs,
        )

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.position_embedding(x)
        x = self.flattener(x)
        x = self.transformer_classifier(x)
        x = self.norm(x)
        out, unsup_concept_attn, concept_attn, spatial_concept_attn = self.concept_transformer(x)
        return out.squeeze(-2), unsup_concept_attn, concept_attn, spatial_concept_attn
