# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import torch


def extend_valid_token_mask(
    valid_token_mask: Optional[torch.Tensor], num_registers: int
) -> Optional[torch.Tensor]:
    """
    Extends the valid token mask to include the register tokens.

    Parameters:
    - valid_token_mask (B, N): The original valid token mask.

    Returns:
    - extended_valid_token_mask (B, N + R): The extended valid token mask.
    """
    if valid_token_mask is None:
        return None
    return torch.cat(
        [
            valid_token_mask,
            torch.ones(
                valid_token_mask.shape[0],
                num_registers,
                dtype=torch.bool,
                device=valid_token_mask.device,
            ),
        ],
        dim=1,
    )


class MLPBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act_layer: Type[torch.nn.Module],
    ) -> None:
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(n, k), act_layer())
            for n, k in zip([input_dim] + h, [hidden_dim] * num_layers)
        )
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.fc(x)


class MultiMLP(torch.nn.Module):
    def __init__(
        self,
        width: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        act_layer: Type[torch.nn.Module],
    ) -> None:
        super().__init__()

        self.layers = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Conv1d(n * width, k * width, 1, groups=width), act_layer()
            )
            for n, k in zip(
                [input_dim] + [hidden_dim] * (num_layers - 1), [hidden_dim] * num_layers
            )
        )
        self.fc = torch.nn.Conv1d(
            hidden_dim * width, output_dim * width, 1, groups=width
        )

    def forward(self, x):
        y = x.flatten(1).unsqueeze(-1)
        for layer in self.layers:
            y = layer(y)
        return self.fc(y).squeeze(-1).unflatten(-1, (x.shape[1], -1))


class Block(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        act_layer: Type[torch.nn.Module],
        mlp_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if mlp_dim is None:
            mlp_dim = 4 * dim

        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.mlp = MLPBlock(
            input_dim=dim,
            hidden_dim=mlp_dim,
            output_dim=dim,
            num_layers=1,
            act_layer=act_layer,
        )

    def forward(
        self, x: torch.Tensor, invalid_token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        x = self.attn(x, x, x, key_padding_mask=invalid_token_mask)[0]
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoder(torch.nn.Module):
    def __init__(
        self,
        num_tokens: int,
        num_registers: int,
        patch_size: int,
        depth: int,
        feature_dim: int,
        embedding_dim: int,
        num_heads: int,
        act_layer: Type[torch.nn.Module],
    ):
        super().__init__()

        self.patch_emb = torch.nn.Linear(patch_size * patch_size * 3, embedding_dim)

        self.pos_enc = torch.nn.Parameter(torch.randn(num_tokens, embedding_dim))

        self.reg_tokens = torch.nn.Parameter(torch.randn(num_registers, embedding_dim))

        self.blocks = torch.nn.ModuleList()

        for _ in range(depth):
            self.blocks.append(
                Block(
                    dim=embedding_dim,
                    num_heads=num_heads,
                    act_layer=act_layer,
                )
            )

        self.neck = MLPBlock(
            input_dim=embedding_dim,
            hidden_dim=feature_dim,
            output_dim=feature_dim,
            num_layers=1,
            act_layer=torch.nn.ReLU,
        )

    def forward(
        self, foveated_tokens: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Parameters:
        - foveated_tokens (B, N, E): The input foveated image tokens.
        - valid_token_mask (B, N): An optional mask indicating which tokens are valid.

        Returns:
        - features (B, N, C): The output feature vectors, one per input token
        - register_features (B, N, C): The register feature vectors, one per register token
        """
        embedded_tokens = self.patch_emb(
            foveated_tokens.flatten(2)
        ) + self.pos_enc.unsqueeze(0)

        num_registers = self.reg_tokens.shape[0]

        embedded_tokens = torch.cat(
            [
                embedded_tokens,
                self.reg_tokens.unsqueeze(0).expand(embedded_tokens.shape[0], -1, -1),
            ],
            dim=1,
        )

        valid_token_mask = extend_valid_token_mask(valid_token_mask, num_registers)

        invalid_token_mask = (
            valid_token_mask.logical_not() if valid_token_mask is not None else None
        )

        transformed_tokens = embedded_tokens
        for block in self.blocks:
            transformed_tokens = block(transformed_tokens, invalid_token_mask)

        features = self.neck(transformed_tokens)

        return features.split([features.shape[1] - num_registers, num_registers], dim=1)


class MultiheadCrossAttention(torch.nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()

        self.num_heads = num_heads

        self.q_proj = torch.nn.Linear(embedding_dim, attention_dim)
        self.kv_proj = torch.nn.Linear(embedding_dim, 2 * attention_dim)

        self.out_proj = torch.nn.Linear(attention_dim, embedding_dim)

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.q_proj(queries)
        k, v = self.kv_proj(context).chunk(2, dim=-1)

        x = (
            torch.nn.functional.scaled_dot_product_attention(
                q.unflatten(-1, (self.num_heads, -1)).permute(0, 2, 1, 3),
                k.unflatten(-1, (self.num_heads, -1)).permute(0, 2, 1, 3),
                v.unflatten(-1, (self.num_heads, -1)).permute(0, 2, 1, 3),
                attn_mask=attn_mask.unsqueeze(1) if attn_mask is not None else None,
            )
            .permute(0, 2, 1, 3)
            .flatten(2)
        )

        projected = self.out_proj(x)

        return projected


class TwoWayAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        act_layer: Type[torch.nn.Module],
        cross_attention_downsample_rate: int,
    ):
        super().__init__()

        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)
        self.norm3 = torch.nn.LayerNorm(embedding_dim)
        self.norm4 = torch.nn.LayerNorm(embedding_dim)

        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_attn_token_to_image = MultiheadCrossAttention(
            embedding_dim=embedding_dim,
            attention_dim=embedding_dim // cross_attention_downsample_rate,
            num_heads=num_heads,
        )
        self.cross_attn_image_to_token = MultiheadCrossAttention(
            embedding_dim=embedding_dim,
            attention_dim=embedding_dim // cross_attention_downsample_rate,
            num_heads=num_heads,
        )

        self.mlp = MLPBlock(
            input_dim=embedding_dim,
            hidden_dim=mlp_dim,
            output_dim=embedding_dim,
            num_layers=1,
            act_layer=act_layer,
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shortcut = query_tokens

        query_tokens = self.norm1(query_tokens)

        query_tokens = self.self_attn(
            query_tokens,
            query_tokens,
            query_tokens,
        )[0]
        shortcut = shortcut + query_tokens

        query_tokens = self.norm2(shortcut)

        # When image tokens are keys, the valid token mask should broadcast across values (dim 1)
        query_tokens = self.cross_attn_token_to_image(
            query_tokens,
            context_tokens,
            attn_mask=valid_token_mask.unsqueeze(1)
            if valid_token_mask is not None
            else None,
        )

        shortcut = shortcut + query_tokens

        query_tokens = self.norm3(shortcut)

        query_tokens = shortcut + self.mlp(query_tokens)

        shortcut = context_tokens

        context_tokens = self.norm4(context_tokens)

        # When image tokens are queries, we don't need to mask out the invalid tokens as their values
        # won't be used anyway.
        context_tokens = self.cross_attn_image_to_token(context_tokens, query_tokens)

        context_tokens = shortcut + context_tokens

        return query_tokens, context_tokens


class TwoWayTransformer(torch.nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        act_layer: Type[torch.nn.Module],
        cross_attention_downsample_rate: int = 2,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList()

        for _ in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    act_layer=act_layer,
                    cross_attention_downsample_rate=cross_attention_downsample_rate,
                )
            )

        self.final_norm = torch.nn.LayerNorm(embedding_dim)
        self.final_attn_token_to_image = MultiheadCrossAttention(
            embedding_dim=embedding_dim,
            attention_dim=embedding_dim // cross_attention_downsample_rate,
            num_heads=num_heads,
        )

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            query_tokens, context_tokens = layer(
                query_tokens, context_tokens, valid_token_mask
            )

        query_tokens = query_tokens + self.final_attn_token_to_image(
            self.final_norm(query_tokens),
            context_tokens,
            attn_mask=valid_token_mask.unsqueeze(1)
            if valid_token_mask is not None
            else None,
        )

        return query_tokens, context_tokens


# From https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/common.py
# Itself from https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(torch.nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MaskDecoder(torch.nn.Module):
    def __init__(
        self,
        num_tokens: int,
        feature_dim: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        num_masks: int,
        num_registers: int,
        upsample_dim_reduction_rates: List[int],
        mask_head_hidden_dim: int,
        mask_head_depth: int,
        iou_head_hidden_dim: int,
        iou_head_depth: int,
        transformer_act_layer: Type[torch.nn.Module] = torch.nn.ReLU,
        upsample_act_layer: Type[torch.nn.Module] = torch.nn.GELU,
        head_act_layer: Type[torch.nn.Module] = torch.nn.ReLU,
        cross_attention_downsample_rate: int = 2,
    ):
        super().__init__()

        self.iou_token = torch.nn.Parameter(torch.randn(feature_dim))

        self.mask_tokens = torch.nn.Parameter(torch.randn(num_masks, feature_dim))

        self.reg_enc = torch.nn.Parameter(torch.randn(num_registers, feature_dim))

        self.pos_enc = torch.nn.Parameter(torch.randn(num_tokens, feature_dim))

        self.transformer = TwoWayTransformer(
            depth=depth,
            embedding_dim=feature_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            act_layer=transformer_act_layer,
            cross_attention_downsample_rate=cross_attention_downsample_rate,
        )

        self.output_upscaling = torch.nn.Sequential()
        previous_dim_reduction_rate = 1
        for idx, dim_reduction_rate in enumerate(upsample_dim_reduction_rates):
            self.output_upscaling.append(
                torch.nn.ConvTranspose2d(
                    feature_dim // previous_dim_reduction_rate,
                    feature_dim // dim_reduction_rate,
                    kernel_size=2,
                    stride=2,
                )
            )
            if idx < (len(upsample_dim_reduction_rates) - 1):
                self.output_upscaling.append(
                    LayerNorm2d(feature_dim // dim_reduction_rate)
                )
            self.output_upscaling.append(upsample_act_layer())

            previous_dim_reduction_rate = dim_reduction_rate

        self.mask_heads = MultiMLP(
            width=num_masks,
            input_dim=feature_dim,
            hidden_dim=mask_head_hidden_dim,
            output_dim=feature_dim // upsample_dim_reduction_rates[-1],
            num_layers=mask_head_depth - 1,
            act_layer=head_act_layer,
        )

        self.iou_prediction_head = MLPBlock(
            input_dim=feature_dim,
            hidden_dim=iou_head_hidden_dim,
            output_dim=num_masks,
            num_layers=iou_head_depth - 1,
            act_layer=head_act_layer,
        )

    def forward(
        self,
        image_features: torch.Tensor,
        register_features: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
        - image_features (B, N, C): The input image features.
        - register_features (B, N, C): The input register features.
        - valid_token_mask (B, N): An optional boolean mask indicating which tokens are valid.

        Returns:
        - segmentation logit tokens (B, K, N, H, W): The output foveated segmentation logit tokens.
        - iou predictions (B, K): The output IoU prediction logits.
        """

        query_tokens = (
            torch.cat([self.iou_token.unsqueeze(0), self.mask_tokens], dim=0)
            .unsqueeze(0)
            .expand(image_features.shape[0], -1, -1)
        )

        context_tokens = torch.cat(
            [
                image_features + self.pos_enc.unsqueeze(0),
                register_features + self.reg_enc.unsqueeze(0),
            ],
            dim=1,
        )

        valid_token_mask = extend_valid_token_mask(
            valid_token_mask, register_features.shape[1]
        )

        query_tokens, context_tokens = self.transformer(
            query_tokens, context_tokens, valid_token_mask
        )

        image_tokens = context_tokens[:, : image_features.shape[1], :]

        # add H, W dimensions to end, flatten batch and token (B, N, E) -> (BN, E, H, W)
        b, n = image_tokens.shape[:2]
        upscaled_embedding = image_tokens.view(*image_tokens.shape, 1, 1).flatten(0, 1)

        upscaled_embedding = self.output_upscaling(upscaled_embedding).unflatten(
            0, (b, n)
        )

        h, w = upscaled_embedding.shape[-2:]

        mask_head_out = self.mask_heads(query_tokens[:, 1:, :])

        masks = torch.einsum(
            "bke,bnehw->bknhw",
            mask_head_out,
            upscaled_embedding,
        )

        iou_prediction = self.iou_prediction_head(query_tokens[:, 0, :])

        return masks, iou_prediction


class SegmentThisThing(torch.nn.Module):
    """
    A neural network module implementing the Segment This Thing model.

    Attributes:
    - num_tokens (int): Number of tokens in the input (depends on the foveation pattern).
    - num_registers (int): Number of register tokens used in addition to image patch tokens.
    - patch_size (int): The size of the input image patches.
    - encoder_depth (int): The number of Transformer layers in the image encoder.
    - encoder_embedding_dim (int): The embedding dimenson of the image encoder.
    - encoder_num_heads (int): The number of attention heads in each layer of the image encooder.
    - feature_dim (int): The output size of the image encoder. Each input token is represented by a vector of length `feature_dim` which is computed by linear projection of the final Transformer layers output, and thus may differ from the embedding dimension.
    - decoder_depth (int): The number of bidirectional Transformer layers in the mask decoder.
    - decoder_num_heads (int): The number of attention heads in each layer of the mask decoder.
    - decoder_upsample_dim_reduction_rates (List[int]): The mask decoder performs N doublings of the segmentation resolution (via deconvolution) where N is the length of this list. The values in the list determine the number of features in the output of each upsampling, specified as a ratio w.r.t. the number of input features (equal to `feature_dim`).
    - decoder_mask_head_hidden_dim (int): The dimensionality of hidden layers in the mask head MLP.
    - decoder_mask_head_depth (int): The depth of the mask head MLP.
    - decoder_mlp_dim (int): The feedforward dimension in bidirectional Transformer layers in the mask decoder.
    - decoder_iou_head_depth (int): The depth of the IoU prediction head MLP.
    - decoder_iou_head_hidden_dim (int): The hidden dimension for the IoU prediction head MLP.
    - num_masks (int): The number of masks to predict for each token set.
    """

    def __init__(
        self,
        num_tokens: int,
        num_registers: int,
        patch_size: int,
        encoder_depth: int,
        encoder_embedding_dim: int,
        encoder_num_heads: int,
        feature_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        decoder_upsample_dim_reduction_rates: List[int],
        decoder_mask_head_hidden_dim: int,
        decoder_mask_head_depth: int,
        decoder_mlp_dim: int = 2048,
        decoder_iou_head_depth: int = 3,
        decoder_iou_head_hidden_dim: int = 256,
        num_masks: int = 3,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.patch_size = patch_size

        self.image_encoder = ImageEncoder(
            num_tokens=num_tokens,
            num_registers=num_registers,
            patch_size=patch_size,
            depth=encoder_depth,
            feature_dim=feature_dim,
            embedding_dim=encoder_embedding_dim,
            num_heads=encoder_num_heads,
            act_layer=torch.nn.ReLU,
        )

        self.mask_decoder = MaskDecoder(
            num_tokens=num_tokens,
            feature_dim=feature_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_dim=decoder_mlp_dim,
            num_masks=num_masks,
            num_registers=num_registers,
            upsample_dim_reduction_rates=decoder_upsample_dim_reduction_rates,
            mask_head_hidden_dim=decoder_mask_head_hidden_dim,
            mask_head_depth=decoder_mask_head_depth,
            iou_head_depth=decoder_iou_head_depth,
            iou_head_hidden_dim=decoder_iou_head_hidden_dim,
        )

    def forward(
        self,
        foveated_tokens: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Given foveated image tokens, return foveated segmentation tokens. The number of input
        tokens must match the `num_tokens` parameter passed to the constructor, and the width and
        height of the patches must match the `patch_size` parameter. An optional boolean mask
        can be passed. Any tokens with a value of False in the mask will be ignored.

        The output segmentation is also foveated, with the same number of tokens as the input. They
        are logits (before sigmoid), and there will be K segmentation masks for each input token set
        where K corresponds to the `num_masks` parameter passed to the constructor.

        Parameters:
        - foveated_tokens (B, N, C, H, W): The input foveated image tokens.
        - valid_token_mask (B, N): An optional mask indicating which tokens are valid.

        Returns:
        - segmentation logit tokens (B, K, N, H, W): The output foveated segmentation logit tokens.
        """
        if foveated_tokens.ndim != 5:
            raise ValueError(
                "[SegmentThisThing.forward]: Expected 5D foveated token input (B, N, C, H, W)"
            )
        if valid_token_mask is not None:
            if valid_token_mask.ndim != 2:
                raise ValueError("Expected 2D valid token mask (B, N)")
            if valid_token_mask.dtype != torch.bool:
                raise ValueError("Expected boolean valid token mask")

        if (
            foveated_tokens.size(-1) != self.patch_size
            or foveated_tokens.size(-2) != self.patch_size
        ):
            raise ValueError(
                f"[SegmentThisThing.forward]: Expected foveated token input to have patch size {self.patch_size}, got {foveated_tokens.size(-1)}x{foveated_tokens.size(-2)}"
            )
        if foveated_tokens.size(-3) != 3:
            raise ValueError(
                f"[SegmentThisThing.forward]: Expected foveated token input to have 3 channels, got {foveated_tokens.size(-3)}"
            )
        if foveated_tokens.size(1) != self.num_tokens:
            raise ValueError(
                f"[SegmentThisThing.forward]: Expected foveated token input to have {self.num_tokens} tokens, got {foveated_tokens.size(1)}"
            )

        image_features, register_features = self.image_encoder(
            foveated_tokens, valid_token_mask
        )

        return self.mask_decoder(image_features, register_features, valid_token_mask)


def build_segment_this_thing(
    token_size: int,
    num_tokens: int,
    encoder_depth: int,
    encoder_embedding_dim: int,
    encoder_num_heads: int,
) -> SegmentThisThing:
    return SegmentThisThing(
        num_tokens=num_tokens,
        num_registers=1,
        patch_size=token_size,
        encoder_depth=encoder_depth,
        encoder_embedding_dim=encoder_embedding_dim,
        encoder_num_heads=encoder_num_heads,
        decoder_depth=2,
        decoder_num_heads=8,
        decoder_upsample_dim_reduction_rates=[2, 4, 8, 8],
        decoder_mask_head_hidden_dim=256,
        decoder_mask_head_depth=3,
        feature_dim=256,
        num_masks=3,
    )


def build_segment_this_thing_b(
    token_size: int,
    num_tokens: int,
) -> SegmentThisThing:
    return build_segment_this_thing(
        token_size=token_size,
        num_tokens=num_tokens,
        encoder_depth=12,
        encoder_embedding_dim=768,
        encoder_num_heads=12,
    )


def build_segment_this_thing_l(
    token_size: int,
    num_tokens: int,
) -> SegmentThisThing:
    return build_segment_this_thing(
        token_size=token_size,
        num_tokens=num_tokens,
        encoder_depth=24,
        encoder_embedding_dim=1024,
        encoder_num_heads=16,
    )


def build_segment_this_thing_h(
    token_size: int,
    num_tokens: int,
) -> SegmentThisThing:
    return build_segment_this_thing(
        token_size=token_size,
        num_tokens=num_tokens,
        encoder_depth=32,
        encoder_embedding_dim=1280,
        encoder_num_heads=16,
    )
