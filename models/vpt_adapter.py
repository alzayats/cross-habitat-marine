"""Visual Prompt Tuning (VPT-Deep) for Vision Transformers.

Implements VPT-Deep from Jia et al. (2022) "Visual Prompt Tuning".
Adds learnable prompt tokens at each transformer layer while keeping
the backbone frozen. Only prompt tokens and classification head are trained.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VPTDeep(nn.Module):
    """Visual Prompt Tuning - Deep variant.

    Adds learnable prompt tokens that are prepended to the input sequence
    at each transformer layer. The backbone remains frozen; only prompt
    tokens are trained.

    Args:
        embed_dim: Embedding dimension of the transformer.
        num_layers: Number of transformer layers.
        num_tokens: Number of prompt tokens per layer.
        dropout: Dropout probability for prompt tokens.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_tokens: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_tokens = num_tokens

        # Learnable prompt tokens for each layer
        # Shape: [num_layers, num_tokens, embed_dim]
        self.prompt_embeddings = nn.Parameter(
            torch.zeros(num_layers, num_tokens, embed_dim)
        )
        nn.init.xavier_uniform_(self.prompt_embeddings.data.view(-1, embed_dim))

        self.prompt_dropout = nn.Dropout(p=dropout)

        total_params = num_layers * num_tokens * embed_dim
        logger.info(
            "VPT-Deep initialized: %d layers x %d tokens x %d dim = %s params",
            num_layers,
            num_tokens,
            embed_dim,
            f"{total_params:,}",
        )

    def get_prompt_tokens(self, layer_idx: int, batch_size: int) -> torch.Tensor:
        """Get prompt tokens for a specific layer.

        Args:
            layer_idx: Index of the transformer layer.
            batch_size: Current batch size.

        Returns:
            Prompt tokens tensor [B, num_tokens, embed_dim].
        """
        prompts = self.prompt_embeddings[layer_idx].unsqueeze(0)
        prompts = prompts.expand(batch_size, -1, -1)
        prompts = self.prompt_dropout(prompts)
        return prompts

    def inject_prompts(
        self, tokens: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Prepend prompt tokens to the input sequence for a given layer.

        The CLS token (position 0) is preserved, and prompt tokens are
        inserted between CLS and patch tokens.

        Args:
            tokens: Input token sequence [B, seq_len, embed_dim].
                    Position 0 is assumed to be the CLS token.
            layer_idx: Index of the current transformer layer.

        Returns:
            Token sequence with prompts: [B, 1 + num_tokens + (seq_len-1), embed_dim].
        """
        batch_size = tokens.shape[0]
        prompt_tokens = self.get_prompt_tokens(layer_idx, batch_size)

        # Separate CLS token and patch tokens
        cls_token = tokens[:, :1, :]       # [B, 1, D]
        patch_tokens = tokens[:, 1:, :]    # [B, N, D]

        # Remove previous prompt tokens if they exist (from previous layer)
        # This is handled by remove_prompts(), so patch_tokens should be clean

        # Concatenate: [CLS] + [prompts] + [patches]
        tokens_with_prompts = torch.cat(
            [cls_token, prompt_tokens, patch_tokens], dim=1
        )

        return tokens_with_prompts

    def remove_prompts(
        self, tokens: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Remove prompt tokens from the sequence after a layer forward pass.

        This ensures the next layer's inject_prompts starts from a clean
        sequence. CLS token and patch tokens are preserved.

        Args:
            tokens: Token sequence with prompts [B, 1 + num_tokens + N, D].
            layer_idx: Index of the current transformer layer.

        Returns:
            Clean token sequence [B, 1 + N, D] without prompt tokens.
        """
        cls_token = tokens[:, :1, :]
        # Skip prompt tokens (positions 1 to 1+num_tokens)
        patch_tokens = tokens[:, 1 + self.num_tokens :, :]

        return torch.cat([cls_token, patch_tokens], dim=1)

    def forward(self, tokens: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Combined inject for a single layer (convenience method).

        Args:
            tokens: Input token sequence.
            layer_idx: Transformer layer index.

        Returns:
            Tokens with prompts injected.
        """
        return self.inject_prompts(tokens, layer_idx)


class VPTShallow(nn.Module):
    """Visual Prompt Tuning - Shallow variant.

    Adds learnable prompt tokens only at the first transformer layer.
    Simpler but less expressive than VPT-Deep.

    Args:
        embed_dim: Embedding dimension.
        num_tokens: Number of prompt tokens.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_tokens: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        self.prompt_embeddings = nn.Parameter(
            torch.zeros(1, num_tokens, embed_dim)
        )
        nn.init.xavier_uniform_(self.prompt_embeddings.data.view(-1, embed_dim))

        self.prompt_dropout = nn.Dropout(p=dropout)

        logger.info(
            "VPT-Shallow initialized: %d tokens x %d dim = %s params",
            num_tokens,
            embed_dim,
            f"{num_tokens * embed_dim:,}",
        )

    def inject_prompts(self, tokens: torch.Tensor) -> torch.Tensor:
        """Inject prompt tokens at the input layer only.

        Args:
            tokens: Input token sequence [B, seq_len, embed_dim].

        Returns:
            Tokens with prompts prepended after CLS: [B, 1 + num_tokens + N, D].
        """
        batch_size = tokens.shape[0]
        prompts = self.prompt_embeddings.expand(batch_size, -1, -1)
        prompts = self.prompt_dropout(prompts)

        cls_token = tokens[:, :1, :]
        patch_tokens = tokens[:, 1:, :]

        return torch.cat([cls_token, prompts, patch_tokens], dim=1)
