"""
Instruction Encoder for OcuMamba-Lite.

Encodes natural language instructions into embeddings for conditioning
the visual encoder. Uses a lightweight transformer encoder.

For production, can be initialized from T5-small or similar pretrained model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InstructionEncoderConfig:
    """Configuration for instruction encoder."""
    
    vocab_size: int = 32000  # Standard vocab size
    max_length: int = 64     # Max instruction tokens
    hidden_dim: int = 384    # Match visual encoder
    num_layers: int = 4      # Lightweight
    num_heads: int = 6       # Attention heads
    ff_dim: int = 1536       # Feed-forward dimension
    dropout: float = 0.1


class InstructionEncoder(nn.Module):
    """
    Lightweight instruction encoder for GUI grounding.
    
    Encodes natural language instructions like "click the save button"
    into embeddings that can condition the visual encoder.
    """
    
    def __init__(self, config: Optional[InstructionEncoderConfig] = None):
        super().__init__()
        self.config = config or InstructionEncoderConfig()
        
        # Token embedding
        self.token_embed = nn.Embedding(
            self.config.vocab_size, 
            self.config.hidden_dim
        )
        
        # Position embedding
        self.pos_embed = nn.Embedding(
            self.config.max_length, 
            self.config.hidden_dim
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.ff_dim,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
        )
        
        # Output projection (for different uses)
        self.pooler = nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        self.norm = nn.LayerNorm(self.config.hidden_dim)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode instruction tokens.
        
        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) mask for padding
            
        Returns:
            Dict with:
                - last_hidden_state: (B, L, D) token embeddings
                - pooled_output: (B, D) sentence embedding
        """
        B, L = input_ids.shape
        
        # Token + position embeddings
        pos_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(pos_ids)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (True = masked)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Encode
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        
        # Pool: take first token or mean pool
        if attention_mask is not None:
            # Mean pooling with mask
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)
        
        pooled = self.pooler(pooled)
        pooled = torch.tanh(pooled)
        
        return {
            "last_hidden_state": x,
            "pooled_output": pooled,
        }
    
    @property
    def hidden_dim(self) -> int:
        return self.config.hidden_dim
    
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SimpleTokenizer:
    """
    Simple character/word tokenizer for prototyping.
    For production, use pretrained tokenizer (T5, BERT, etc.)
    """
    
    def __init__(self, vocab_size: int = 32000, max_length: int = 64):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Special tokens
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Simple word vocab (built on-the-fly)
        self._word_to_id = {
            "<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3,
        }
        self._id_to_word = {v: k for k, v in self._word_to_id.items()}
        self._next_id = 4
    
    def _get_word_id(self, word: str) -> int:
        """Get or create word ID."""
        word = word.lower()
        if word not in self._word_to_id:
            if self._next_id < self.vocab_size:
                self._word_to_id[word] = self._next_id
                self._id_to_word[self._next_id] = word
                self._next_id += 1
            else:
                return self.unk_token_id
        return self._word_to_id[word]
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        words = text.lower().split()
        ids = [self.bos_token_id]
        for word in words[:self.max_length - 2]:
            ids.append(self._get_word_id(word))
        ids.append(self.eos_token_id)
        return ids
    
    def __call__(
        self,
        texts: List[str],
        padding: bool = True,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize batch of texts."""
        batch_ids = [self.encode(t) for t in texts]
        
        if padding:
            max_len = max(len(ids) for ids in batch_ids)
            max_len = min(max_len, self.max_length)
            
            padded = []
            masks = []
            for ids in batch_ids:
                if len(ids) < max_len:
                    pad_len = max_len - len(ids)
                    padded.append(ids + [self.pad_token_id] * pad_len)
                    masks.append([1] * len(ids) + [0] * pad_len)
                else:
                    padded.append(ids[:max_len])
                    masks.append([1] * max_len)
            
            result = {
                "input_ids": padded,
                "attention_mask": masks,
            }
        else:
            result = {"input_ids": batch_ids}
        
        if return_tensors == "pt":
            result = {k: torch.tensor(v) for k, v in result.items()}
        
        return result


def create_instruction_encoder(
    size: str = "small",
    **kwargs,
) -> Tuple[InstructionEncoder, SimpleTokenizer]:
    """
    Create instruction encoder with tokenizer.
    
    Args:
        size: "tiny", "small", or "base"
        
    Returns:
        (encoder, tokenizer) tuple
    """
    configs = {
        "tiny": InstructionEncoderConfig(hidden_dim=192, num_layers=2, num_heads=3),
        "small": InstructionEncoderConfig(hidden_dim=384, num_layers=4, num_heads=6),
        "base": InstructionEncoderConfig(hidden_dim=512, num_layers=6, num_heads=8),
    }
    
    config = configs.get(size, configs["small"])
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    encoder = InstructionEncoder(config)
    tokenizer = SimpleTokenizer(
        vocab_size=config.vocab_size,
        max_length=config.max_length,
    )
    
    return encoder, tokenizer
