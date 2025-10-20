#!/usr/bin/env python3
"""
T5 Text Encoder for MUG-DiT-10B Video Generation.

This module provides T5-based text encoding capabilities for conditioning
video generation. It uses the T5-XXL model from HuggingFace for high-quality
text understanding and embedding generation.

Author: MUG-DiT Team (modified from PixArt)
License: Apache 2.0
"""

import html
import re
from typing import Dict, List, Optional, Union, Any

import torch
from transformers import AutoTokenizer, T5EncoderModel


class T5Embedder:
    """T5-based text embedder for generating text conditioning embeddings.
    
    This class wraps the T5 model to provide text encoding capabilities
    for the video generation pipeline. It handles tokenization, encoding,
    and returns both embeddings and attention masks.
    """
    
    def __init__(
        self,
        device: Union[str, torch.device],
        from_pretrained: Optional[str] = None,
        *,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        t5_model_kwargs: Optional[Dict[str, Any]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        max_model_len: int = 120,
        local_files_only: bool = False,
    ):
        """Initialize the T5 embedder.
        
        Args:
            device: Device to run the model on (e.g., 'cuda', 'cpu')
            from_pretrained: Path or name of the pretrained T5 model
            cache_dir: Directory to cache downloaded models
            hf_token: HuggingFace authentication token
            t5_model_kwargs: Additional kwargs for T5 model initialization
            torch_dtype: PyTorch data type for the model
            max_model_len: Maximum sequence length for tokenization
            local_files_only: Whether to use only local files
            
        Raises:
            ValueError: If from_pretrained is None or invalid
            RuntimeError: If model initialization fails
        """
        if from_pretrained is None:
            raise ValueError("from_pretrained must be specified")
            
        if max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {max_model_len}")
            
        self.device = torch.device(device)
        self.torch_dtype = torch_dtype or torch.bfloat16
        self.cache_dir = cache_dir
        self.max_model_len = max_model_len
        
        # Set default model kwargs if not provided
        if t5_model_kwargs is None:
            t5_model_kwargs = {
                "low_cpu_mem_usage": True,
                "torch_dtype": self.torch_dtype,
                "device_map": {
                    "shared": self.device,
                    "encoder": self.device,
                },
            }
        
        self.hf_token = hf_token
        
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                from_pretrained,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            
            # Initialize T5 encoder model
            self.model = T5EncoderModel.from_pretrained(
                from_pretrained,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                **t5_model_kwargs,
            ).eval()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize T5 model: {e}")

    def get_text_embeddings(
        self, 
        texts: Union[str, List[str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate text embeddings from input texts.
        
        Args:
            texts: Input text(s) to encode. Can be a single string or list of strings.
            
        Returns:
            Tuple containing:
                - text_encoder_embs: Text embeddings tensor of shape [batch, seq_len, hidden_dim]
                - attention_mask: Attention mask tensor of shape [batch, seq_len]
                
        Raises:
            TypeError: If texts is not string or list of strings
            RuntimeError: If text encoding fails
        """
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("texts must be a string or list of strings")
            
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("texts cannot be empty or contain only whitespace")
            
        try:
            # Tokenize the input texts
            text_tokens_and_mask = self.tokenizer(
                texts,
                max_length=self.max_model_len,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            
            # Move tensors to the correct device
            input_ids = text_tokens_and_mask["input_ids"].to(self.device)
            attention_mask = text_tokens_and_mask["attention_mask"].to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                text_encoder_embs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )["last_hidden_state"].detach()
                
            return text_encoder_embs, attention_mask
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate text embeddings: {e}")


class T5Encoder:
    """High-level T5 text encoder interface for video generation.
    
    This class provides a convenient interface for text encoding with
    the T5 model, including proper device management and output formatting
    for the video generation pipeline.
    """
    
    def __init__(
        self,
        from_pretrained: Optional[str] = None,
        max_model_len: int = 120,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float,
        cache_dir: Optional[str] = None,
        shardformer: bool = False,  # Currently unused, kept for compatibility
        local_files_only: bool = False,
    ):
        """Initialize the T5 encoder.
        
        Args:
            from_pretrained: Path or name of the pretrained T5 model
            max_model_len: Maximum sequence length for tokenization
            device: Device to run the model on
            dtype: Data type for the model
            cache_dir: Directory to cache downloaded models
            shardformer: Whether to use shardformer (currently unused)
            local_files_only: Whether to use only local files
            
        Raises:
            ValueError: If from_pretrained is None
        """
        if from_pretrained is None:
            raise ValueError("Please specify the path to the T5 model")
            
        if max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {max_model_len}")
        
        self.max_model_len = max_model_len
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Initialize the T5 embedder
        self.t5 = T5Embedder(
            device=device,
            torch_dtype=dtype,
            from_pretrained=from_pretrained,
            cache_dir=cache_dir,
            max_model_len=max_model_len,
            local_files_only=local_files_only,
        )
        
        # Ensure model is on correct device and dtype
        self.t5.model.to(dtype=dtype)
        
        # Store output dimension for compatibility with other components
        self.output_dim = self.t5.model.config.d_model

    def encode(self, text: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """Encode text into conditioning embeddings for video generation.
        
        Args:
            text: Input text(s) to encode
            
        Returns:
            Dictionary containing:
                - 'y': Text embeddings tensor of shape [batch, 1, seq_len, hidden_dim]
                - 'mask': Attention mask tensor of shape [batch, seq_len]
                
        Raises:
            TypeError: If text is not string or list of strings
            RuntimeError: If text encoding fails
        """
        try:
            # Generate embeddings using the T5 embedder
            caption_embs, emb_masks = self.t5.get_text_embeddings(text)
            
            # Add extra dimension for compatibility with video generation pipeline
            # Shape: [batch, seq_len, hidden_dim] -> [batch, 1, seq_len, hidden_dim]
            caption_embs = caption_embs[:, None]
            
            return {
                "y": caption_embs,
                "mask": emb_masks
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to encode text: {e}")
