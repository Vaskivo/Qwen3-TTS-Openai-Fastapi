# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Configuration module for TTS backend settings.

This module centralizes all configuration options for the TTS API,
including backend selection and device settings.
"""

import os

# ============================================================================
# Backend Selection
# ============================================================================

TTS_BACKEND = os.getenv("TTS_BACKEND", "official")
"""
TTS backend to use.
Options: 'official', 'optimized'
- 'official': Official Qwen3-TTS implementation (default, GPU/CPU auto-detect)
- 'optimized': GPU production backend with model switching, native PCM
  streaming, and the voice library
"""

TTS_MODEL_ID = os.getenv("TTS_MODEL_ID", os.getenv("TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"))
"""
Model identifier for HuggingFace.
Examples: 
- Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (default, voice design)
- Qwen/Qwen3-TTS-12Hz-1.7B-Base (voice cloning support)
- Qwen/Qwen3-TTS-12Hz-0.6B-Base (smaller model for CPU)
"""

# ============================================================================
# Device and Precision Settings
# ============================================================================

TTS_DEVICE = os.getenv("TTS_DEVICE", "auto")
"""
Device to run inference on.
Options: 'auto', 'cpu', 'cuda', 'cuda:0', etc.
- 'auto': Automatically detect (GPU if available, otherwise CPU)
- 'cpu': Force CPU inference
- 'cuda' or 'cuda:0': Use specific GPU
"""

TTS_DTYPE = os.getenv("TTS_DTYPE", "auto")
"""
Data type for model weights and computation.
Options: 'auto', 'float32', 'float16', 'bfloat16'
- 'auto': bfloat16 on GPU, float32 on CPU
- 'float32': Full precision (recommended for CPU, slower but stable)
- 'float16': Half precision (GPU only, may cause issues on some models)
- 'bfloat16': BFloat16 precision (Ampere+ GPUs, best for GPU)
"""

TTS_ATTN = os.getenv("TTS_ATTN", "auto")
"""
Attention implementation to use.
Options: 'auto', 'flash_attention_2', 'sdpa', 'eager'
- 'auto': Try flash_attention_2, fall back to sdpa, then eager
- 'flash_attention_2': Flash Attention 2 (fastest, Ampere+ GPUs)
- 'sdpa': Scaled Dot Product Attention (PyTorch native, good for CPU/GPU)
- 'eager': Standard attention (slowest, most compatible, good for CPU)
"""

# ============================================================================
# Warmup and Optimization Settings
# ============================================================================

TTS_WARMUP_ON_START = os.getenv("TTS_WARMUP_ON_START", "false").lower() == "true"
"""
Whether to run a warmup inference on server startup.
Recommended: true for production to initialize torch.compile() and cuDNN.
"""
