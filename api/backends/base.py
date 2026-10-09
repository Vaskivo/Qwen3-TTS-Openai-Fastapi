# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Base class for TTS backends.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


def flush_cuda_memory() -> None:
    """Break dead reference cycles and return freed GPU blocks to the driver.

    nn.Module trees hold reference cycles (parent↔child via ``_modules``,
    hooks, closures), so refcount-only deletion often leaves the weights
    live even after every *named* reference is gone. ``gc.collect()`` breaks
    those cycles and releases the tensors; ``torch.cuda.empty_cache()``
    then returns the now-truly-free blocks to the driver — without it they
    would linger in the allocator's reserved pool and keep showing as used
    in nvidia-smi.

    IMPORTANT: call this only AFTER dropping *every* reference to the model
    (instance attributes, dict entries, **and local variables in the
    calling frames** — ``del`` inside a helper cannot delete the caller's
    name). If any reference survives, ``gc.collect()`` cannot free the
    model, the empty cache flush is a no-op, and the weights stay resident
    in VRAM even though the code logged a successful unload.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class TTSBackend(ABC):
    """Abstract base class for TTS backends."""

    def __init__(self):
        """Initialize the backend."""
        self.model = None
        self.device = None
        self.dtype = None
        self._custom_voices: Dict[str, Any] = {}
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the backend and load the model.
        
        This method should:
        - Load the model
        - Set up device and dtype
        - Perform any necessary warmup
        """
        pass
    
    @abstractmethod
    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text.
        
        Args:
            text: The text to synthesize
            voice: Voice name/identifier to use
            language: Language code (e.g., "English", "Chinese", "Auto")
            instruct: Optional instruction for voice style/emotion
            speed: Speech speed multiplier (0.25 to 4.0)
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        pass
    
    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the name of this backend."""
        pass
    
    @abstractmethod
    def get_model_id(self) -> str:
        """Return the model identifier."""
        pass
    
    @abstractmethod
    def get_supported_voices(self) -> List[str]:
        """Return list of supported voice names."""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Return whether the backend is initialized and ready."""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """
        Return device information.

        Returns:
            Dict with keys: device, gpu_available, gpu_name, vram_total, vram_used
        """
        pass

    def unload(self) -> bool:
        """Release the model and any cached GPU resources (VRAM).

        Called by the idle-unload watchdog or the manual unload endpoint
        when the server has been idle. Must be safe to call when the backend
        is already unloaded, and must make ``is_ready()`` return False so
        the next request re-initializes it (lazy load).

        This is a synchronous (atomic w.r.t. the event loop) operation.

        Returns:
            True if a loaded model was released, False if there was nothing
            to unload.
        """
        logger.warning("This backend does not support unloading")
        return False

    def supports_voice_cloning(self) -> bool:
        """
        Return whether the backend supports voice cloning.

        Voice cloning requires the Base model (Qwen3-TTS-12Hz-1.7B-Base).
        The CustomVoice model does not support voice cloning.

        Returns:
            True if voice cloning is supported, False otherwise
        """
        return False

    async def generate_voice_clone(
        self,
        text: str,
        ref_audio: np.ndarray,
        ref_audio_sr: int,
        ref_text: Optional[str] = None,
        language: str = "Auto",
        x_vector_only_mode: bool = False,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech by cloning a voice from reference audio.

        Args:
            text: The text to synthesize
            ref_audio: Reference audio as numpy array
            ref_audio_sr: Sample rate of reference audio
            ref_text: Transcript of reference audio (required for ICL mode)
            language: Language code (e.g., "English", "Chinese", "Auto")
            x_vector_only_mode: If True, use x-vector only (no ref_text needed)
            speed: Speech speed multiplier (0.25 to 4.0)

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            NotImplementedError: If voice cloning is not supported by this backend
        """
        raise NotImplementedError("Voice cloning is not supported by this backend")

    async def load_custom_voices(self, custom_voices_dir: str) -> None:
        """
        Load custom voices from a directory.

        Each subdirectory should contain a reference audio file and optional
        reference.txt for ICL mode. Override in subclasses to implement.

        Args:
            custom_voices_dir: Path to the custom voices directory
        """
        logger.info("Custom voice loading is not supported by this backend")

    def get_custom_voice_names(self) -> List[str]:
        """Return list of loaded custom voice names."""
        return list(self._custom_voices.keys())

    def is_custom_voice(self, voice_name: str) -> bool:
        """Check if a voice name is a custom voice."""
        return voice_name in self._custom_voices

    async def generate_speech_with_custom_voice(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech using a custom cloned voice.

        Args:
            text: The text to synthesize
            voice: Custom voice name
            language: Language code
            speed: Speech speed multiplier

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            NotImplementedError: If not supported by this backend
        """
        raise NotImplementedError("Custom voice generation is not supported by this backend")
