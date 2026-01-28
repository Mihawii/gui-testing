"""
Input validation utilities for Plura Backend.

Validates images, instructions, and other user inputs to prevent:
- DoS via oversized uploads
- Malformed image attacks
- Injection attacks
- Path traversal
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import numpy as np


@dataclass
class ValidationConfig:
    """Configuration for input validation."""
    
    # Image limits
    max_image_bytes: int = 20 * 1024 * 1024  # 20MB
    max_dimension: int = 8192  # 8K max
    min_dimension: int = 32  # Minimum viable size
    allowed_formats: Set[str] = None  # Set in __post_init__
    
    # Instruction limits
    max_instruction_length: int = 2000
    min_instruction_length: int = 1
    
    # Cache key limits
    max_cache_key_length: int = 128
    
    def __post_init__(self):
        if self.allowed_formats is None:
            self.allowed_formats = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}


# Default configuration
DEFAULT_CONFIG = ValidationConfig()


class ValidationError(Exception):
    """Raised when validation fails."""
    
    def __init__(self, message: str, field: str = "unknown"):
        self.message = message
        self.field = field
        super().__init__(f"{field}: {message}")


class ImageValidator:
    """
    Validates image uploads for safety and compatibility.
    
    Checks:
    - File size limits
    - Image dimensions
    - Format/magic bytes
    - Malformed image detection
    """
    
    # Magic bytes for common image formats
    MAGIC_BYTES = {
        b'\x89PNG\r\n\x1a\n': 'png',
        b'\xff\xd8\xff': 'jpg',
        b'GIF87a': 'gif',
        b'GIF89a': 'gif',
        b'RIFF': 'webp',  # WebP starts with RIFF
        b'BM': 'bmp',
    }
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or DEFAULT_CONFIG
    
    def validate(self, image_bytes: bytes) -> Tuple[str, int, int]:
        """
        Validate image bytes and return format and dimensions.
        
        Args:
            image_bytes: Raw image data
            
        Returns:
            Tuple of (format, width, height)
            
        Raises:
            ValidationError: If validation fails
        """
        if not image_bytes:
            raise ValidationError("Image data is empty", "image")
        
        # Check file size
        size = len(image_bytes)
        if size > self.config.max_image_bytes:
            max_mb = self.config.max_image_bytes / (1024 * 1024)
            raise ValidationError(
                f"Image exceeds maximum size of {max_mb:.1f}MB",
                "image"
            )
        
        # Check magic bytes
        detected_format = self._detect_format(image_bytes)
        if detected_format not in self.config.allowed_formats:
            raise ValidationError(
                f"Image format '{detected_format}' not allowed. "
                f"Allowed: {', '.join(sorted(self.config.allowed_formats))}",
                "image"
            )
        
        # Try to decode and check dimensions
        width, height = self._get_dimensions(image_bytes, detected_format)
        
        if width < self.config.min_dimension or height < self.config.min_dimension:
            raise ValidationError(
                f"Image too small ({width}x{height}). "
                f"Minimum: {self.config.min_dimension}x{self.config.min_dimension}",
                "image"
            )
        
        if width > self.config.max_dimension or height > self.config.max_dimension:
            raise ValidationError(
                f"Image too large ({width}x{height}). "
                f"Maximum: {self.config.max_dimension}x{self.config.max_dimension}",
                "image"
            )
        
        return detected_format, width, height
    
    def _detect_format(self, data: bytes) -> str:
        """Detect image format from magic bytes."""
        for magic, fmt in self.MAGIC_BYTES.items():
            if data.startswith(magic):
                return fmt
        
        # Check for WebP more carefully (RIFF....WEBP)
        if len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            return 'webp'
        
        # Unknown format
        return 'unknown'
    
    def _get_dimensions(self, data: bytes, fmt: str) -> Tuple[int, int]:
        """Extract image dimensions without full decode if possible."""
        try:
            # Try using PIL for reliable dimension extraction
            from PIL import Image
            import io
            
            with Image.open(io.BytesIO(data)) as img:
                return img.size  # (width, height)
        except ImportError:
            pass
        except Exception:
            pass
        
        try:
            # Fallback to cv2
            import cv2
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValidationError("Failed to decode image", "image")
            h, w = img.shape[:2]
            return w, h
        except ImportError:
            pass
        except Exception:
            pass
        
        # Can't determine dimensions - allow but log warning
        return 1920, 1080  # Default assumption


class RequestValidator:
    """Validates API request parameters."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or DEFAULT_CONFIG
    
    def validate_instruction(self, instruction: str) -> str:
        """
        Validate and sanitize instruction text.
        
        Args:
            instruction: User-provided instruction text
            
        Returns:
            Sanitized instruction
            
        Raises:
            ValidationError: If validation fails
        """
        if not instruction:
            raise ValidationError("Instruction cannot be empty", "instruction")
        
        # Normalize whitespace
        instruction = " ".join(instruction.split())
        
        if len(instruction) < self.config.min_instruction_length:
            raise ValidationError(
                f"Instruction too short (min {self.config.min_instruction_length} chars)",
                "instruction"
            )
        
        if len(instruction) > self.config.max_instruction_length:
            raise ValidationError(
                f"Instruction too long (max {self.config.max_instruction_length} chars)",
                "instruction"
            )
        
        # Remove potential injection characters
        instruction = self._sanitize_text(instruction)
        
        return instruction
    
    def _sanitize_text(self, text: str) -> str:
        """Remove potentially dangerous characters."""
        # Remove null bytes and control characters (except newlines/tabs)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text


def validate_image_bytes(image_bytes: bytes) -> Tuple[str, int, int]:
    """
    Convenience function to validate image bytes.
    
    Returns:
        Tuple of (format, width, height)
    """
    validator = ImageValidator()
    return validator.validate(image_bytes)


def validate_instruction(instruction: str) -> str:
    """
    Convenience function to validate instruction.
    
    Returns:
        Sanitized instruction
    """
    validator = RequestValidator()
    return validator.validate_instruction(instruction)


def sanitize_cache_key(key: str) -> str:
    """
    Sanitize a cache key to prevent path traversal.
    
    Args:
        key: Raw cache key
        
    Returns:
        Safe cache key (alphanumeric + limited punctuation)
    """
    if not key:
        return hashlib.md5(b"empty").hexdigest()
    
    # Only allow alphanumeric and underscores
    safe = re.sub(r'[^a-zA-Z0-9_-]', '', str(key))
    
    # Limit length
    if len(safe) > 128:
        safe = safe[:128]
    
    if not safe:
        return hashlib.md5(key.encode('utf-8', errors='ignore')).hexdigest()
    
    return safe
