"""
Rate limiting utilities for Plura Backend.

Prevents abuse by limiting request frequency per client.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, List, Optional


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    # Request limits
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    
    # Burst allowance
    burst_size: int = 10
    
    # Cleanup interval
    cleanup_interval_seconds: float = 60.0


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after_seconds: float = 60.0):
        self.message = message
        self.retry_after = retry_after_seconds
        super().__init__(message)


class RateLimiter:
    """
    Token bucket rate limiter with per-client tracking.
    
    Supports:
    - Per-minute and per-hour limits
    - Burst allowance
    - Automatic cleanup of stale entries
    
    Example:
        >>> limiter = RateLimiter()
        >>> limiter.check("client_123")  # Returns True if allowed
        >>> limiter.check("client_123")  # Track request
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._buckets: Dict[str, _TokenBucket] = {}
        self._lock = Lock()
        self._last_cleanup = time.time()
    
    def check(self, client_id: str, *, cost: int = 1) -> bool:
        """
        Check if request is allowed and consume tokens.
        
        Args:
            client_id: Unique identifier for the client
            cost: Number of tokens to consume (default 1)
            
        Returns:
            True if request is allowed
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        with self._lock:
            self._maybe_cleanup()
            
            bucket = self._get_or_create_bucket(client_id)
            
            if not bucket.consume(cost):
                retry_after = bucket.time_until_available(cost)
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {client_id}",
                    retry_after_seconds=retry_after
                )
            
            return True
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for a client."""
        with self._lock:
            bucket = self._buckets.get(client_id)
            if bucket is None:
                return self.config.requests_per_minute
            return int(bucket.tokens)
    
    def reset(self, client_id: str) -> None:
        """Reset rate limit for a client."""
        with self._lock:
            if client_id in self._buckets:
                del self._buckets[client_id]
    
    def _get_or_create_bucket(self, client_id: str) -> "_TokenBucket":
        """Get or create token bucket for client."""
        if client_id not in self._buckets:
            self._buckets[client_id] = _TokenBucket(
                capacity=self.config.requests_per_minute,
                refill_rate=self.config.requests_per_minute / 60.0,
                burst=self.config.burst_size,
            )
        return self._buckets[client_id]
    
    def _maybe_cleanup(self) -> None:
        """Remove stale entries periodically."""
        now = time.time()
        if now - self._last_cleanup < self.config.cleanup_interval_seconds:
            return
        
        self._last_cleanup = now
        stale_threshold = now - 3600  # Remove entries older than 1 hour
        
        stale_keys = [
            k for k, v in self._buckets.items()
            if v.last_update < stale_threshold
        ]
        
        for k in stale_keys:
            del self._buckets[k]


class _TokenBucket:
    """Token bucket implementation for rate limiting."""
    
    def __init__(self, *, capacity: int, refill_rate: float, burst: int = 0):
        self.capacity = capacity + burst
        self.refill_rate = refill_rate
        self.tokens = float(capacity + burst)
        self.last_update = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Consume tokens, returning True if successful."""
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate seconds until requested tokens are available."""
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        needed = tokens - self.tokens
        return needed / self.refill_rate
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        self.last_update = now
        
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate
        )


def rate_limit(
    *,
    requests_per_minute: int = 60,
    key_func: Optional[Callable[[Any], str]] = None,
) -> Callable:
    """
    Decorator for rate limiting function calls.
    
    Args:
        requests_per_minute: Maximum requests per minute
        key_func: Function to extract client ID from first argument
        
    Example:
        >>> @rate_limit(requests_per_minute=10)
        ... def process_request(client_id: str, data: bytes):
        ...     ...
    """
    limiter = RateLimiter(RateLimitConfig(requests_per_minute=requests_per_minute))
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if key_func and args:
                client_id = key_func(args[0])
            elif args:
                client_id = str(args[0])
            else:
                client_id = "default"
            
            limiter.check(client_id)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator
