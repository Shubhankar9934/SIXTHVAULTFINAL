"""
Memory Management Utilities for SIXTHVAULT Backend

This module provides memory monitoring, cleanup, and optimization utilities
to prevent memory leaks and crashes in the RAG backend system.
"""

import gc
import psutil
import logging
import asyncio
import time
from typing import Dict, Optional, Callable, Any
from functools import wraps
from contextlib import asynccontextmanager
import threading
import weakref

logger = logging.getLogger(__name__)

class MemoryManager:
    """Advanced memory management for the RAG backend"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.memory_threshold_mb = 1500  # 1.5GB threshold (reduced for stability)
        self.cleanup_callbacks = []
        self.monitoring_enabled = True
        self._last_cleanup = time.time()
        self._cleanup_interval = 120  # 2 minutes (more aggressive cleanup)
        self._critical_threshold_mb = 2000  # Critical threshold for emergency cleanup
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            return {
                "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
                "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                "percent": memory_percent,
                "available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "total_mb": psutil.virtual_memory().total / 1024 / 1024
            }
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {"error": str(e)}
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is critical"""
        try:
            memory_info = self.get_memory_usage()
            if "error" in memory_info:
                return False
                
            return (memory_info["rss_mb"] > self.memory_threshold_mb or 
                    memory_info["percent"] > 60)  # Reduced to 60% for earlier intervention
        except Exception:
            return False
    
    def is_memory_emergency(self) -> bool:
        """Check if memory usage requires emergency intervention"""
        try:
            memory_info = self.get_memory_usage()
            if "error" in memory_info:
                return False
                
            return (memory_info["rss_mb"] > self._critical_threshold_mb or 
                    memory_info["percent"] > 80)  # Emergency threshold
        except Exception:
            return False
    
    def register_cleanup_callback(self, callback: Callable[[], None]):
        """Register a cleanup callback function"""
        self.cleanup_callbacks.append(weakref.ref(callback))
    
    def force_cleanup(self, emergency: bool = False):
        """Force immediate memory cleanup"""
        cleanup_type = "EMERGENCY" if emergency else "STANDARD"
        logger.info(f"Forcing {cleanup_type} memory cleanup...")
        
        # Run registered cleanup callbacks
        active_callbacks = []
        for callback_ref in self.cleanup_callbacks:
            callback = callback_ref()
            if callback is not None:
                try:
                    callback()
                    active_callbacks.append(callback_ref)
                except Exception as e:
                    logger.error(f"Cleanup callback failed: {e}")
            
        self.cleanup_callbacks = active_callbacks
        
        # Multiple rounds of garbage collection for emergency cleanup
        total_collected = 0
        rounds = 3 if emergency else 1
        
        for i in range(rounds):
            collected = gc.collect()
            total_collected += collected
            if emergency and i < rounds - 1:
                time.sleep(0.1)  # Brief pause between rounds
        
        logger.info(f"Garbage collection freed {total_collected} objects in {rounds} rounds")
        
        # Emergency: Clear all generation caches
        if emergency:
            try:
                # Force collection of all generations
                for generation in range(3):
                    gc.collect(generation)
                logger.info("Emergency: Cleared all GC generations")
            except Exception as e:
                logger.error(f"Emergency GC failed: {e}")
        
        # Update last cleanup time
        self._last_cleanup = time.time()
        
        # Log memory status after cleanup
        memory_info = self.get_memory_usage()
        logger.info(f"Memory after {cleanup_type} cleanup: {memory_info['rss_mb']:.1f}MB ({memory_info['percent']:.1f}%)")
    
    def should_cleanup(self) -> bool:
        """Check if cleanup should be performed"""
        time_since_cleanup = time.time() - self._last_cleanup
        return (self.is_memory_critical() or 
                time_since_cleanup > self._cleanup_interval)
    
    def auto_cleanup_if_needed(self):
        """Automatically cleanup if needed"""
        if not self.monitoring_enabled:
            return
            
        if self.is_memory_emergency():
            logger.warning("EMERGENCY MEMORY CLEANUP TRIGGERED")
            self.force_cleanup(emergency=True)
        elif self.should_cleanup():
            self.force_cleanup(emergency=False)

# Global memory manager instance
memory_manager = MemoryManager()

def memory_monitor(threshold_mb: Optional[float] = None):
    """Decorator to monitor memory usage of functions"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Check memory before execution
            memory_before = memory_manager.get_memory_usage()
            
            try:
                # Auto cleanup if needed
                memory_manager.auto_cleanup_if_needed()
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Check memory after execution
                memory_after = memory_manager.get_memory_usage()
                memory_diff = memory_after["rss_mb"] - memory_before["rss_mb"]
                
                # Log if significant memory increase
                if memory_diff > 100:  # 100MB increase
                    logger.warning(f"{func.__name__} increased memory by {memory_diff:.1f}MB")
                
                # Force cleanup if threshold exceeded
                if threshold_mb and memory_after["rss_mb"] > threshold_mb:
                    logger.warning(f"Memory threshold exceeded after {func.__name__}: {memory_after['rss_mb']:.1f}MB")
                    memory_manager.force_cleanup()
                
                return result
                
            except Exception as e:
                # Cleanup on error
                memory_manager.auto_cleanup_if_needed()
                raise e
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Check memory before execution
            memory_before = memory_manager.get_memory_usage()
            
            try:
                # Auto cleanup if needed
                memory_manager.auto_cleanup_if_needed()
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Check memory after execution
                memory_after = memory_manager.get_memory_usage()
                memory_diff = memory_after["rss_mb"] - memory_before["rss_mb"]
                
                # Log if significant memory increase
                if memory_diff > 100:  # 100MB increase
                    logger.warning(f"{func.__name__} increased memory by {memory_diff:.1f}MB")
                
                # Force cleanup if threshold exceeded
                if threshold_mb and memory_after["rss_mb"] > threshold_mb:
                    logger.warning(f"Memory threshold exceeded after {func.__name__}: {memory_after['rss_mb']:.1f}MB")
                    memory_manager.force_cleanup()
                
                return result
                
            except Exception as e:
                # Cleanup on error
                memory_manager.auto_cleanup_if_needed()
                raise e
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

@asynccontextmanager
async def memory_context(cleanup_after: bool = True):
    """Context manager for memory-aware operations"""
    memory_before = memory_manager.get_memory_usage()
    logger.debug(f"Memory context started: {memory_before['rss_mb']:.1f}MB")
    
    try:
        yield memory_manager
    finally:
        if cleanup_after:
            memory_manager.auto_cleanup_if_needed()
        
        memory_after = memory_manager.get_memory_usage()
        memory_diff = memory_after["rss_mb"] - memory_before["rss_mb"]
        logger.debug(f"Memory context ended: {memory_after['rss_mb']:.1f}MB (diff: {memory_diff:+.1f}MB)")

class MemoryOptimizedCache:
    """Memory-optimized cache with automatic cleanup"""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 500):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self._lock = threading.Lock()
        
        # Register cleanup callback
        memory_manager.register_cleanup_callback(self.cleanup_old_entries)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            if key in self.cache:
                self.access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any):
        """Set item in cache with memory management"""
        with self._lock:
            # Check if we need to cleanup first
            if len(self.cache) >= self.max_size:
                self._cleanup_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            
            # Check memory usage
            if self._estimate_memory_usage() > self.max_memory_mb:
                self._cleanup_lru()
    
    def _estimate_memory_usage(self) -> float:
        """Estimate cache memory usage in MB"""
        try:
            import sys
            total_size = 0
            for key, value in self.cache.items():
                total_size += sys.getsizeof(key) + sys.getsizeof(value)
            return total_size / 1024 / 1024
        except Exception:
            return 0
    
    def _cleanup_lru(self):
        """Cleanup least recently used items"""
        if not self.cache:
            return
        
        # Sort by access time and remove oldest 25%
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        items_to_remove = len(sorted_items) // 4
        
        for key, _ in sorted_items[:items_to_remove]:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)
        
        logger.debug(f"Cache cleanup: removed {items_to_remove} items")
    
    def cleanup_old_entries(self):
        """Cleanup old cache entries (called by memory manager)"""
        with self._lock:
            current_time = time.time()
            old_keys = []
            
            # Remove entries older than 1 hour
            for key, access_time in self.access_times.items():
                if current_time - access_time > 3600:  # 1 hour
                    old_keys.append(key)
            
            for key in old_keys:
                self.cache.pop(key, None)
                self.access_times.pop(key, None)
            
            if old_keys:
                logger.debug(f"Cache cleanup: removed {len(old_keys)} old entries")
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "estimated_memory_mb": self._estimate_memory_usage(),
                "max_memory_mb": self.max_memory_mb
            }

def optimize_for_large_documents():
    """Apply optimizations for large document processing"""
    # Reduce garbage collection threshold for more frequent cleanup
    import gc
    gc.set_threshold(700, 10, 10)  # More aggressive GC
    
    # Set memory manager to more aggressive mode
    memory_manager.memory_threshold_mb = 2000  # 2GB threshold
    memory_manager._cleanup_interval = 180  # 3 minutes
    
    logger.info("Applied large document memory optimizations")

def get_memory_report() -> Dict[str, Any]:
    """Get comprehensive memory report"""
    memory_info = memory_manager.get_memory_usage()
    
    return {
        "current_memory": memory_info,
        "is_critical": memory_manager.is_memory_critical(),
        "threshold_mb": memory_manager.memory_threshold_mb,
        "monitoring_enabled": memory_manager.monitoring_enabled,
        "cleanup_callbacks": len(memory_manager.cleanup_callbacks),
        "gc_stats": {
            "counts": gc.get_count(),
            "threshold": gc.get_threshold()
        }
    }

# Background memory monitoring task
async def start_memory_monitoring():
    """Start background memory monitoring"""
    logger.info("Starting background memory monitoring")
    
    while memory_manager.monitoring_enabled:
        try:
            memory_info = memory_manager.get_memory_usage()
            
            # Log memory status every 10 minutes
            if int(time.time()) % 600 == 0:
                logger.info(f"Memory status: {memory_info['rss_mb']:.1f}MB ({memory_info['percent']:.1f}%)")
            
            # Auto cleanup if needed
            memory_manager.auto_cleanup_if_needed()
            
            # Check for critical memory usage
            if memory_manager.is_memory_critical():
                logger.warning(f"Critical memory usage detected: {memory_info['rss_mb']:.1f}MB")
                memory_manager.force_cleanup()
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"Memory monitoring error: {e}")
            await asyncio.sleep(60)  # Wait longer on error

# Utility functions for specific cleanup tasks
def cleanup_ml_models():
    """Cleanup ML model caches"""
    try:
        # Clear sklearn caches if available
        from sklearn.utils._testing import clear_cache
        clear_cache()
        logger.debug("Cleared sklearn caches")
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"Failed to clear ML model caches: {e}")

def cleanup_text_processing():
    """Cleanup text processing caches"""
    try:
        # Clear any text processing caches
        import re
        re.purge()  # Clear regex cache
        logger.debug("Cleared text processing caches")
    except Exception as e:
        logger.error(f"Failed to clear text processing caches: {e}")

# Register default cleanup callbacks
memory_manager.register_cleanup_callback(cleanup_ml_models)
memory_manager.register_cleanup_callback(cleanup_text_processing)
