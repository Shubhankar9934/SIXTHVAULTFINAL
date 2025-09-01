"""
Lazy Model Loading Utilities for Memory Optimization

This module provides lazy loading capabilities for ML models to reduce
memory usage and improve startup times.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable
from functools import wraps
import threading
import weakref

logger = logging.getLogger(__name__)

class LazyModelLoader:
    """Lazy loader for ML models with memory management"""
    
    def __init__(self, model_name: str, loader_func: Callable, cleanup_func: Optional[Callable] = None):
        self.model_name = model_name
        self.loader_func = loader_func
        self.cleanup_func = cleanup_func
        self.model = None
        self.loading = False
        self.load_time = None
        self.last_used = None
        self.usage_count = 0
        self._lock = threading.Lock()
        
        # Register for memory cleanup
        from app.utils.memory_manager import memory_manager
        memory_manager.register_cleanup_callback(self.cleanup_if_unused)
    
    async def get_model(self):
        """Get model, loading it if necessary"""
        with self._lock:
            if self.model is not None:
                self.last_used = time.time()
                self.usage_count += 1
                return self.model
            
            if self.loading:
                # Wait for loading to complete
                while self.loading:
                    await asyncio.sleep(0.1)
                if self.model is not None:
                    self.last_used = time.time()
                    self.usage_count += 1
                    return self.model
        
        # Load the model
        return await self._load_model()
    
    async def _load_model(self):
        """Load the model with proper error handling"""
        with self._lock:
            if self.model is not None:
                self.last_used = time.time()
                self.usage_count += 1
                return self.model
            
            if self.loading:
                return None
            
            self.loading = True
        
        try:
            start_time = time.time()
            logger.info(f"🔄 Lazy loading {self.model_name}...")
            
            # Load the model
            if asyncio.iscoroutinefunction(self.loader_func):
                model = await self.loader_func()
            else:
                model = self.loader_func()
            
            load_duration = time.time() - start_time
            
            with self._lock:
                self.model = model
                self.load_time = load_duration
                self.last_used = time.time()
                self.usage_count = 1
                self.loading = False
            
            logger.info(f"✅ {self.model_name} loaded successfully in {load_duration:.2f}s")
            return model
            
        except Exception as e:
            with self._lock:
                self.loading = False
            logger.error(f"❌ Failed to load {self.model_name}: {e}")
            raise e
    
    def cleanup_if_unused(self):
        """Cleanup model if unused for too long"""
        with self._lock:
            if self.model is None:
                return
            
            # Don't cleanup if recently used (within 30 minutes)
            if self.last_used and (time.time() - self.last_used) < 1800:
                return
            
            # Don't cleanup if frequently used
            if self.usage_count > 10:
                return
        
        self.cleanup()
    
    def cleanup(self):
        """Force cleanup of the model"""
        with self._lock:
            if self.model is None:
                return
            
            logger.info(f"🧹 Cleaning up {self.model_name}")
            
            try:
                if self.cleanup_func:
                    self.cleanup_func(self.model)
                
                # Clear the model reference
                self.model = None
                
                # Force garbage collection
                import gc
                gc.collect()
                
                logger.info(f"✅ {self.model_name} cleaned up successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to cleanup {self.model_name}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model loading statistics"""
        with self._lock:
            return {
                "model_name": self.model_name,
                "loaded": self.model is not None,
                "loading": self.loading,
                "load_time": self.load_time,
                "last_used": self.last_used,
                "usage_count": self.usage_count
            }

class LazyEmbeddingModel:
    """Lazy loading wrapper for embedding models"""
    
    def __init__(self):
        self._model_loader = None
        self._initialized = False
    
    def _create_loader(self):
        """Create the lazy loader for embedding model"""
        if self._model_loader is not None:
            return
        
        async def load_embedding_model():
            """Load the JinaAI embedding model"""
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('jinaai/jina-embeddings-v2-base-en')
            logger.info("JinaAI v2 embedding model loaded (768 dimensions)")
            return model
        
        def cleanup_embedding_model(model):
            """Cleanup embedding model"""
            try:
                # Clear model from memory
                if hasattr(model, 'cpu'):
                    model.cpu()
                del model
            except Exception as e:
                logger.error(f"Error cleaning up embedding model: {e}")
        
        self._model_loader = LazyModelLoader(
            "JinaAI Embedding Model",
            load_embedding_model,
            cleanup_embedding_model
        )
        self._initialized = True
    
    async def encode(self, texts, **kwargs):
        """Encode texts using the embedding model"""
        if not self._initialized:
            self._create_loader()
        
        model = await self._model_loader.get_model()
        return model.encode(texts, **kwargs)
    
    def get_stats(self):
        """Get embedding model statistics"""
        if not self._initialized:
            return {"status": "not_initialized"}
        return self._model_loader.get_stats()

class LazyRerankerModel:
    """Lazy loading wrapper for reranker models"""
    
    def __init__(self):
        self._model_loader = None
        self._initialized = False
    
    def _create_loader(self):
        """Create the lazy loader for reranker model"""
        if self._model_loader is not None:
            return
        
        async def load_reranker_model():
            """Load the BGE reranker model"""
            from sentence_transformers import CrossEncoder
            
            model = CrossEncoder('BAAI/bge-reranker-large')
            logger.info("BGE reranker model loaded")
            return model
        
        def cleanup_reranker_model(model):
            """Cleanup reranker model"""
            try:
                if hasattr(model, 'model') and hasattr(model.model, 'cpu'):
                    model.model.cpu()
                del model
            except Exception as e:
                logger.error(f"Error cleaning up reranker model: {e}")
        
        self._model_loader = LazyModelLoader(
            "BGE Reranker Model",
            load_reranker_model,
            cleanup_reranker_model
        )
        self._initialized = True
    
    async def predict(self, sentence_pairs, **kwargs):
        """Predict relevance scores using the reranker model"""
        if not self._initialized:
            self._create_loader()
        
        model = await self._model_loader.get_model()
        return model.predict(sentence_pairs, **kwargs)
    
    def get_stats(self):
        """Get reranker model statistics"""
        if not self._initialized:
            return {"status": "not_initialized"}
        return self._model_loader.get_stats()

class LazyKeyBERTModel:
    """Lazy loading wrapper for KeyBERT model"""
    
    def __init__(self):
        self._model_loader = None
        self._initialized = False
    
    def _create_loader(self):
        """Create the lazy loader for KeyBERT model"""
        if self._model_loader is not None:
            return
        
        async def load_keybert_model():
            """Load the KeyBERT model"""
            try:
                from keybert import KeyBERT
                from sentence_transformers import SentenceTransformer
                
                # Use a lightweight model for KeyBERT
                sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                model = KeyBERT(model=sentence_model)
                logger.info("KeyBERT model loaded with all-MiniLM-L6-v2")
                return model
            except ImportError:
                logger.warning("KeyBERT not available")
                return None
        
        def cleanup_keybert_model(model):
            """Cleanup KeyBERT model"""
            try:
                if model and hasattr(model, 'model'):
                    if hasattr(model.model, 'cpu'):
                        model.model.cpu()
                del model
            except Exception as e:
                logger.error(f"Error cleaning up KeyBERT model: {e}")
        
        self._model_loader = LazyModelLoader(
            "KeyBERT Model",
            load_keybert_model,
            cleanup_keybert_model
        )
        self._initialized = True
    
    async def extract_keywords(self, text, **kwargs):
        """Extract keywords using KeyBERT"""
        if not self._initialized:
            self._create_loader()
        
        model = await self._model_loader.get_model()
        if model is None:
            return []  # Return empty list if KeyBERT not available
        
        return model.extract_keywords(text, **kwargs)
    
    def get_stats(self):
        """Get KeyBERT model statistics"""
        if not self._initialized:
            return {"status": "not_initialized"}
        return self._model_loader.get_stats()

# Global lazy model instances
lazy_embedding_model = LazyEmbeddingModel()
lazy_reranker_model = LazyRerankerModel()
lazy_keybert_model = LazyKeyBERTModel()

def get_model_stats() -> Dict[str, Any]:
    """Get statistics for all lazy-loaded models"""
    return {
        "embedding_model": lazy_embedding_model.get_stats(),
        "reranker_model": lazy_reranker_model.get_stats(),
        "keybert_model": lazy_keybert_model.get_stats()
    }

def cleanup_all_models():
    """Cleanup all lazy-loaded models"""
    logger.info("🧹 Cleaning up all lazy-loaded models")
    
    try:
        if lazy_embedding_model._initialized and lazy_embedding_model._model_loader:
            lazy_embedding_model._model_loader.cleanup()
    except Exception as e:
        logger.error(f"Error cleaning up embedding model: {e}")
    
    try:
        if lazy_reranker_model._initialized and lazy_reranker_model._model_loader:
            lazy_reranker_model._model_loader.cleanup()
    except Exception as e:
        logger.error(f"Error cleaning up reranker model: {e}")
    
    try:
        if lazy_keybert_model._initialized and lazy_keybert_model._model_loader:
            lazy_keybert_model._model_loader.cleanup()
    except Exception as e:
        logger.error(f"Error cleaning up KeyBERT model: {e}")
    
    logger.info("✅ All lazy-loaded models cleaned up")

# Register cleanup callback with memory manager
try:
    from app.utils.memory_manager import memory_manager
    memory_manager.register_cleanup_callback(cleanup_all_models)
except ImportError:
    logger.warning("Memory manager not available for lazy model cleanup")
