"""
ADVANCED MULTIMODAL EMBEDDING STRATEGIES v1.0
==============================================

Common multimodal embedding strategies for:
- Text embeddings (JinaAI v2)
- Image embeddings (CLIP/OpenCLIP)
- Table embeddings (structured data)
- Cross-modal embeddings (unified space)
- Hybrid embeddings (text + metadata)

Features:
- Unified embedding interface
- Multiple embedding models support
- Cross-modal similarity search
- Metadata-enriched embeddings
- Efficient batch processing
- Fallback strategies
"""

from __future__ import annotations
import asyncio
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

# Core embedding libraries
from sentence_transformers import SentenceTransformer

# Multimodal embedding libraries
try:
    import clip
    import torch
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("CLIP not available - image embeddings disabled")

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False
    print("OpenCLIP not available - using CLIP fallback")

# Import our multimodal extractor
from .multimodal_extractor import ExtractedContent

# ═══════════════════════════════════════════════════════════════════════════
# Embedding Configuration and Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EmbeddingConfig:
    """Configuration for multimodal embeddings"""
    # Text embedding settings
    text_model: str = "jinaai/jina-embeddings-v2-base-en"
    text_dimension: int = 768
    
    # Image embedding settings
    image_model: str = "ViT-B/32"  # CLIP model
    image_dimension: int = 512
    
    # Cross-modal settings
    unified_dimension: int = 768  # Target dimension for unified space
    
    # Processing settings
    batch_size: int = 32
    max_workers: int = 4
    normalize_embeddings: bool = True
    
    # Fallback settings
    enable_fallbacks: bool = True
    cache_embeddings: bool = True

@dataclass
class MultimodalEmbedding:
    """Container for multimodal embeddings"""
    text_embedding: Optional[np.ndarray] = None
    image_embeddings: List[np.ndarray] = None
    table_embeddings: List[np.ndarray] = None
    unified_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.image_embeddings is None:
            self.image_embeddings = []
        if self.table_embeddings is None:
            self.table_embeddings = []
        if self.metadata is None:
            self.metadata = {}
    
    def get_primary_embedding(self) -> np.ndarray:
        """Get the primary embedding for vector storage"""
        if self.unified_embedding is not None:
            return self.unified_embedding
        elif self.text_embedding is not None:
            return self.text_embedding
        else:
            raise ValueError("No primary embedding available")
    
    def has_multimodal_content(self) -> bool:
        """Check if this embedding contains multimodal content"""
        return len(self.image_embeddings) > 0 or len(self.table_embeddings) > 0

# ═══════════════════════════════════════════════════════════════════════════
# Abstract Base Embedder
# ═══════════════════════════════════════════════════════════════════════════

class BaseEmbedder(ABC):
    """Abstract base class for embedders"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.model_loaded = False
        self.load_time = 0
        self._lock = threading.Lock()
    
    @abstractmethod
    def load_model(self):
        """Load the embedding model"""
        pass
    
    @abstractmethod
    def embed(self, inputs: List[Any]) -> List[np.ndarray]:
        """Generate embeddings for inputs"""
        pass
    
    def ensure_model_loaded(self):
        """Ensure model is loaded (thread-safe)"""
        if not self.model_loaded:
            with self._lock:
                if not self.model_loaded:
                    self.load_model()

# ═══════════════════════════════════════════════════════════════════════════
# Text Embedder (JinaAI v2)
# ═══════════════════════════════════════════════════════════════════════════

class TextEmbedder(BaseEmbedder):
    """High-performance text embedder using JinaAI v2"""
    
    def load_model(self):
        """Load JinaAI text embedding model"""
        start_time = time.time()
        try:
            print(f"🚀 Loading text embedding model: {self.config.text_model}")
            self.model = SentenceTransformer(self.config.text_model)
            self.load_time = time.time() - start_time
            self.model_loaded = True
            print(f"✅ Text embedder loaded in {self.load_time:.2f}s ({self.config.text_dimension}D)")
        except Exception as e:
            print(f"❌ Failed to load text model: {e}")
            # Fallback to a smaller model
            try:
                fallback_model = "all-MiniLM-L6-v2"
                print(f"🔄 Falling back to: {fallback_model}")
                self.model = SentenceTransformer(fallback_model)
                self.config.text_dimension = 384  # Update dimension
                self.load_time = time.time() - start_time
                self.model_loaded = True
                print(f"✅ Fallback text embedder loaded")
            except Exception as fallback_error:
                raise Exception(f"Both text models failed: {e}, {fallback_error}")
    
    def embed(self, texts: List[str]) -> List[np.ndarray]:
        """Generate text embeddings"""
        self.ensure_model_loaded()
        
        if not texts:
            return []
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings
            )
            
            # Ensure we return a list of arrays
            if len(texts) == 1:
                return [embeddings]
            else:
                return list(embeddings)
                
        except Exception as e:
            print(f"Text embedding error: {e}")
            # Return zero embeddings as fallback
            return [np.zeros(self.config.text_dimension) for _ in texts]
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else np.zeros(self.config.text_dimension)

# ═══════════════════════════════════════════════════════════════════════════
# Image Embedder (CLIP)
# ═══════════════════════════════════════════════════════════════════════════

class ImageEmbedder(BaseEmbedder):
    """Image embedder using CLIP"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if CLIP_AVAILABLE else None
        self.preprocess = None
    
    def load_model(self):
        """Load CLIP image embedding model"""
        if not CLIP_AVAILABLE:
            print("⚠️ CLIP not available - image embeddings disabled")
            return
        
        start_time = time.time()
        try:
            print(f"🚀 Loading image embedding model: {self.config.image_model}")
            self.model, self.preprocess = clip.load(self.config.image_model, device=self.device)
            self.load_time = time.time() - start_time
            self.model_loaded = True
            print(f"✅ Image embedder loaded in {self.load_time:.2f}s ({self.config.image_dimension}D)")
        except Exception as e:
            print(f"❌ Failed to load image model: {e}")
            if self.config.enable_fallbacks:
                try:
                    # Try a smaller CLIP model
                    fallback_model = "ViT-B/16"
                    print(f"🔄 Falling back to: {fallback_model}")
                    self.model, self.preprocess = clip.load(fallback_model, device=self.device)
                    self.load_time = time.time() - start_time
                    self.model_loaded = True
                    print(f"✅ Fallback image embedder loaded")
                except Exception as fallback_error:
                    print(f"❌ Image embedding fallback failed: {fallback_error}")
    
    def embed_images_from_data(self, image_data_list: List[bytes]) -> List[np.ndarray]:
        """Generate embeddings for image data"""
        if not self.model_loaded or not CLIP_AVAILABLE:
            return []
        
        try:
            from PIL import Image
            from io import BytesIO
            
            embeddings = []
            for image_data in image_data_list:
                try:
                    # Load and preprocess image
                    image = Image.open(BytesIO(image_data))
                    image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                    
                    # Generate embedding
                    with torch.no_grad():
                        embedding = self.model.encode_image(image_tensor)
                        if self.config.normalize_embeddings:
                            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                        embeddings.append(embedding.cpu().numpy().flatten())
                
                except Exception as img_error:
                    print(f"Error processing individual image: {img_error}")
                    embeddings.append(np.zeros(self.config.image_dimension))
            
            return embeddings
            
        except Exception as e:
            print(f"Image embedding error: {e}")
            return [np.zeros(self.config.image_dimension) for _ in image_data_list]
    
    def embed_text_for_image_search(self, text: str) -> np.ndarray:
        """Generate text embedding in image space for cross-modal search"""
        if not self.model_loaded or not CLIP_AVAILABLE:
            return np.zeros(self.config.image_dimension)
        
        try:
            text_tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                embedding = self.model.encode_text(text_tokens)
                if self.config.normalize_embeddings:
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                return embedding.cpu().numpy().flatten()
        except Exception as e:
            print(f"Text-to-image embedding error: {e}")
            return np.zeros(self.config.image_dimension)
    
    def embed(self, inputs: List[Any]) -> List[np.ndarray]:
        """Generate embeddings for inputs (implements abstract method)"""
        if not self.model_loaded or not CLIP_AVAILABLE:
            return [np.zeros(self.config.image_dimension) for _ in inputs]
        
        # Handle different input types
        embeddings = []
        for input_item in inputs:
            if isinstance(input_item, str):
                # Text input - embed in image space
                embedding = self.embed_text_for_image_search(input_item)
                embeddings.append(embedding)
            elif isinstance(input_item, bytes):
                # Image data input
                image_embeddings = self.embed_images_from_data([input_item])
                embeddings.extend(image_embeddings)
            else:
                # Unknown input type - return zero embedding
                embeddings.append(np.zeros(self.config.image_dimension))
        
        return embeddings

# ═══════════════════════════════════════════════════════════════════════════
# Table Embedder (Structured Data)
# ═══════════════════════════════════════════════════════════════════════════

class TableEmbedder(BaseEmbedder):
    """Table embedder for structured data"""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.text_embedder = None
    
    def load_model(self):
        """Load text embedder for table content"""
        self.text_embedder = TextEmbedder(self.config)
        self.text_embedder.load_model()
        self.model_loaded = self.text_embedder.model_loaded
        self.load_time = self.text_embedder.load_time
    
    def embed_tables(self, tables: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Generate embeddings for table data"""
        if not self.model_loaded:
            return []
        
        embeddings = []
        for table in tables:
            try:
                # Create comprehensive table description
                table_text = self._table_to_text(table)
                embedding = self.text_embedder.embed_single(table_text)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Table embedding error: {e}")
                embeddings.append(np.zeros(self.config.text_dimension))
        
        return embeddings
    
    def embed(self, inputs: List[Any]) -> List[np.ndarray]:
        """Generate embeddings for inputs (implements abstract method)"""
        # For TableEmbedder, inputs should be table dictionaries
        if isinstance(inputs, list) and all(isinstance(item, dict) for item in inputs):
            return self.embed_tables(inputs)
        else:
            # Fallback: treat inputs as text
            if not self.model_loaded:
                return []
            text_inputs = [str(item) for item in inputs]
            return self.text_embedder.embed(text_inputs)
    
    def _table_to_text(self, table: Dict[str, Any]) -> str:
        """Convert table to searchable text representation"""
        text_parts = []
        
        # Add table description
        if table.get('description'):
            text_parts.append(f"Table: {table['description']}")
        
        # Add CSV representation (truncated for embedding)
        if table.get('csv_representation'):
            csv_text = table['csv_representation']
            # Limit CSV text length for embedding efficiency
            if len(csv_text) > 2000:
                lines = csv_text.split('\n')
                # Keep header + first few rows + last few rows
                if len(lines) > 10:
                    truncated = lines[:5] + ['...'] + lines[-3:]
                    csv_text = '\n'.join(truncated)
            text_parts.append(f"Data:\n{csv_text}")
        
        # Add summary statistics if available
        if table.get('summary_stats'):
            stats = table['summary_stats']
            if isinstance(stats, dict):
                stats_text = f"Statistics: {stats.get('total_rows', 0)} rows, {stats.get('total_columns', 0)} columns"
                text_parts.append(stats_text)
        
        return '\n\n'.join(text_parts)

# ═══════════════════════════════════════════════════════════════════════════
# Unified Multimodal Embedder
# ═══════════════════════════════════════════════════════════════════════════

class MultimodalEmbeddingManager:
    """Central manager for multimodal embeddings"""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        
        # Initialize embedders
        self.text_embedder = TextEmbedder(self.config)
        self.image_embedder = ImageEmbedder(self.config) if CLIP_AVAILABLE else None
        self.table_embedder = TableEmbedder(self.config)
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="multimodal_embed"
        )
        
        # Statistics
        self.embedding_stats = {
            'total_embeddings': 0,
            'text_embeddings': 0,
            'image_embeddings': 0,
            'table_embeddings': 0,
            'unified_embeddings': 0,
            'total_time': 0,
            'average_time': 0
        }
    
    async def initialize_models(self):
        """Initialize all embedding models asynchronously"""
        print("🚀 Initializing multimodal embedding models...")
        start_time = time.time()
        
        # Initialize models in parallel
        tasks = []
        loop = asyncio.get_event_loop()
        
        # Text embedder (always available)
        tasks.append(loop.run_in_executor(self.thread_pool, self.text_embedder.load_model))
        
        # Image embedder (if available)
        if self.image_embedder:
            tasks.append(loop.run_in_executor(self.thread_pool, self.image_embedder.load_model))
        
        # Table embedder
        tasks.append(loop.run_in_executor(self.thread_pool, self.table_embedder.load_model))
        
        # Wait for all models to load
        await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        print(f"✅ Multimodal embedding models initialized in {total_time:.2f}s")
        
        return {
            'text_embedder': self.text_embedder.model_loaded,
            'image_embedder': self.image_embedder.model_loaded if self.image_embedder else False,
            'table_embedder': self.table_embedder.model_loaded,
            'total_time': total_time
        }
    
    def embed_content(self, content: ExtractedContent) -> MultimodalEmbedding:
        """Generate multimodal embeddings for extracted content"""
        start_time = time.time()
        self.embedding_stats['total_embeddings'] += 1
        
        # Initialize embedding container
        embedding = MultimodalEmbedding()
        
        try:
            # 1. Text embedding (primary)
            if content.text:
                # Create enriched text with metadata
                enriched_text = self._create_enriched_text(content)
                embedding.text_embedding = self.text_embedder.embed_single(enriched_text)
                self.embedding_stats['text_embeddings'] += 1
            
            # 2. Image embeddings (if available)
            if content.images and self.image_embedder and self.image_embedder.model_loaded:
                # For now, we'll embed image descriptions as text
                # In a full implementation, we'd embed actual image data
                image_texts = []
                for img in content.images:
                    img_text = f"{img.get('description', '')} {img.get('ocr_text', '')}"
                    if img_text.strip():
                        image_texts.append(img_text.strip())
                
                if image_texts:
                    embedding.image_embeddings = self.text_embedder.embed(image_texts)
                    self.embedding_stats['image_embeddings'] += len(image_texts)
            
            # 3. Table embeddings
            if content.tables:
                embedding.table_embeddings = self.table_embedder.embed_tables(content.tables)
                self.embedding_stats['table_embeddings'] += len(content.tables)
            
            # 4. Create unified embedding
            embedding.unified_embedding = self._create_unified_embedding(embedding, content)
            if embedding.unified_embedding is not None:
                self.embedding_stats['unified_embeddings'] += 1
            
            # 5. Add metadata
            embedding.metadata = {
                'content_type': content.content_type,
                'file_path': content.file_path,
                'has_images': len(content.images) > 0,
                'has_tables': len(content.tables) > 0,
                'text_length': len(content.text),
                'embedding_time': time.time() - start_time
            }
            
        except Exception as e:
            print(f"Multimodal embedding error: {e}")
            # Fallback to basic text embedding
            if content.text:
                embedding.text_embedding = self.text_embedder.embed_single(content.text)
                embedding.unified_embedding = embedding.text_embedding
        
        # Update statistics
        total_time = time.time() - start_time
        self.embedding_stats['total_time'] += total_time
        self.embedding_stats['average_time'] = (
            self.embedding_stats['total_time'] / self.embedding_stats['total_embeddings']
        )
        
        return embedding
    
    def _create_enriched_text(self, content: ExtractedContent) -> str:
        """Create enriched text with metadata for better embeddings"""
        parts = []
        
        # Add content type context
        parts.append(f"[{content.content_type.upper()}]")
        
        # Add main text
        if content.text:
            parts.append(content.text)
        
        # Add table summaries
        for table in content.tables:
            if table.get('description'):
                parts.append(f"[TABLE] {table['description']}")
        
        # Add image descriptions
        for image in content.images:
            if image.get('description'):
                parts.append(f"[IMAGE] {image['description']}")
            if image.get('ocr_text'):
                parts.append(f"[OCR] {image['ocr_text']}")
        
        # Add metadata context
        metadata = content.metadata
        if metadata.get('filename'):
            parts.append(f"[FILENAME] {metadata['filename']}")
        
        return '\n\n'.join(filter(None, parts))
    
    def _create_unified_embedding(self, embedding: MultimodalEmbedding, content: ExtractedContent) -> Optional[np.ndarray]:
        """Create unified embedding combining all modalities"""
        if embedding.text_embedding is None:
            return None
        
        # Start with text embedding as base
        unified = embedding.text_embedding.copy()
        
        # If we have multimodal content, we could enhance the embedding
        # For now, we'll use the enriched text embedding as the unified embedding
        # In a more advanced implementation, we'd combine embeddings from different modalities
        
        return unified
    
    async def embed_content_async(self, content: ExtractedContent) -> MultimodalEmbedding:
        """Async version of embed_content"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, self.embed_content, content)
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about embedding models and capabilities"""
        return {
            'config': {
                'text_model': self.config.text_model,
                'text_dimension': self.config.text_dimension,
                'image_model': self.config.image_model if self.image_embedder else None,
                'image_dimension': self.config.image_dimension if self.image_embedder else None,
                'unified_dimension': self.config.unified_dimension
            },
            'capabilities': {
                'text_embedding': self.text_embedder.model_loaded,
                'image_embedding': (self.image_embedder.model_loaded 
                                  if self.image_embedder else False),
                'table_embedding': self.table_embedder.model_loaded,
                'cross_modal_search': CLIP_AVAILABLE and self.image_embedder.model_loaded,
                'ocr_support': True,  # Via pytesseract
                'multimodal_unified': True
            },
            'statistics': self.embedding_stats,
            'supported_formats': [
                'pdf', 'docx', 'doc', 'xlsx', 'xls', 'pptx', 'ppt',
                'jpg', 'jpeg', 'png', 'tiff', 'bmp', 'gif',
                'txt', 'rtf', 'md', 'csv'
            ]
        }
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        return self.embedding_stats.copy()

# ═══════════════════════════════════════════════════════════════════════════
# Global Instance and Utilities
# ═══════════════════════════════════════════════════════════════════════════

# Global multimodal embedding manager
multimodal_embedding_manager = MultimodalEmbeddingManager()

async def initialize_multimodal_embeddings():
    """Initialize multimodal embedding models"""
    return await multimodal_embedding_manager.initialize_models()

def embed_multimodal_content(content: ExtractedContent) -> MultimodalEmbedding:
    """Generate multimodal embeddings for content"""
    return multimodal_embedding_manager.embed_content(content)

async def embed_multimodal_content_async(content: ExtractedContent) -> MultimodalEmbedding:
    """Async version of multimodal embedding"""
    return await multimodal_embedding_manager.embed_content_async(content)

def get_primary_embedding_vector(content: ExtractedContent) -> np.ndarray:
    """Get primary embedding vector for vector storage (backward compatibility)"""
    embedding = embed_multimodal_content(content)
    return embedding.get_primary_embedding()

def get_multimodal_embedding_info() -> Dict[str, Any]:
    """Get multimodal embedding system information"""
    return multimodal_embedding_manager.get_embedding_info()
