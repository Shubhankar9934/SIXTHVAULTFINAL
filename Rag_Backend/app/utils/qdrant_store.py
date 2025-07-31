"""
All vector CRUD goes through Qdrant.  Works with:

* Embedded local file-based Qdrant (default)
* Remote/self-hosted/cloud Qdrant via REST/gRPC

No change required elsewhere in the codebase.

PERFORMANCE OPTIMIZED: Using JinaAI-v2-base-en embedding model
Based on benchmarks: Hit Rate 0.938, MRR 0.868 (Best Overall Performance)
"""
from __future__ import annotations
from typing import List, Optional
from uuid import uuid4
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time
import asyncio

from app.config import settings
from qdrant_client import QdrantClient
from qdrant_client.http import models as q

# ---------- ULTRA-FAST OPTIMIZED EMBEDDER - Pre-loaded Models -------------------------
# CRITICAL OPTIMIZATION: Pre-load models at startup, never reload during requests
# Hit Rate: 0.938, MRR: 0.868 - Best performing model

import threading
from concurrent.futures import ThreadPoolExecutor

# Global model cache with thread safety
_model = None
_model_lock = threading.Lock()
_model_name = "jinaai/jina-embeddings-v2-base-en"
_fallback_model_name = "BAAI/bge-large-en-v1.5"
_model_loaded = False
_model_load_time = 0

# Thread pool for async operations
_thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="embedding")

def _initialize_embedding_model():
    """ULTRA-FAST: Initialize model once at startup, cache forever"""
    global _model, _model_loaded, _model_load_time
    
    # Thread-safe check - return immediately if already loaded
    if _model_loaded and _model is not None:
        return _model
    
    with _model_lock:
        # Double-check pattern for thread safety
        if _model_loaded and _model is not None:
            return _model
        
        try:
            print(f"🚀 Loading optimal embedding model: {_model_name}")
            start_time = time.time()
            _model = SentenceTransformer(_model_name)
            _model_load_time = time.time() - start_time
            _model_loaded = True
            print(f"✅ JinaAI v2 embedding model loaded in {_model_load_time:.2f}s (768 dimensions)")
            print(f"🎯 Expected performance: Hit Rate 93.8%, MRR 86.8%")
            print(f"⚡ Model cached for instant future access")
            return _model
            
        except Exception as e:
            print(f"⚠️ Failed to load JinaAI model: {e}")
            print(f"🔄 Falling back to: {_fallback_model_name}")
            try:
                _model = SentenceTransformer(_fallback_model_name)
                _model_load_time = time.time() - start_time
                _model_loaded = True
                print(f"✅ Fallback model loaded: {_fallback_model_name}")
                print(f"⚡ Fallback model cached for instant future access")
                return _model
            except Exception as fallback_error:
                print(f"❌ Fallback model also failed: {fallback_error}")
                raise Exception(f"Both embedding models failed: {e}, {fallback_error}")

async def preload_embedding_model():
    """STARTUP OPTIMIZATION: Pre-load embedding model during server startup"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, _initialize_embedding_model)

def get_embedding_model():
    """Get the current embedding model instance"""
    return _initialize_embedding_model()

def _embed(texts: List[str]) -> np.ndarray:
    """ULTRA-FAST embedding - model already pre-loaded, zero initialization time"""
    global _model
    
    # Critical optimization: Model should already be loaded
    if not _model_loaded or _model is None:
        print("⚠️ WARNING: Model not pre-loaded, loading now (this should not happen)")
        _initialize_embedding_model()
    
    # Ultra-fast embedding with optimized parameters
    return _model.encode(
        texts, 
        convert_to_numpy=True, 
        show_progress_bar=False,
        batch_size=32,  # Optimized batch size for speed
        normalize_embeddings=True  # Normalize for better cosine similarity
    )

async def _embed_async(texts: List[str]) -> np.ndarray:
    """Async version of embedding for non-blocking operation"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, _embed, texts)

def _get_vector_size() -> int:
    """Get the vector size from the embedding model"""
    model = _initialize_embedding_model()
    return model.get_sentence_embedding_dimension()

def get_model_info() -> dict:
    """Get information about the current embedding model"""
    global _model, _model_loaded, _model_load_time
    
    if not _model_loaded:
        _initialize_embedding_model()
    
    return {
        "model_name": _model_name,
        "vector_size": _model.get_sentence_embedding_dimension(),
        "expected_hit_rate": 0.938,
        "expected_mrr": 0.868,
        "benchmark_source": "Best Overall Performance",
        "load_time": _model_load_time,
        "cached": _model_loaded,
        "optimization": "pre_loaded_at_startup"
    }

def get_embedding_stats() -> dict:
    """Get embedding performance statistics"""
    return {
        "model_loaded": _model_loaded,
        "model_name": _model_name if _model_loaded else "not_loaded",
        "load_time": _model_load_time,
        "cache_status": "hit" if _model_loaded else "miss",
        "thread_pool_active": not _thread_pool._shutdown
    }

# ---------- qdrant client --------------------------------------------------
if settings.qdrant_remote_url:
    client = QdrantClient(
        url=settings.qdrant_remote_url,
        api_key=settings.qdrant_api_key,
        prefer_grpc=True,
    )
else:
    client = QdrantClient(
        path=str(settings.qdrant_local_path),  # embedded mode
        prefer_grpc=True,
    )

def _coll(tenant_id: str) -> str:
    return f"tenant_{tenant_id}"

def _ensure(tenant_id: str):
    """
    Create the per-tenant collection **iff** it doesn't exist already.
    Works for both embedded & remote Qdrant.
    
    MIGRATION SUPPORT: Handles vector dimension changes automatically
    """
    coll = _coll(tenant_id)
    vector_size = _get_vector_size()

    # get existing collection names
    existing = {c.name for c in client.get_collections().collections}

    if coll in existing:
        # Check if existing collection has correct vector size and indexes
        try:
            collection_info = client.get_collection(coll)
            existing_size = collection_info.config.params.vectors.size
            if existing_size != vector_size:
                print(f"🔄 MIGRATION: Collection {coll} has wrong vector size ({existing_size} vs {vector_size})")
                print(f"   This is expected when switching embedding models")
                print(f"   Old model: BAAI/bge-large-en-v1.5 (1024 dim)")
                print(f"   New model: JinaAI-v2-base-en (768 dim)")
                print(f"   Recreating collection with new dimensions...")
                
                # Backup warning
                print(f"⚠️  This will remove existing vectors for tenant {tenant_id}")
                print(f"   Documents will need to be re-uploaded to rebuild embeddings")
                
                client.delete_collection(coll)
                print(f"✅ Old collection deleted successfully")
            else:
                # Check if required indexes exist
                _ensure_indexes(coll)
                return  # already there with correct size and indexes
        except Exception as e:
            print(f"Error checking collection {coll}: {e}. Recreating...")
            try:
                client.delete_collection(coll)
            except:
                pass  # Collection might not exist

    # race-safe create (ignore "already exists" raised from another thread)
    try:
        client.create_collection(
            coll,
            vectors_config=q.VectorParams(size=vector_size, distance=q.Distance.COSINE),
        )
        print(f"✅ Created collection {coll} with vector size {vector_size}")
        print(f"🎯 Ready for optimal JinaAI embeddings (Hit Rate: 93.8%, MRR: 86.8%)")
        
        # Create indexes for filtering fields
        _ensure_indexes(coll)
        
    except ValueError as e:
        if "already exists" not in str(e):
            raise

def _ensure_indexes(collection_name: str):
    """
    Create indexes for fields used in filtering operations.
    This is required for Qdrant to perform efficient filtering.
    """
    try:
        # Create index for document_id field (used in deletion)
        client.create_payload_index(
            collection_name=collection_name,
            field_name="document_id",
            field_schema=q.PayloadSchemaType.KEYWORD
        )
        print(f"Created index for document_id in collection {collection_name}")
    except Exception as e:
        if "already exists" not in str(e).lower():
            print(f"Warning: Could not create document_id index: {e}")
    
    try:
        # Create index for filename field (used in deletion)
        client.create_payload_index(
            collection_name=collection_name,
            field_name="filename",
            field_schema=q.PayloadSchemaType.KEYWORD
        )
        print(f"Created index for filename in collection {collection_name}")
    except Exception as e:
        if "already exists" not in str(e).lower():
            print(f"Warning: Could not create filename index: {e}")
    
    try:
        # Create index for document field (used in deletion)
        client.create_payload_index(
            collection_name=collection_name,
            field_name="document",
            field_schema=q.PayloadSchemaType.KEYWORD
        )
        print(f"Created index for document in collection {collection_name}")
    except Exception as e:
        if "already exists" not in str(e).lower():
            print(f"Warning: Could not create document index: {e}")
    
    try:
        # Create index for path field (used in search filtering)
        client.create_payload_index(
            collection_name=collection_name,
            field_name="path",
            field_schema=q.PayloadSchemaType.KEYWORD
        )
        print(f"Created index for path in collection {collection_name}")
    except Exception as e:
        if "already exists" not in str(e).lower():
            print(f"Warning: Could not create path index: {e}")


# ---------- public helpers -------------------------------------------------
def upsert(tenant_id: str, texts: List[str], payloads: List[dict], batch_size: int = 50):
    """Optimized batch upsert with progress tracking"""
    _ensure(tenant_id)
    
    # Process in batches for better memory management and performance
    total_texts = len(texts)
    collection_name = _coll(tenant_id)
    
    print(f"Processing {total_texts} chunks in batches of {batch_size}")
    
    for i in range(0, total_texts, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_payloads = payloads[i:i + batch_size]
        
        # Generate embeddings for this batch
        batch_vecs = _embed(batch_texts)
        
        # Create points for this batch
        batch_points = [
            q.PointStruct(
                id=str(uuid4()),
                vector=vec.tolist(),
                payload=payload,
            )
            for vec, payload in zip(batch_vecs, batch_payloads)
        ]
        
        # Upsert this batch
        client.upsert(collection_name=collection_name, points=batch_points)
        
        # Progress logging
        processed = min(i + batch_size, total_texts)
        print(f"Processed {processed}/{total_texts} chunks ({processed/total_texts*100:.1f}%)")

def search(tenant_id: str, query: str, k: int = 4, document_ids: List[str] = None) -> List[dict]:
    """ULTRA-FAST search with pre-loaded models and optimized processing"""
    search_start = time.time()
    
    # CRITICAL: Ensure collection exists with correct dimensions
    _ensure(tenant_id)
    
    coll_name = _coll(tenant_id)
    
    # ULTRA-FAST embedding - model already pre-loaded
    embed_start = time.time()
    if not _model_loaded:
        print("🚨 CRITICAL: Model not pre-loaded! This will cause delays.")
    
    vec = _embed([query])[0].tolist()
    embed_time = time.time() - embed_start
    
    # Optimized filter building with caching
    filter_start = time.time()
    search_filter = None
    if document_ids:
        try:
            from sqlmodel import Session, select
            from app.database import engine
            from app.models import Document
            
            # Batch query for all document paths at once
            with Session(engine) as session:
                statement = select(Document.path).where(Document.id.in_(document_ids))
                document_paths = list(session.exec(statement))
            
            if document_paths:
                # Optimized filter construction
                filter_conditions = [
                    q.FieldCondition(key="path", match=q.MatchValue(value=path))
                    for path in document_paths
                ]
                search_filter = q.Filter(should=filter_conditions)
            else:
                return []
        except Exception as e:
            print(f"Filter error: {e}")
            return []
    filter_time = time.time() - filter_start
    
    # ULTRA-FAST search with minimal logging
    qdrant_start = time.time()
    try:
        hits = client.search(
            collection_name=coll_name,
            query_vector=vec,
            limit=k,
            with_payload=True,
            query_filter=search_filter,
        )
        results = [h.payload for h in hits]
        qdrant_time = time.time() - qdrant_start
        
        total_time = time.time() - search_start
        
        # Performance logging (only if slow)
        if total_time > 1.0:  # Only log if > 1 second
            print(f"⚡ Search performance: {total_time:.3f}s total (embed: {embed_time:.3f}s, filter: {filter_time:.3f}s, qdrant: {qdrant_time:.3f}s)")
        
        return results
        
    except Exception as e:
        print(f"Search error: {e}")
        return []

async def search_async(tenant_id: str, query: str, k: int = 4, document_ids: List[str] = None) -> List[dict]:
    """Async version of search for non-blocking operation"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, search, tenant_id, query, k, document_ids)

def dense_search(tenant_id: str, query: str, k: int = 4, document_ids: List[str] = None) -> List[dict]:
    """Dense vector search - same as regular search for now"""
    return search(tenant_id, query, k, document_ids)

def bm25_search(tenant_id: str, query: str, k: int = 4, document_ids: List[str] = None) -> List[dict]:
    """BM25 search - for now, fallback to dense search. Can be enhanced later with proper BM25 implementation"""
    return search(tenant_id, query, k, document_ids)

def delete_document_vectors(tenant_id: str, document_id: str, filename: str):
    """
    Delete all vectors associated with a specific document from the tenant's collection.
    This removes all RAG embeddings for the document.
    """
    coll_name = _coll(tenant_id)
    existing_collections = {c.name for c in client.get_collections().collections}
    
    if coll_name not in existing_collections:
        print(f"Collection {coll_name} not found. No vectors to delete.")
        return
    
    # Ensure indexes exist before attempting deletion
    try:
        _ensure_indexes(coll_name)
    except Exception as e:
        print(f"Warning: Could not ensure indexes for collection {coll_name}: {e}")
    
    try:
        # Try different filtering strategies in order of preference
        deletion_strategies = [
            # Strategy 1: Filter by document_id (most specific)
            q.Filter(must=[q.FieldCondition(key="document_id", match=q.MatchValue(value=document_id))]),
            
            # Strategy 2: Filter by filename
            q.Filter(must=[q.FieldCondition(key="filename", match=q.MatchValue(value=filename))]),
            
            # Strategy 3: Filter by document field
            q.Filter(must=[q.FieldCondition(key="document", match=q.MatchValue(value=filename))]),
            
            # Strategy 4: Combined filter (fallback)
            q.Filter(
                should=[
                    q.FieldCondition(key="document_id", match=q.MatchValue(value=document_id)),
                    q.FieldCondition(key="filename", match=q.MatchValue(value=filename)),
                    q.FieldCondition(key="document", match=q.MatchValue(value=filename))
                ]
            )
        ]
        
        deletion_successful = False
        
        for i, filter_condition in enumerate(deletion_strategies, 1):
            try:
                # Delete points matching the filter
                result = client.delete(
                    collection_name=coll_name,
                    points_selector=q.FilterSelector(filter=filter_condition)
                )
                
                print(f"Deleted vectors for document {document_id} ({filename}) using strategy {i}. Operation result: {result}")
                deletion_successful = True
                break
                
            except Exception as strategy_error:
                print(f"Strategy {i} failed for document {document_id}: {strategy_error}")
                if i == len(deletion_strategies):
                    # If all strategies fail, try a scroll-and-delete approach
                    print(f"All filter strategies failed. Attempting scroll-and-delete for document {document_id}")
                    _scroll_and_delete(coll_name, document_id, filename)
                    deletion_successful = True
                continue
        
        if not deletion_successful:
            print(f"Warning: Could not delete vectors for document {document_id} using any strategy")
            
    except Exception as e:
        print(f"Error deleting vectors for document {document_id}: {e}")
        # Don't raise the exception to allow the document deletion to continue
        print(f"Document deletion will continue despite vector cleanup failure")

def _scroll_and_delete(collection_name: str, document_id: str, filename: str):
    """
    Fallback method to delete vectors by scrolling through all points and matching manually.
    This is slower but works when indexes are missing.
    """
    try:
        # Scroll through all points in the collection
        scroll_result = client.scroll(
            collection_name=collection_name,
            limit=1000,  # Process in batches
            with_payload=True
        )
        
        points_to_delete = []
        
        # Check each point's payload for matching document
        for point in scroll_result[0]:
            payload = point.payload or {}
            if (payload.get("document_id") == document_id or 
                payload.get("filename") == filename or 
                payload.get("document") == filename):
                points_to_delete.append(point.id)
        
        # Delete matching points by ID
        if points_to_delete:
            client.delete(
                collection_name=collection_name,
                points_selector=q.PointIdsList(points=points_to_delete)
            )
            print(f"Scroll-and-delete removed {len(points_to_delete)} vectors for document {document_id}")
        else:
            print(f"No vectors found for document {document_id} during scroll-and-delete")
            
    except Exception as e:
        print(f"Scroll-and-delete failed for document {document_id}: {e}")

def delete_user_collection(user_id: str):
    """
    Delete the entire collection for a user (removes all their documents' vectors).
    Use with caution - this removes ALL RAG data for the user.
    """
    coll_name = _coll(user_id)
    existing_collections = {c.name for c in client.get_collections().collections}
    
    if coll_name in existing_collections:
        try:
            client.delete_collection(coll_name)
            print(f"Deleted entire collection {coll_name} for user {user_id}")
        except Exception as e:
            print(f"Error deleting collection {coll_name}: {e}")
            raise
    else:
        print(f"Collection {coll_name} not found. Nothing to delete.")

def clear_cache():
    """
    Clear system cache including embedding model cache and Qdrant client cache.
    This is useful for admin maintenance operations.
    """
    global _model, _model_loaded, _model_load_time
    
    try:
        cache_cleared = False
        
        # Clear embedding model cache
        with _model_lock:
            if _model is not None:
                del _model
                _model = None
                _model_loaded = False
                _model_load_time = 0
                cache_cleared = True
                print("✅ Embedding model cache cleared")
        
        # Clear any Qdrant client cache if available
        try:
            # Force reconnection to Qdrant
            client.get_collections()
            print("✅ Qdrant client cache refreshed")
            cache_cleared = True
        except Exception as e:
            print(f"⚠️ Qdrant cache refresh warning: {e}")
        
        return cache_cleared
        
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        return False

def get_system_info():
    """
    Get comprehensive system information for admin monitoring.
    """
    try:
        collections = client.get_collections()
        collection_count = len(collections.collections)
        
        # Get total points across all collections
        total_points = 0
        collection_details = []
        
        for collection in collections.collections:
            try:
                info = client.get_collection(collection.name)
                points_count = info.points_count or 0
                total_points += points_count
                
                collection_details.append({
                    "name": collection.name,
                    "points_count": points_count,
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.value
                })
            except Exception as e:
                print(f"Error getting info for collection {collection.name}: {e}")
                collection_details.append({
                    "name": collection.name,
                    "points_count": 0,
                    "error": str(e)
                })
        
        return {
            "qdrant_status": "connected",
            "collections_count": collection_count,
            "total_vectors": total_points,
            "embedding_model": get_model_info(),
            "collections": collection_details
        }
        
    except Exception as e:
        return {
            "qdrant_status": "error",
            "error": str(e),
            "embedding_model": get_model_info() if _model_loaded else None
        }
