"""
Processing Tracker - Database-backed wrapper for backward compatibility
This module now redirects all calls to the new ProcessingService
"""
import asyncio
from typing import Dict, Set, Optional, Any
from enum import Enum

# Import the new database-backed service
from app.services.processing_service import ProcessingService, ProcessingStatus
from app.models import ProcessingDocument as DBProcessingDocument

# Re-export ProcessingStatus for backward compatibility
__all__ = ['ProcessingStatus', 'ProcessingDocument']

# Backward compatibility class that wraps the database model
class ProcessingDocument:
    def __init__(self, db_doc: DBProcessingDocument):
        self.doc_id = db_doc.doc_id
        self.filename = db_doc.filename
        self.batch_id = db_doc.batch_id
        self.owner_id = db_doc.owner_id
        self.status = ProcessingStatus(db_doc.status)
        self.start_time = db_doc.created_at.timestamp() if db_doc.created_at else 0
        self.file_path = db_doc.file_path
        self.progress = db_doc.progress
        self.stage = db_doc.stage
        self.cancellation_requested = db_doc.cancellation_requested
        self.error_message = db_doc.error_message

# Wrapper functions that redirect to the new ProcessingService
async def register_processing_document(
    doc_id: str, 
    filename: str, 
    batch_id: str, 
    owner_id: str, 
    file_path: Optional[str] = None
) -> None:
    """Register a document as being processed"""
    await ProcessingService.register_processing_document(
        doc_id=doc_id,
        filename=filename,
        batch_id=batch_id,
        owner_id=owner_id,
        file_path=file_path
    )

async def update_processing_status(
    doc_id: str, 
    status: ProcessingStatus, 
    progress: int = None, 
    stage: str = None,
    error_message: str = None
) -> None:
    """Update the processing status of a document"""
    await ProcessingService.update_processing_status(
        doc_id=doc_id,
        status=status,
        progress=progress,
        stage=stage,
        error_message=error_message
    )

async def request_cancellation(doc_id: str, owner_id: str = None) -> bool:
    """Request cancellation of a processing document"""
    if owner_id:
        return await ProcessingService.request_cancellation(doc_id, owner_id)
    else:
        # Try to get the document first to find the owner
        db_doc = await ProcessingService.get_processing_document(doc_id)
        if db_doc:
            return await ProcessingService.request_cancellation(doc_id, db_doc.owner_id)
        return False

async def is_cancellation_requested(doc_id: str) -> bool:
    """Check if cancellation has been requested for a document"""
    return await ProcessingService.is_cancellation_requested(doc_id)

async def get_processing_document(doc_id: str) -> Optional[ProcessingDocument]:
    """Get processing document by ID"""
    db_doc = await ProcessingService.get_processing_document(doc_id)
    return ProcessingDocument(db_doc) if db_doc else None

async def get_processing_document_by_filename(filename: str) -> Optional[ProcessingDocument]:
    """Get processing document by filename - requires owner_id for isolation"""
    # This function is deprecated as it doesn't provide user isolation
    # Return None to prevent cross-user access
    return None

async def find_processing_document_by_name(filename: str, owner_id: str) -> Optional[ProcessingDocument]:
    """Find processing document by filename and owner"""
    db_doc = await ProcessingService.find_processing_document_by_filename(filename, owner_id)
    return ProcessingDocument(db_doc) if db_doc else None

async def get_batch_documents(batch_id: str, owner_id: str = None) -> Set[str]:
    """Get all document IDs in a batch"""
    if owner_id:
        docs = await ProcessingService.get_batch_documents(batch_id, owner_id)
        return {doc.doc_id for doc in docs}
    return set()

async def remove_processing_document(doc_id: str, owner_id: str = None) -> Optional[ProcessingDocument]:
    """Remove a document from processing tracking"""
    db_doc = await ProcessingService.remove_processing_document(doc_id, owner_id)
    return ProcessingDocument(db_doc) if db_doc else None

async def cleanup_batch(batch_id: str, owner_id: str = None) -> int:
    """Clean up all documents in a batch"""
    if owner_id:
        return await ProcessingService.cleanup_batch(batch_id, owner_id)
    return 0

async def get_all_processing_documents(owner_id: str = None) -> Dict[str, ProcessingDocument]:
    """Get all processing documents, optionally filtered by owner"""
    if owner_id:
        docs = await ProcessingService.get_user_processing_documents(owner_id)
        return {doc.doc_id: ProcessingDocument(doc) for doc in docs}
    return {}

async def find_processing_document_by_batch_id(batch_id: str, owner_id: str) -> Optional[ProcessingDocument]:
    """Find any processing document by batch ID and owner"""
    db_doc = await ProcessingService.find_processing_document_by_batch_id(batch_id, owner_id)
    return ProcessingDocument(db_doc) if db_doc else None

async def cleanup_completed_documents(max_age_seconds: int = 3600) -> int:
    """Clean up completed/cancelled documents older than max_age_seconds"""
    max_age_hours = max_age_seconds / 3600
    return await ProcessingService.cleanup_completed_documents(max_age_hours)

# Background cleanup task - now handled by ProcessingService
async def start_cleanup_task():
    """Background cleanup is now handled by ProcessingService"""
    pass

def ensure_cleanup_task():
    """Cleanup task is now handled by ProcessingService"""
    pass

# Clear all in-memory state since we're now database-backed
async def clear_all_processing_documents() -> int:
    """Emergency function to clear all processing documents"""
    return await ProcessingService.cleanup_all_orphaned_documents()

print("🔄 Processing tracker now using database-backed ProcessingService")
