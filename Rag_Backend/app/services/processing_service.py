"""
Database-backed Processing Service - Replaces in-memory processing tracker
Provides user isolation and persistence across sessions
"""
import asyncio
import time
from typing import Dict, Set, Optional, Any, List
from datetime import datetime, timedelta
from sqlmodel import Session, select
from app.database import engine
from app.models import ProcessingDocument
from enum import Enum

class ProcessingStatus(Enum):
    QUEUED = "queued"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"

class ProcessingService:
    """Database-backed processing service for user-isolated document tracking"""
    
    @staticmethod
    async def register_processing_document(
        doc_id: str, 
        filename: str, 
        batch_id: str, 
        owner_id: str, 
        file_path: Optional[str] = None,
        s3_key: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        file_size: Optional[int] = None,
        content_type: Optional[str] = None
    ) -> ProcessingDocument:
        """Register a document as being processed"""
        with Session(engine) as session:
            # Check if document already exists
            existing = session.exec(
                select(ProcessingDocument).where(ProcessingDocument.doc_id == doc_id)
            ).first()
            
            if existing:
                # Update existing document
                existing.filename = filename
                existing.batch_id = batch_id
                existing.owner_id = owner_id
                existing.file_path = file_path
                existing.s3_key = s3_key
                existing.s3_bucket = s3_bucket
                existing.file_size = file_size
                existing.content_type = content_type
                existing.updated_at = datetime.utcnow()
                existing.status = ProcessingStatus.QUEUED.value
                existing.progress = 0
                existing.stage = "queued"
                existing.cancellation_requested = False
                existing.cleanup_completed = False
                
                session.add(existing)
                session.commit()
                session.refresh(existing)
                
                print(f"Updated existing processing document: {doc_id} ({filename})")
                return existing
            
            # Create new processing document
            processing_doc = ProcessingDocument(
                doc_id=doc_id,
                filename=filename,
                batch_id=batch_id,
                owner_id=owner_id,
                file_path=file_path,
                s3_key=s3_key,
                s3_bucket=s3_bucket,
                file_size=file_size,
                content_type=content_type,
                status=ProcessingStatus.QUEUED.value,
                progress=0,
                stage="queued"
            )
            
            session.add(processing_doc)
            session.commit()
            session.refresh(processing_doc)
            
            print(f"Registered new processing document: {doc_id} ({filename}) for user {owner_id}")
            return processing_doc
    
    @staticmethod
    async def update_processing_status(
        doc_id: str, 
        status: ProcessingStatus, 
        progress: Optional[int] = None, 
        stage: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update the processing status of a document"""
        with Session(engine) as session:
            doc = session.exec(
                select(ProcessingDocument).where(ProcessingDocument.doc_id == doc_id)
            ).first()
            
            if not doc:
                print(f"Processing document not found: {doc_id}")
                return False
            
            doc.status = status.value
            doc.updated_at = datetime.utcnow()
            
            if progress is not None:
                doc.progress = progress
            if stage is not None:
                doc.stage = stage
            if error_message is not None:
                doc.error_message = error_message
            
            # Set timestamps based on status
            if status == ProcessingStatus.PROCESSING and not doc.started_at:
                doc.started_at = datetime.utcnow()
            elif status in [ProcessingStatus.COMPLETED, ProcessingStatus.CANCELLED, ProcessingStatus.ERROR]:
                doc.completed_at = datetime.utcnow()
            
            session.add(doc)
            session.commit()
            
            print(f"Updated processing status for {doc_id}: {status.value} ({progress}% - {stage})")
            return True
    
    @staticmethod
    async def request_cancellation(doc_id: str, owner_id: str) -> bool:
        """Request cancellation of a processing document (user-isolated)"""
        with Session(engine) as session:
            doc = session.exec(
                select(ProcessingDocument).where(
                    ProcessingDocument.doc_id == doc_id,
                    ProcessingDocument.owner_id == owner_id
                )
            ).first()
            
            if not doc:
                print(f"Processing document not found for cancellation: {doc_id} (user: {owner_id})")
                return False
            
            if doc.status in [ProcessingStatus.QUEUED.value, ProcessingStatus.UPLOADING.value, ProcessingStatus.PROCESSING.value]:
                doc.cancellation_requested = True
                doc.status = ProcessingStatus.CANCELLED.value
                doc.updated_at = datetime.utcnow()
                doc.completed_at = datetime.utcnow()
                
                session.add(doc)
                session.commit()
                
                print(f"Cancellation requested for document: {doc_id}")
                return True
            
            return False
    
    @staticmethod
    async def is_cancellation_requested(doc_id: str) -> bool:
        """Check if cancellation has been requested for a document"""
        with Session(engine) as session:
            doc = session.exec(
                select(ProcessingDocument).where(ProcessingDocument.doc_id == doc_id)
            ).first()
            
            return doc.cancellation_requested if doc else False
    
    @staticmethod
    async def get_processing_document(doc_id: str, owner_id: Optional[str] = None) -> Optional[ProcessingDocument]:
        """Get processing document by ID (optionally user-isolated)"""
        with Session(engine) as session:
            query = select(ProcessingDocument).where(ProcessingDocument.doc_id == doc_id)
            
            if owner_id:
                query = query.where(ProcessingDocument.owner_id == owner_id)
            
            return session.exec(query).first()
    
    @staticmethod
    async def find_processing_document_by_filename(filename: str, owner_id: str) -> Optional[ProcessingDocument]:
        """Find processing document by filename and owner (user-isolated)"""
        with Session(engine) as session:
            return session.exec(
                select(ProcessingDocument).where(
                    ProcessingDocument.filename == filename,
                    ProcessingDocument.owner_id == owner_id
                )
            ).first()
    
    @staticmethod
    async def find_processing_document_by_batch_id(batch_id: str, owner_id: str) -> Optional[ProcessingDocument]:
        """Find any processing document by batch ID and owner (user-isolated)"""
        with Session(engine) as session:
            return session.exec(
                select(ProcessingDocument).where(
                    ProcessingDocument.batch_id == batch_id,
                    ProcessingDocument.owner_id == owner_id
                )
            ).first()
    
    @staticmethod
    async def get_batch_documents(batch_id: str, owner_id: str) -> List[ProcessingDocument]:
        """Get all documents in a batch for a specific user"""
        with Session(engine) as session:
            return session.exec(
                select(ProcessingDocument).where(
                    ProcessingDocument.batch_id == batch_id,
                    ProcessingDocument.owner_id == owner_id
                )
            ).all()
    
    @staticmethod
    async def remove_processing_document(doc_id: str, owner_id: Optional[str] = None) -> Optional[ProcessingDocument]:
        """Remove a document from processing tracking (optionally user-isolated)"""
        with Session(engine) as session:
            query = select(ProcessingDocument).where(ProcessingDocument.doc_id == doc_id)
            
            if owner_id:
                query = query.where(ProcessingDocument.owner_id == owner_id)
            
            doc = session.exec(query).first()
            
            if doc:
                session.delete(doc)
                session.commit()
                print(f"Removed processing document: {doc_id}")
                return doc
            
            return None
    
    @staticmethod
    async def cleanup_batch(batch_id: str, owner_id: str) -> int:
        """Clean up all documents in a batch for a specific user"""
        with Session(engine) as session:
            docs = session.exec(
                select(ProcessingDocument).where(
                    ProcessingDocument.batch_id == batch_id,
                    ProcessingDocument.owner_id == owner_id
                )
            ).all()
            
            count = 0
            for doc in docs:
                session.delete(doc)
                count += 1
            
            session.commit()
            
            print(f"Cleaned up batch {batch_id} for user {owner_id}: {count} documents")
            return count
    
    @staticmethod
    async def get_user_processing_documents(owner_id: str, status_filter: Optional[List[str]] = None) -> List[ProcessingDocument]:
        """Get all processing documents for a user, optionally filtered by status"""
        with Session(engine) as session:
            query = select(ProcessingDocument).where(ProcessingDocument.owner_id == owner_id)
            
            if status_filter:
                query = query.where(ProcessingDocument.status.in_(status_filter))
            
            return session.exec(query).all()
    
    @staticmethod
    async def cleanup_completed_documents(max_age_hours: int = 24) -> int:
        """Clean up completed/cancelled documents older than max_age_hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        with Session(engine) as session:
            docs_to_remove = session.exec(
                select(ProcessingDocument).where(
                    ProcessingDocument.status.in_([
                        ProcessingStatus.COMPLETED.value,
                        ProcessingStatus.CANCELLED.value,
                        ProcessingStatus.ERROR.value
                    ]),
                    ProcessingDocument.updated_at < cutoff_time
                )
            ).all()
            
            count = 0
            for doc in docs_to_remove:
                session.delete(doc)
                count += 1
            
            session.commit()
            
            if count > 0:
                print(f"Cleaned up {count} old completed/cancelled documents")
            
            return count
    
    @staticmethod
    async def cleanup_orphaned_documents(max_age_hours: int = 2) -> int:
        """Clean up documents stuck in processing state for too long"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        with Session(engine) as session:
            orphaned_docs = session.exec(
                select(ProcessingDocument).where(
                    ProcessingDocument.status.in_([
                        ProcessingStatus.QUEUED.value,
                        ProcessingStatus.UPLOADING.value,
                        ProcessingStatus.PROCESSING.value
                    ]),
                    ProcessingDocument.created_at < cutoff_time
                )
            ).all()
            
            count = 0
            for doc in orphaned_docs:
                doc.status = ProcessingStatus.ERROR.value
                doc.error_message = "Processing timeout - document orphaned"
                doc.updated_at = datetime.utcnow()
                doc.completed_at = datetime.utcnow()
                session.add(doc)
                count += 1
            
            session.commit()
            
            if count > 0:
                print(f"Marked {count} orphaned documents as error")
            
            return count
    
    @staticmethod
    async def mark_cleanup_completed(doc_id: str) -> bool:
        """Mark a document as having completed cleanup"""
        with Session(engine) as session:
            doc = session.exec(
                select(ProcessingDocument).where(ProcessingDocument.doc_id == doc_id)
            ).first()
            
            if doc:
                doc.cleanup_completed = True
                doc.updated_at = datetime.utcnow()
                session.add(doc)
                session.commit()
                return True
            
            return False

    @staticmethod
    async def cleanup_all_orphaned_documents() -> int:
        """Emergency cleanup of all orphaned documents across all users"""
        with Session(engine) as session:
            # Find all documents that are stuck in processing states
            orphaned_docs = session.exec(
                select(ProcessingDocument).where(
                    ProcessingDocument.status.in_([
                        ProcessingStatus.QUEUED.value,
                        ProcessingStatus.UPLOADING.value,
                        ProcessingStatus.PROCESSING.value
                    ])
                )
            ).all()
            
            count = 0
            for doc in orphaned_docs:
                session.delete(doc)
                count += 1
            
            session.commit()
            
            if count > 0:
                print(f"Emergency cleanup: Removed {count} orphaned documents")
            
            return count

# Background cleanup task
async def start_cleanup_task():
    """Start background cleanup task for old and orphaned documents"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            
            # Clean up completed documents older than 24 hours
            await ProcessingService.cleanup_completed_documents(24)
            
            # Mark orphaned documents as error after 2 hours
            await ProcessingService.cleanup_orphaned_documents(2)
            
        except Exception as e:
            print(f"Processing service cleanup error: {e}")

# Global cleanup task reference
_cleanup_task = None

def ensure_cleanup_task():
    """Ensure cleanup task is running"""
    global _cleanup_task
    try:
        if _cleanup_task is None or _cleanup_task.done():
            loop = asyncio.get_running_loop()
            _cleanup_task = loop.create_task(start_cleanup_task())
    except RuntimeError:
        # No running event loop, cleanup task will be started when needed
        pass

# Try to start cleanup task if event loop is available
try:
    ensure_cleanup_task()
except:
    pass
