from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from typing import List
from app.database import get_session
from app.models import Document
from app.deps import get_current_user, get_current_user_with_tenant, get_current_admin_user, get_current_tenant_id_dependency
from app.database import User
from app.utils import qdrant_store
from app.services.processing_service import ProcessingService, ProcessingStatus
from app.services.curation_service import CurationService
from app.utils.s3_storage import cleanup_file_resources
import os
from pathlib import Path

router = APIRouter(tags=["documents"])

@router.get("/documents")
async def get_user_documents(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
) -> List[dict]:
    """Get documents accessible to the user within their tenant"""
    
    # Use user's tenant_id directly from the user object
    tenant_id = user.tenant_id
    
    # Use DocumentAccessService to get only documents the user can access
    from app.services.document_access_service import DocumentAccessService
    accessible_documents = DocumentAccessService.get_user_accessible_documents(
        session, str(user.id), tenant_id
    )
    
    result = []
    for doc in accessible_documents:
        # Use the original filename from the document record instead of extracting from path
        # This preserves the original filename without UUID prefixes
        filename = doc.filename if hasattr(doc, 'filename') and doc.filename else os.path.basename(doc.path)
        
        # Use stored file size from database, fallback to file system check if needed
        file_size = doc.file_size if hasattr(doc, 'file_size') and doc.file_size else 0
        
        # If no stored file size, try to get it from file system (for backward compatibility)
        if file_size == 0 and os.path.exists(doc.path):
            try:
                file_size = os.path.getsize(doc.path)
                print(f"📊 Fallback file size calculation for {filename}: {file_size} bytes")
            except Exception as e:
                print(f"❌ Error getting file size for {doc.path}: {e}")
                file_size = 0
        
        # Use stored content type or determine from extension
        file_type = doc.content_type if hasattr(doc, 'content_type') and doc.content_type else None
        
        if not file_type:
            # Fallback to extension-based detection
            file_ext = Path(filename).suffix.lower()
            file_type = "application/pdf" if file_ext == ".pdf" else \
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if file_ext == ".docx" else \
                       "text/plain" if file_ext == ".txt" else \
                       "application/rtf" if file_ext == ".rtf" else \
                       "application/octet-stream"
        
        result.append({
            "id": doc.id,
            "name": filename,
            "size": file_size,
            "type": file_type,
            "uploadDate": doc.created_at.strftime("%Y-%m-%d"),
            "status": "completed",
            "progress": 100,
            "language": "English",  # Default, could be enhanced
            "themes": doc.tags or [],
            "keywords": doc.tags or [],  # Using tags as keywords for now
            "demographics": doc.demo_tags or [],
            "summary": doc.summary,
            "keyInsights": [doc.insight] if doc.insight else [],  # Frontend expects array of insights
            "mainTopics": doc.tags or [],  # Map tags to mainTopics for frontend
            "sentiment": "neutral",  # Default sentiment
            "readingLevel": "intermediate",  # Default reading level
            "content": "",  # Add empty content field
            "path": doc.path
        })
    
    return result

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    user: User = Depends(get_current_user_with_tenant),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Delete a document and all associated data (file, database record, RAG embeddings)
    
    This endpoint handles both completed documents and documents currently being processed.
    For processing documents, it will cancel the processing and clean up resources.
    """
    
    # First, check if this is a processing document
    processing_doc = None
    try:
        # Try to find by document ID first
        processing_doc = await ProcessingService.get_processing_document(document_id, str(user.id))
        
        # If not found by ID, check if this is a reconnected document ID pattern
        if not processing_doc and document_id.startswith("reconnect_"):
            # Extract batch ID from reconnect pattern: reconnect_batchId_index
            parts = document_id.split("_")
            if len(parts) >= 3:
                batch_id = "_".join(parts[1:-1])  # Everything between "reconnect" and the last index
                print(f"Extracted batch ID from reconnect document: {batch_id}")
                
                # Find processing document by batch ID
                processing_doc = await ProcessingService.find_processing_document_by_batch_id(batch_id, str(user.id))
                if processing_doc:
                    print(f"Found processing document by batch ID: {processing_doc.doc_id} ({processing_doc.filename})")
                else:
                    print(f"No processing document found for batch ID: {batch_id}, user: {str(user.id)}")
                    # Debug: List all processing documents for this user
                    all_docs = await ProcessingService.get_user_processing_documents(str(user.id))
                    print(f"All processing documents for user {str(user.id)}: {len(all_docs)}")
                    for doc in all_docs:
                        print(f"  - {doc.doc_id}: batch={doc.batch_id}, filename={doc.filename}, status={doc.status}")
        
        # If still not found, try to find by filename pattern
        if not processing_doc:
            # Extract potential filename from document_id (common pattern: filename_uuid)
            if "_" in document_id and not document_id.startswith("reconnect_"):
                potential_filename = document_id.split("_")[0]
                processing_doc = await ProcessingService.find_processing_document_by_filename(potential_filename, str(user.id))
            
            # Also try the full document_id as filename
            if not processing_doc:
                processing_doc = await ProcessingService.find_processing_document_by_filename(document_id, str(user.id))
        
    except Exception as e:
        print(f"Error checking processing documents: {e}")
    
    # Handle processing document
    if processing_doc and processing_doc.owner_id == str(user.id):
        print(f"Found processing document: {processing_doc.doc_id} ({processing_doc.filename})")
        
        # Request cancellation if still processing
        if processing_doc.status in [ProcessingStatus.QUEUED.value, ProcessingStatus.UPLOADING.value, ProcessingStatus.PROCESSING.value]:
            cancellation_success = await ProcessingService.request_cancellation(processing_doc.doc_id, str(user.id))
            print(f"Cancellation requested for {processing_doc.doc_id}: {cancellation_success}")
        
        # Clean up comprehensive file resources (S3, local temp files)
        cleanup_results = await cleanup_file_resources(
            s3_key=processing_doc.s3_key,
            local_path=processing_doc.file_path,
            user_id=str(user.id),
            batch_id=processing_doc.batch_id
        )
        
        # Remove from processing tracker
        removed_doc = await ProcessingService.remove_processing_document(processing_doc.doc_id, str(user.id))
        
        # Try to clean up any partial vector embeddings
        vector_cleanup_success = False
        try:
            qdrant_store.delete_document_vectors(str(user.id), processing_doc.doc_id, processing_doc.filename)
            vector_cleanup_success = True
            print(f"Cleaned up partial vectors for processing document {processing_doc.doc_id}")
        except Exception as e:
            print(f"No vectors to clean up for processing document {processing_doc.doc_id}: {e}")
        
        # If this was a reconnected document, also try to clean up the entire batch
        # to prevent other stuck documents in the same batch
        batch_cleanup_count = 0
        if document_id.startswith("reconnect_"):
            try:
                batch_cleanup_count = await ProcessingService.cleanup_batch(processing_doc.batch_id, str(user.id))
                print(f"Cleaned up entire batch {processing_doc.batch_id}: {batch_cleanup_count} documents")
            except Exception as e:
                print(f"Failed to cleanup batch {processing_doc.batch_id}: {e}")
        
        return {
            "message": "Processing document cancelled and cleaned up successfully",
            "document_id": processing_doc.doc_id,
            "filename": processing_doc.filename,
            "was_processing": True,
            "status": processing_doc.status,
            "batch_cleanup_count": batch_cleanup_count,
            "deleted_items": {
                "processing_state": removed_doc is not None,
                "physical_file": cleanup_results.get("local_deleted", False),
                "s3_file": cleanup_results.get("s3_deleted", False),
                "batch_cleaned": cleanup_results.get("batch_cleaned", False),
                "partial_embeddings": vector_cleanup_success,
                "database_record": False  # Processing docs aren't in DB yet
            }
        }
    
    # Handle completed document with proper tenant isolation
    statement = select(Document).where(
        Document.id == document_id,
        Document.tenant_id == tenant_id
    )
    document = session.exec(statement).first()
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found in your tenant")
    
    # Use DocumentAccessService to check if user can access this document
    from app.services.document_access_service import DocumentAccessService
    can_access = DocumentAccessService.can_user_access_document(
        session, str(user.id), document_id, tenant_id, "manage"
    )
    
    if not can_access:
        raise HTTPException(status_code=403, detail="Permission denied. You don't have access to this document.")
    
    # Extract filename for vector store cleanup
    filename = os.path.basename(document.path)
    
    try:
        # Delete RAG embeddings from vector store
        # This removes all vectors associated with this document
        qdrant_store.delete_document_vectors(str(user.id), document_id, filename)
        print(f"Deleted vectors for document {document_id} ({filename})")
    except Exception as e:
        print(f"Failed to delete vectors for document {document_id}: {e}")
        # Continue with deletion even if vector cleanup fails
    
    # Delete the physical file if it exists
    if os.path.exists(document.path):
        try:
            os.remove(document.path)
            print(f"Deleted file: {document.path}")
        except Exception as e:
            print(f"Failed to delete file {document.path}: {e}")
    
    # Delete document access records first (to avoid foreign key constraint)
    from app.models_document_access import DocumentAccess, DocumentAccessLog
    
    doc_access_records = session.exec(
        select(DocumentAccess).where(DocumentAccess.document_id == document_id)
    ).all()
    for access in doc_access_records:
        session.delete(access)
    
    # Delete document access logs
    doc_access_logs = session.exec(
        select(DocumentAccessLog).where(DocumentAccessLog.document_id == document_id)
    ).all()
    for log in doc_access_logs:
        session.delete(log)
    
    # Delete document curation mappings
    from app.models import DocumentCurationMapping, DocumentSummaryMapping
    doc_curation_mappings = session.exec(
        select(DocumentCurationMapping).where(DocumentCurationMapping.document_id == document_id)
    ).all()
    for mapping in doc_curation_mappings:
        session.delete(mapping)
    
    # Delete document summary mappings
    doc_summary_mappings = session.exec(
        select(DocumentSummaryMapping).where(DocumentSummaryMapping.document_id == document_id)
    ).all()
    for mapping in doc_summary_mappings:
        session.delete(mapping)
    
    # Delete from database (this removes all metadata, tags, summary, insights)
    session.delete(document)
    session.commit()
    
    # Trigger curation update for document deletion
    try:
        curation_service = CurationService()
        await curation_service.handle_document_deletion(
            str(user.id), [document_id], session
        )
        print(f"Triggered curation update for document deletion: {document_id}")
    except Exception as e:
        print(f"Failed to update curations after document deletion: {e}")
        # Don't fail the deletion if curation update fails
    
    return {
        "message": "Document and all associated data deleted successfully",
        "document_id": document_id,
        "filename": filename,
        "was_processing": False,
        "deleted_items": {
            "database_record": True,
            "physical_file": os.path.exists(document.path) == False,
            "rag_embeddings": True,
            "tags": True,
            "summary": True,
            "demographics": True,
            "insights": True,
            "curations_updated": True
        }
    }
