"""
AI Summary Routes - REST API endpoints for summary management
Mirrors the curation routes with summary-specific functionality
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session
from typing import List, Dict, Optional
import logging

from app.database import get_session
from app.deps import get_current_user
from app.database import User
from app.services.summary_service import summary_service
from app.models import AISummary, SummarySettings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summaries", tags=["AI Summaries"])

@router.get("/", response_model=List[Dict])
async def get_user_summaries(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Get all summaries for the current user"""
    try:
        summaries = await summary_service.get_user_summaries(current_user.id, session)
        return summaries
    except Exception as e:
        logger.error(f"Failed to get summaries for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve summaries: {str(e)}"
        )

@router.post("/generate")
async def generate_summaries(
    request_data: Dict = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Generate or regenerate summaries for the user"""
    try:
        # Extract parameters from request
        generation_type = "manual"
        trigger_event = "manual_refresh"
        
        if request_data:
            generation_type = request_data.get("generationType", "manual")
            trigger_event = request_data.get("triggerEvent", "manual_refresh")
        
        result = await summary_service.generate_summaries_for_user(
            current_user.id, 
            session,
            generation_type=generation_type,
            trigger_event=trigger_event
        )
        
        return result
    except Exception as e:
        logger.error(f"Failed to generate summaries for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summaries: {str(e)}"
        )

@router.get("/settings")
async def get_summary_settings(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Get summary settings for the current user"""
    try:
        settings = await summary_service.get_user_settings(current_user.id, session)
        return settings
    except Exception as e:
        logger.error(f"Failed to get summary settings for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve summary settings: {str(e)}"
        )

@router.put("/settings")
async def update_summary_settings(
    settings_data: Dict,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Update summary settings for the current user"""
    try:
        updated_settings = await summary_service.update_user_settings(
            current_user.id, 
            settings_data, 
            session
        )
        return {
            "success": True,
            "message": "Summary settings updated successfully",
            "settings": updated_settings
        }
    except Exception as e:
        logger.error(f"Failed to update summary settings for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update summary settings: {str(e)}"
        )

@router.get("/status")
async def get_summary_status(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Get the current status of summaries for the user"""
    try:
        status_info = await summary_service.get_summary_status(current_user.id, session)
        return status_info
    except Exception as e:
        logger.error(f"Failed to get summary status for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve summary status: {str(e)}"
        )

@router.post("/custom")
async def create_custom_summary(
    request_data: Dict,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Create a custom summary with user-defined focus and keywords"""
    try:
        # Validate required fields
        title = request_data.get("title", "").strip()
        if not title:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Title is required for custom summary"
            )
        
        # Extract optional parameters
        description = request_data.get("description")
        keywords = request_data.get("keywords", [])
        focus_area = request_data.get("focusArea")
        provider = request_data.get("provider", "ollama")
        model = request_data.get("model", "llama3.2")
        
        # Ensure keywords is a list
        if isinstance(keywords, str):
            keywords = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        
        result = await summary_service.create_custom_summary(
            user_id=current_user.id,
            session=session,
            title=title,
            description=description,
            keywords=keywords,
            focus_area=focus_area,
            provider=provider,
            model=model
        )
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create custom summary for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create custom summary: {str(e)}"
        )

@router.get("/{summary_id}")
async def get_summary_by_id(
    summary_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Get a specific summary by ID"""
    try:
        from sqlmodel import select
        from app.models import AISummary
        
        statement = select(AISummary).where(
            AISummary.id == summary_id,
            AISummary.owner_id == current_user.id
        )
        summary = session.exec(statement).first()
        
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Summary not found"
            )
        
        return {
            'id': summary.id,
            'title': summary.title,
            'description': summary.description,
            'summaryType': summary.summary_type,
            'keywords': summary.focus_keywords,
            'documentCount': summary.document_count,
            'confidence': summary.confidence_score,
            'status': summary.status,
            'lastUpdated': summary.last_updated.isoformat(),
            'autoGenerated': summary.auto_generated,
            'generationMethod': summary.generation_method,
            'contentLength': summary.content_length,
            'content': summary.summary_content
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get summary {summary_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve summary: {str(e)}"
        )

@router.delete("/{summary_id}")
async def delete_summary(
    summary_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Delete a specific summary"""
    try:
        from sqlmodel import select
        from app.models import AISummary, DocumentSummaryMapping
        
        # Find the summary
        statement = select(AISummary).where(
            AISummary.id == summary_id,
            AISummary.owner_id == current_user.id
        )
        summary = session.exec(statement).first()
        
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Summary not found"
            )
        
        # Delete associated mappings
        mapping_statement = select(DocumentSummaryMapping).where(
            DocumentSummaryMapping.summary_id == summary_id,
            DocumentSummaryMapping.owner_id == current_user.id
        )
        mappings = session.exec(mapping_statement).all()
        
        for mapping in mappings:
            session.delete(mapping)
        
        # Delete the summary
        session.delete(summary)
        session.commit()
        
        return {
            "success": True,
            "message": f"Summary '{summary.title}' deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete summary {summary_id} for user {current_user.id}: {e}")
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete summary: {str(e)}"
        )

@router.post("/{summary_id}/refresh")
async def refresh_summary(
    summary_id: str,
    request_data: Dict = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Refresh a specific summary with new AI generation"""
    try:
        from sqlmodel import select
        from app.models import AISummary, Document
        
        # Find the summary
        statement = select(AISummary).where(
            AISummary.id == summary_id,
            AISummary.owner_id == current_user.id
        )
        summary = session.exec(statement).first()
        
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Summary not found"
            )
        
        # Get provider and model from request
        provider = "ollama"
        model = "llama3.2"
        
        if request_data:
            provider = request_data.get("provider", provider)
            model = request_data.get("model", model)
        
        # Get documents for this summary
        doc_statement = select(Document).where(
            Document.id.in_(summary.document_ids),
            Document.owner_id == current_user.id
        )
        documents = session.exec(doc_statement).all()
        
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents found for this summary"
            )
        
        # Regenerate content based on summary type
        if summary.summary_type == "custom":
            # For custom summaries, regenerate with original parameters using specified provider/model
            new_content = await summary_service._generate_custom_summary_content(
                title=summary.title,
                description=summary.description,
                keywords=summary.focus_keywords,
                focus_area=None,  # We don't store focus_area separately
                documents=documents,
                provider=provider,
                model=model
            )
        else:
            # For auto-generated summaries, use the engine
            if summary.summary_type == "individual" and len(documents) == 1:
                summary_data = await summary_service.engine._generate_individual_summary(
                    documents[0], current_user.id
                )
                new_content = summary_data['summary_content'] if summary_data else None
            elif summary.summary_type == "combined":
                summary_data = await summary_service.engine._generate_combined_summary(
                    documents, current_user.id
                )
                new_content = summary_data['summary_content'] if summary_data else None
            elif summary.summary_type == "thematic":
                # For thematic, use the first keyword as theme
                theme = summary.focus_keywords[0] if summary.focus_keywords else "general"
                summary_data = await summary_service.engine._generate_thematic_summary(
                    theme, documents, current_user.id
                )
                new_content = summary_data['summary_content'] if summary_data else None
            else:
                # Fallback to combined summary
                summary_data = await summary_service.engine._generate_combined_summary(
                    documents, current_user.id
                )
                new_content = summary_data['summary_content'] if summary_data else None
        
        if not new_content:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate new summary content"
            )
        
        # Update the summary
        import datetime as dt
        summary.summary_content = new_content
        summary.content_length = len(new_content)
        summary.last_updated = dt.datetime.utcnow()
        summary.status = 'active'
        
        session.commit()
        session.refresh(summary)
        
        # Record generation history
        from app.models import SummaryGenerationHistory
        history = SummaryGenerationHistory(
            owner_id=current_user.id,
            generation_type="refresh",
            trigger_event="manual_refresh",
            documents_processed=len(documents),
            summaries_created=0,
            summaries_updated=1,
            summaries_deleted=0,
            processing_time_ms=0,
            success=True,
            meta_data={
                'summary_id': summary_id,
                'summary_type': summary.summary_type,
                'provider': provider,
                'model': model
            }
        )
        session.add(history)
        session.commit()
        
        return {
            "success": True,
            "message": f"Summary '{summary.title}' refreshed successfully",
            "summary": {
                'id': summary.id,
                'title': summary.title,
                'description': summary.description,
                'summaryType': summary.summary_type,
                'keywords': summary.focus_keywords,
                'documentCount': summary.document_count,
                'confidence': summary.confidence_score,
                'status': summary.status,
                'lastUpdated': summary.last_updated.isoformat(),
                'autoGenerated': summary.auto_generated,
                'generationMethod': summary.generation_method,
                'contentLength': summary.content_length,
                'content': summary.summary_content
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to refresh summary {summary_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh summary: {str(e)}"
        )

@router.get("/{summary_id}/content")
async def get_summary_content(
    summary_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Get the full content of a specific summary"""
    try:
        from sqlmodel import select
        from app.models import AISummary
        
        statement = select(AISummary).where(
            AISummary.id == summary_id,
            AISummary.owner_id == current_user.id
        )
        summary = session.exec(statement).first()
        
        if not summary:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Summary not found"
            )
        
        return {
            "id": summary.id,
            "title": summary.title,
            "content": summary.summary_content or "No content available",
            "contentLength": summary.content_length,
            "lastUpdated": summary.last_updated.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get summary content {summary_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve summary content: {str(e)}"
        )

@router.post("/bulk-delete")
async def bulk_delete_summaries(
    request_data: Dict,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Delete multiple summaries at once"""
    try:
        summary_ids = request_data.get("summaryIds", [])
        
        if not summary_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No summary IDs provided"
            )
        
        from sqlmodel import select
        from app.models import AISummary, DocumentSummaryMapping
        
        deleted_count = 0
        deleted_titles = []
        
        for summary_id in summary_ids:
            # Find the summary
            statement = select(AISummary).where(
                AISummary.id == summary_id,
                AISummary.owner_id == current_user.id
            )
            summary = session.exec(statement).first()
            
            if summary:
                # Delete associated mappings
                mapping_statement = select(DocumentSummaryMapping).where(
                    DocumentSummaryMapping.summary_id == summary_id,
                    DocumentSummaryMapping.owner_id == current_user.id
                )
                mappings = session.exec(mapping_statement).all()
                
                for mapping in mappings:
                    session.delete(mapping)
                
                # Delete the summary
                deleted_titles.append(summary.title)
                session.delete(summary)
                deleted_count += 1
        
        session.commit()
        
        return {
            "success": True,
            "message": f"Successfully deleted {deleted_count} summaries",
            "deletedCount": deleted_count,
            "deletedTitles": deleted_titles
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to bulk delete summaries for user {current_user.id}: {e}")
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete summaries: {str(e)}"
        )

@router.post("/archive-stale")
async def archive_stale_summaries(
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Archive all stale summaries for the user"""
    try:
        from sqlmodel import select
        from app.models import AISummary
        
        # Find all stale summaries
        statement = select(AISummary).where(
            AISummary.owner_id == current_user.id,
            AISummary.status == 'stale'
        )
        stale_summaries = session.exec(statement).all()
        
        archived_count = 0
        archived_titles = []
        
        for summary in stale_summaries:
            summary.status = 'archived'
            archived_titles.append(summary.title)
            archived_count += 1
        
        session.commit()
        
        return {
            "success": True,
            "message": f"Successfully archived {archived_count} stale summaries",
            "archivedCount": archived_count,
            "archivedTitles": archived_titles
        }
    except Exception as e:
        logger.error(f"Failed to archive stale summaries for user {current_user.id}: {e}")
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to archive stale summaries: {str(e)}"
        )

# Document lifecycle integration endpoints
@router.post("/handle-document-addition")
async def handle_document_addition(
    request_data: Dict,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Handle summary updates when documents are added"""
    try:
        document_ids = request_data.get("documentIds", [])
        
        if not document_ids:
            return {
                "success": True,
                "message": "No documents to process",
                "action": "no_action"
            }
        
        result = await summary_service.handle_document_addition(
            current_user.id, 
            document_ids, 
            session
        )
        
        return result
    except Exception as e:
        logger.error(f"Failed to handle document addition for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to handle document addition: {str(e)}"
        )

@router.post("/handle-document-deletion")
async def handle_document_deletion(
    request_data: Dict,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Handle summary updates when documents are deleted"""
    try:
        document_ids = request_data.get("documentIds", [])
        
        if not document_ids:
            return {
                "success": True,
                "message": "No documents to process",
                "stats": {
                    "summariesUpdated": 0,
                    "summariesDeleted": 0,
                    "documentsRemoved": 0
                }
            }
        
        result = await summary_service.handle_document_deletion(
            current_user.id, 
            document_ids, 
            session
        )
        
        return result
    except Exception as e:
        logger.error(f"Failed to handle document deletion for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to handle document deletion: {str(e)}"
        )
