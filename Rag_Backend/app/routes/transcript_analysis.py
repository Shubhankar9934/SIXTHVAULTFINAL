"""
Transcript Analysis API Routes

Provides endpoints for transcript analysis functionality including analysis creation,
retrieval, updating, deletion, and CSV export.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from app.deps import get_current_user
from app.database import User
from app.services.transcript_analysis_service import transcript_analysis_service

router = APIRouter(prefix="/transcript-analysis", tags=["Transcript Analysis"])

# Request/Response Models
class TranscriptAnalysisRequest(BaseModel):
    transcript_text: str
    title: Optional[str] = None
    document_id: Optional[str] = None
    source_type: str = "manual"
    provider: str = "gemini"
    model: str = "gemini-1.5-flash"

class TranscriptAnalysisUpdate(BaseModel):
    title: Optional[str] = None
    company_name: Optional[str] = None
    overall_sentiment: Optional[str] = None
    agreement_to_share: Optional[bool] = None
    satisfaction_reasons: Optional[List[str]] = None
    improvement_areas: Optional[List[str]] = None
    issues_and_stories: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    is_archived: Optional[bool] = None

class TranscriptAnalysisResponse(BaseModel):
    id: str
    title: str
    company_name: Optional[str]
    overall_sentiment: Optional[str]
    agreement_to_share: Optional[bool]
    satisfaction_reasons: List[str]
    improvement_areas: List[str]
    issues_and_stories: List[str]
    raw_transcript: Optional[str]
    analysis_content: Optional[str]
    source_type: str
    document_id: Optional[str]
    status: str
    confidence_score: Optional[float]
    created_at: str
    updated_at: str
    processing_time_ms: Optional[int]
    provider_used: Optional[str]
    model_used: Optional[str]
    tags: List[str]
    is_archived: bool

class TranscriptAnalysisListItem(BaseModel):
    id: str
    title: str
    company_name: Optional[str]
    overall_sentiment: Optional[str]
    agreement_to_share: Optional[bool]
    source_type: str
    status: str
    confidence_score: Optional[float]
    created_at: str
    processing_time_ms: Optional[int]
    provider_used: Optional[str]
    model_used: Optional[str]

class TranscriptAnalysisStats(BaseModel):
    total_analyses: int
    completed_analyses: int
    archived_analyses: int
    recent_analyses: int
    average_confidence_score: float
    success_rate: float

@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_transcript(
    request: TranscriptAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Analyze a transcript using the new template and save to database
    """
    try:
        result = await transcript_analysis_service.analyze_transcript(
            transcript_text=request.transcript_text,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            title=request.title,
            document_id=request.document_id,
            source_type=request.source_type,
            provider=request.provider,
            model=request.model
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.get("/", response_model=List[TranscriptAnalysisListItem])
async def get_transcript_analyses(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    include_archived: bool = Query(False),
    current_user: User = Depends(get_current_user)
):
    """
    Get transcript analyses for the current user
    """
    try:
        analyses = transcript_analysis_service.get_user_analyses(
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            limit=limit,
            offset=offset,
            include_archived=include_archived
        )
        
        return analyses
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analyses: {str(e)}")

@router.get("/{analysis_id}", response_model=TranscriptAnalysisResponse)
async def get_transcript_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get a specific transcript analysis by ID
    """
    try:
        analysis = transcript_analysis_service.get_analysis_by_id(
            analysis_id=analysis_id,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve analysis: {str(e)}")

@router.put("/{analysis_id}", response_model=TranscriptAnalysisResponse)
async def update_transcript_analysis(
    analysis_id: str,
    update_data: TranscriptAnalysisUpdate,
    current_user: User = Depends(get_current_user)
):
    """
    Update an existing transcript analysis
    """
    try:
        # Convert Pydantic model to dict, excluding unset fields
        update_dict = update_data.dict(exclude_unset=True)
        
        updated_analysis = transcript_analysis_service.update_analysis(
            analysis_id=analysis_id,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            **update_dict
        )
        
        if not updated_analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return updated_analysis
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update analysis: {str(e)}")

@router.delete("/{analysis_id}")
async def delete_transcript_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Delete a transcript analysis
    """
    try:
        success = transcript_analysis_service.delete_analysis(
            analysis_id=analysis_id,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Analysis not found")
        
        return {"message": "Analysis deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete analysis: {str(e)}")

@router.get("/stats/summary", response_model=TranscriptAnalysisStats)
async def get_transcript_analysis_stats(
    current_user: User = Depends(get_current_user)
):
    """
    Get statistics about user's transcript analyses
    """
    try:
        stats = transcript_analysis_service.get_analysis_stats(
            user_id=current_user.id,
            tenant_id=current_user.tenant_id
        )
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve stats: {str(e)}")

@router.post("/export/csv")
async def export_transcript_analyses_csv(
    analysis_ids: Optional[List[str]] = None,
    include_archived: bool = Query(False),
    current_user: User = Depends(get_current_user)
):
    """
    Export transcript analyses to CSV format
    """
    try:
        csv_response = transcript_analysis_service.export_to_csv(
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            analysis_ids=analysis_ids,
            include_archived=include_archived
        )
        
        return csv_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export CSV: {str(e)}")

@router.post("/bulk-archive")
async def bulk_archive_analyses(
    analysis_ids: List[str],
    current_user: User = Depends(get_current_user)
):
    """
    Archive multiple transcript analyses at once
    """
    try:
        results = []
        for analysis_id in analysis_ids:
            result = transcript_analysis_service.update_analysis(
                analysis_id=analysis_id,
                user_id=current_user.id,
                tenant_id=current_user.tenant_id,
                is_archived=True
            )
            if result:
                results.append(analysis_id)
        
        return {
            "message": f"Successfully archived {len(results)} analyses",
            "archived_ids": results,
            "failed_ids": [aid for aid in analysis_ids if aid not in results]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to bulk archive: {str(e)}")

@router.post("/bulk-delete")
async def bulk_delete_analyses(
    analysis_ids: List[str],
    current_user: User = Depends(get_current_user)
):
    """
    Delete multiple transcript analyses at once
    """
    try:
        results = []
        for analysis_id in analysis_ids:
            success = transcript_analysis_service.delete_analysis(
                analysis_id=analysis_id,
                user_id=current_user.id,
                tenant_id=current_user.tenant_id
            )
            if success:
                results.append(analysis_id)
        
        return {
            "message": f"Successfully deleted {len(results)} analyses",
            "deleted_ids": results,
            "failed_ids": [aid for aid in analysis_ids if aid not in results]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to bulk delete: {str(e)}")

# Analysis from Document endpoint
@router.post("/analyze-from-document/{document_id}")
async def analyze_transcript_from_document(
    document_id: str,
    title: Optional[str] = None,
    provider: str = "gemini",
    model: str = "gemini-1.5-flash",
    current_user: User = Depends(get_current_user)
):
    """
    Analyze a transcript from an existing document
    """
    try:
        # Import here to avoid circular imports
        from app.models import Document
        from app.database import engine
        from sqlmodel import Session, select, and_
        
        # Get the document content
        with Session(engine) as session:
            doc_query = select(Document).where(
                and_(
                    Document.id == document_id,
                    Document.owner_id == current_user.id,
                    Document.tenant_id == current_user.tenant_id if current_user.tenant_id else True
                )
            )
            document = session.exec(doc_query).first()
            
            if not document:
                raise HTTPException(status_code=404, detail="Document not found")
            
            # Read document content
            try:
                with open(document.path, 'r', encoding='utf-8') as f:
                    transcript_text = f.read()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read document: {str(e)}")
        
        # Analyze the transcript
        result = await transcript_analysis_service.analyze_transcript(
            transcript_text=transcript_text,
            user_id=current_user.id,
            tenant_id=current_user.tenant_id,
            title=title or f"Analysis of {document.filename}",
            document_id=document_id,
            source_type="document",
            provider=provider,
            model=model
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Search analyses endpoint
@router.get("/search/{query}")
async def search_transcript_analyses(
    query: str,
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user)
):
    """
    Search transcript analyses by content
    """
    try:
        # Import here to avoid circular imports
        from sqlmodel import Session, select, and_, or_
        from app.database import engine
        from app.models import TranscriptAnalysis
        
        with Session(engine) as session:
            # Search in title, company name, sentiment, and analysis content
            search_query = select(TranscriptAnalysis).where(
                and_(
                    TranscriptAnalysis.owner_id == current_user.id,
                    TranscriptAnalysis.tenant_id == current_user.tenant_id if current_user.tenant_id else True,
                    or_(
                        TranscriptAnalysis.title.ilike(f"%{query}%"),
                        TranscriptAnalysis.company_name.ilike(f"%{query}%"),
                        TranscriptAnalysis.overall_sentiment.ilike(f"%{query}%"),
                        TranscriptAnalysis.analysis_content.ilike(f"%{query}%")
                    )
                )
            ).order_by(TranscriptAnalysis.created_at.desc()).limit(limit)
            
            analyses = session.exec(search_query).all()
            
            # Convert to list format
            results = []
            for analysis in analyses:
                results.append({
                    'id': analysis.id,
                    'title': analysis.title,
                    'company_name': analysis.company_name,
                    'overall_sentiment': analysis.overall_sentiment,
                    'agreement_to_share': analysis.agreement_to_share,
                    'source_type': analysis.source_type,
                    'status': analysis.status,
                    'confidence_score': analysis.confidence_score,
                    'created_at': analysis.created_at.isoformat(),
                    'processing_time_ms': analysis.processing_time_ms,
                    'provider_used': analysis.provider_used,
                    'model_used': analysis.model_used
                })
            
            return {
                'query': query,
                'total_results': len(results),
                'results': results
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
