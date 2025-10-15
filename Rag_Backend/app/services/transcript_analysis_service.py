"""
Transcript Analysis Service

Handles transcript analysis using the new template, database saving, and CSV export functionality.
"""

import time
import csv
import io
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sqlmodel import Session, select, and_
from fastapi.responses import StreamingResponse

from app.database import get_session, engine
from app.models import TranscriptAnalysis, User
from app.services.summarise import _enhanced_summarizer
import logging

logger = logging.getLogger(__name__)

class TranscriptAnalysisService:
    """Service for handling transcript analysis with database storage and CSV export"""
    
    def __init__(self):
        pass
    
    async def analyze_transcript(
        self,
        transcript_text: str,
        user_id: str,
        tenant_id: Optional[str] = None,
        title: Optional[str] = None,
        document_id: Optional[str] = None,
        source_type: str = "manual",
        provider: str = "gemini",
        model: str = "gemini-1.5-flash"
    ) -> Dict:
        """
        Analyze a transcript using the new template and save to database
        
        Args:
            transcript_text: The raw transcript text to analyze
            user_id: ID of the user performing the analysis
            tenant_id: Tenant ID for isolation
            title: Optional title for the analysis
            document_id: Optional link to source document
            source_type: Source of the transcript (manual, document, upload)
            provider: AI provider to use
            model: AI model to use
            
        Returns:
            Dictionary with analysis results and database record info
        """
        start_time = time.time()
        
        try:
            # Generate analysis using the transcript template
            analysis_content = await _enhanced_summarizer.make_summary(
                transcript_text, 
                force_model=provider
            )
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Parse the structured analysis content
            parsed_data = self._parse_analysis_content(analysis_content)
            
            # Create database record
            with Session(engine) as session:
                analysis = TranscriptAnalysis(
                    owner_id=user_id,
                    tenant_id=tenant_id,
                    title=title or f"Transcript Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    document_id=document_id,
                    source_type=source_type,
                    company_name=parsed_data.get('company_name'),
                    overall_sentiment=parsed_data.get('overall_sentiment'),
                    agreement_to_share=parsed_data.get('agreement_to_share'),
                    satisfaction_reasons=parsed_data.get('satisfaction_reasons', []),
                    improvement_areas=parsed_data.get('improvement_areas', []),
                    issues_and_stories=parsed_data.get('issues_and_stories', []),
                    raw_transcript=transcript_text,
                    analysis_content=analysis_content,
                    provider_used=provider,
                    model_used=model,
                    processing_time_ms=processing_time,
                    confidence_score=self._calculate_confidence_score(parsed_data),
                    status="completed"
                )
                
                session.add(analysis)
                session.commit()
                session.refresh(analysis)
                
                logger.info(f"Transcript analysis saved with ID: {analysis.id}")
                
                return {
                    'success': True,
                    'analysis_id': analysis.id,
                    'processing_time_ms': processing_time,
                    'parsed_data': parsed_data,
                    'analysis_content': analysis_content,
                    'confidence_score': analysis.confidence_score
                }
                
        except Exception as e:
            logger.error(f"Transcript analysis failed: {e}")
            processing_time = int((time.time() - start_time) * 1000)
            
            # Save error record
            with Session(engine) as session:
                analysis = TranscriptAnalysis(
                    owner_id=user_id,
                    tenant_id=tenant_id,
                    title=title or f"Failed Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    document_id=document_id,
                    source_type=source_type,
                    raw_transcript=transcript_text,
                    provider_used=provider,
                    model_used=model,
                    processing_time_ms=processing_time,
                    status="error"
                )
                
                session.add(analysis)
                session.commit()
                session.refresh(analysis)
            
            return {
                'success': False,
                'error': str(e),
                'analysis_id': analysis.id,
                'processing_time_ms': processing_time
            }
    
    def _parse_analysis_content(self, analysis_content: str) -> Dict:
        """
        Parse the structured analysis content to extract individual components
        """
        parsed_data = {}
        
        try:
            # Extract Company Name
            company_match = re.search(r'\*\*Company Name:\*\*\s*(.+)', analysis_content)
            if company_match:
                parsed_data['company_name'] = company_match.group(1).strip()
            
            # Extract Overall Sentiment
            sentiment_match = re.search(r'\*\*Overall Sentiment:\*\*\s*(.+)', analysis_content)
            if sentiment_match:
                parsed_data['overall_sentiment'] = sentiment_match.group(1).strip()
            
            # Extract Agreement to Share Feedback
            agreement_match = re.search(r'\*\*Agreement to Share Feedback:\*\*\s*(.+)', analysis_content)
            if agreement_match:
                agreement_text = agreement_match.group(1).strip().lower()
                parsed_data['agreement_to_share'] = agreement_text.startswith('yes')
            
            # Extract Satisfaction Reasons
            satisfaction_section = re.search(
                r'\*\*Reasons for Satisfaction \(What Makes Them Happy\):\*\*\s*\n(.*?)\n\*\*Areas for Improvement',
                analysis_content,
                re.DOTALL
            )
            if satisfaction_section:
                reasons = self._extract_bullet_points(satisfaction_section.group(1))
                parsed_data['satisfaction_reasons'] = reasons
            
            # Extract Areas for Improvement
            improvement_section = re.search(
                r'\*\*Areas for Improvement \(What Can Be Better\):\*\*\s*\n(.*?)\n\*\*Summary of Issues',
                analysis_content,
                re.DOTALL
            )
            if improvement_section:
                areas = self._extract_bullet_points(improvement_section.group(1))
                parsed_data['improvement_areas'] = areas
            
            # Extract Issues and Stories
            issues_section = re.search(
                r'\*\*Summary of Issues and Stories Mentioned:\*\*\s*\n(.*?)(?:\n\*\*|$)',
                analysis_content,
                re.DOTALL
            )
            if issues_section:
                issues = self._extract_bullet_points(issues_section.group(1))
                parsed_data['issues_and_stories'] = issues
                
        except Exception as e:
            logger.warning(f"Error parsing analysis content: {e}")
        
        return parsed_data
    
    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text"""
        lines = text.strip().split('\n')
        bullet_points = []
        
        for line in lines:
            line = line.strip()
            # Match various bullet point formats
            if re.match(r'^[\*\-\+•]\s+', line):
                bullet_points.append(re.sub(r'^[\*\-\+•]\s+', '', line).strip())
        
        return bullet_points
    
    def _calculate_confidence_score(self, parsed_data: Dict) -> float:
        """Calculate confidence score based on extracted data completeness"""
        total_fields = 6  # company_name, sentiment, agreement, satisfaction, improvement, issues
        filled_fields = 0
        
        if parsed_data.get('company_name'):
            filled_fields += 1
        if parsed_data.get('overall_sentiment'):
            filled_fields += 1
        if parsed_data.get('agreement_to_share') is not None:
            filled_fields += 1
        if parsed_data.get('satisfaction_reasons'):
            filled_fields += 1
        if parsed_data.get('improvement_areas'):
            filled_fields += 1
        if parsed_data.get('issues_and_stories'):
            filled_fields += 1
        
        return round(filled_fields / total_fields, 2)
    
    def get_user_analyses(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = False
    ) -> List[Dict]:
        """Get transcript analyses for a user"""
        
        with Session(engine) as session:
            query = select(TranscriptAnalysis).where(
                and_(
                    TranscriptAnalysis.owner_id == user_id,
                    TranscriptAnalysis.tenant_id == tenant_id if tenant_id else True
                )
            )
            
            if not include_archived:
                query = query.where(TranscriptAnalysis.is_archived == False)
            
            query = query.order_by(TranscriptAnalysis.created_at.desc()).offset(offset).limit(limit)
            
            analyses = session.exec(query).all()
            
            result = []
            for analysis in analyses:
                result.append({
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
            
            return result
    
    def get_analysis_by_id(
        self,
        analysis_id: str,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Optional[Dict]:
        """Get a specific transcript analysis"""
        
        with Session(engine) as session:
            query = select(TranscriptAnalysis).where(
                and_(
                    TranscriptAnalysis.id == analysis_id,
                    TranscriptAnalysis.owner_id == user_id,
                    TranscriptAnalysis.tenant_id == tenant_id if tenant_id else True
                )
            )
            
            analysis = session.exec(query).first()
            
            if not analysis:
                return None
            
            return {
                'id': analysis.id,
                'title': analysis.title,
                'company_name': analysis.company_name,
                'overall_sentiment': analysis.overall_sentiment,
                'agreement_to_share': analysis.agreement_to_share,
                'satisfaction_reasons': analysis.satisfaction_reasons,
                'improvement_areas': analysis.improvement_areas,
                'issues_and_stories': analysis.issues_and_stories,
                'raw_transcript': analysis.raw_transcript,
                'analysis_content': analysis.analysis_content,
                'source_type': analysis.source_type,
                'document_id': analysis.document_id,
                'status': analysis.status,
                'confidence_score': analysis.confidence_score,
                'created_at': analysis.created_at.isoformat(),
                'updated_at': analysis.updated_at.isoformat(),
                'processing_time_ms': analysis.processing_time_ms,
                'provider_used': analysis.provider_used,
                'model_used': analysis.model_used,
                'tags': analysis.tags,
                'is_archived': analysis.is_archived
            }
    
    def export_to_csv(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        analysis_ids: Optional[List[str]] = None,
        include_archived: bool = False
    ) -> StreamingResponse:
        """
        Export transcript analyses to CSV format
        
        Args:
            user_id: User ID
            tenant_id: Tenant ID for isolation
            analysis_ids: Optional list of specific analysis IDs to export
            include_archived: Whether to include archived analyses
            
        Returns:
            StreamingResponse with CSV content
        """
        
        with Session(engine) as session:
            query = select(TranscriptAnalysis).where(
                and_(
                    TranscriptAnalysis.owner_id == user_id,
                    TranscriptAnalysis.tenant_id == tenant_id if tenant_id else True
                )
            )
            
            if analysis_ids:
                query = query.where(TranscriptAnalysis.id.in_(analysis_ids))
            
            if not include_archived:
                query = query.where(TranscriptAnalysis.is_archived == False)
            
            query = query.order_by(TranscriptAnalysis.created_at.desc())
            
            analyses = session.exec(query).all()
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        headers = [
            'ID',
            'Title',
            'Company Name',
            'Overall Sentiment',
            'Agreement to Share',
            'Satisfaction Reasons',
            'Areas for Improvement',
            'Issues and Stories',
            'Source Type',
            'Status',
            'Confidence Score',
            'Created At',
            'Processing Time (ms)',
            'Provider Used',
            'Model Used',
            'Tags',
            'Raw Transcript'
        ]
        writer.writerow(headers)
        
        # Write data rows
        for analysis in analyses:
            # Format list fields as semicolon-separated strings
            satisfaction_reasons = '; '.join(analysis.satisfaction_reasons) if analysis.satisfaction_reasons else ''
            improvement_areas = '; '.join(analysis.improvement_areas) if analysis.improvement_areas else ''
            issues_and_stories = '; '.join(analysis.issues_and_stories) if analysis.issues_and_stories else ''
            tags = '; '.join(analysis.tags) if analysis.tags else ''
            
            row = [
                analysis.id,
                analysis.title,
                analysis.company_name or '',
                analysis.overall_sentiment or '',
                'Yes' if analysis.agreement_to_share else 'No' if analysis.agreement_to_share is False else '',
                satisfaction_reasons,
                improvement_areas,
                issues_and_stories,
                analysis.source_type,
                analysis.status,
                analysis.confidence_score or '',
                analysis.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                analysis.processing_time_ms or '',
                analysis.provider_used or '',
                analysis.model_used or '',
                tags,
                analysis.raw_transcript or ''
            ]
            writer.writerow(row)
        
        # Prepare response
        output.seek(0)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"transcript_analyses_{timestamp}.csv"
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    def update_analysis(
        self,
        analysis_id: str,
        user_id: str,
        tenant_id: Optional[str] = None,
        **update_data
    ) -> Optional[Dict]:
        """Update an existing transcript analysis"""
        
        with Session(engine) as session:
            analysis = session.exec(
                select(TranscriptAnalysis).where(
                    and_(
                        TranscriptAnalysis.id == analysis_id,
                        TranscriptAnalysis.owner_id == user_id,
                        TranscriptAnalysis.tenant_id == tenant_id if tenant_id else True
                    )
                )
            ).first()
            
            if not analysis:
                return None
            
            # Update allowed fields
            allowed_fields = [
                'title', 'company_name', 'overall_sentiment', 'agreement_to_share',
                'satisfaction_reasons', 'improvement_areas', 'issues_and_stories',
                'tags', 'is_archived'
            ]
            
            for field, value in update_data.items():
                if field in allowed_fields and hasattr(analysis, field):
                    setattr(analysis, field, value)
            
            analysis.updated_at = datetime.utcnow()
            
            session.add(analysis)
            session.commit()
            session.refresh(analysis)
            
            return self.get_analysis_by_id(analysis_id, user_id, tenant_id)
    
    def delete_analysis(
        self,
        analysis_id: str,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> bool:
        """Delete a transcript analysis"""
        
        with Session(engine) as session:
            analysis = session.exec(
                select(TranscriptAnalysis).where(
                    and_(
                        TranscriptAnalysis.id == analysis_id,
                        TranscriptAnalysis.owner_id == user_id,
                        TranscriptAnalysis.tenant_id == tenant_id if tenant_id else True
                    )
                )
            ).first()
            
            if analysis:
                session.delete(analysis)
                session.commit()
                return True
            
            return False
    
    def get_analysis_stats(
        self,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Dict:
        """Get statistics about user's transcript analyses"""
        
        with Session(engine) as session:
            # Total analyses
            total_query = select(TranscriptAnalysis).where(
                and_(
                    TranscriptAnalysis.owner_id == user_id,
                    TranscriptAnalysis.tenant_id == tenant_id if tenant_id else True
                )
            )
            total_analyses = len(session.exec(total_query).all())
            
            # Completed analyses
            completed_query = total_query.where(TranscriptAnalysis.status == 'completed')
            completed_analyses = len(session.exec(completed_query).all())
            
            # Archived analyses
            archived_query = total_query.where(TranscriptAnalysis.is_archived == True)
            archived_analyses = len(session.exec(archived_query).all())
            
            # Recent analyses (last 7 days)
            from datetime import timedelta
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_query = total_query.where(TranscriptAnalysis.created_at >= week_ago)
            recent_analyses = len(session.exec(recent_query).all())
            
            # Average confidence score
            all_analyses = session.exec(total_query).all()
            confidence_scores = [a.confidence_score for a in all_analyses if a.confidence_score is not None]
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            return {
                'total_analyses': total_analyses,
                'completed_analyses': completed_analyses,
                'archived_analyses': archived_analyses,
                'recent_analyses': recent_analyses,
                'average_confidence_score': round(avg_confidence, 2),
                'success_rate': round((completed_analyses / total_analyses) * 100, 1) if total_analyses > 0 else 0
            }

# Global service instance
transcript_analysis_service = TranscriptAnalysisService()
