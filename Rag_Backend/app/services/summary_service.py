"""
AI Summary Service - Intelligent Summary Generation and Management
Implements the hybrid approach for automatic summary generation with user control
"""

import time
import asyncio
import os
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter, defaultdict
from sqlmodel import Session, select
from sqlalchemy import and_, or_, func
import logging
from pathlib import Path

from app.database import get_session
from app.models import (
    Document, AISummary, SummarySettings, 
    DocumentSummaryMapping, SummaryGenerationHistory
)
from app.services.llm_factory import get_llm
from app.deps import get_current_user

logger = logging.getLogger(__name__)

def read_document_content(document: Document) -> str:
    """
    Read the actual content from a document file
    Returns the text content or a fallback message if content cannot be read
    """
    try:
        if not document.path or not os.path.exists(document.path):
            return f"DOCUMENT CONTENT: No content available - file not found at {document.path or 'unknown path'}"
        
        file_ext = Path(document.path).suffix.lower()
        
        # Handle different file types
        if file_ext == '.txt':
            with open(document.path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        elif file_ext == '.pdf':
            try:
                import pdfplumber
                with pdfplumber.open(document.path) as pdf:
                    content = ""
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            content += page_text + "\n"
            except ImportError:
                return f"DOCUMENT CONTENT: PDF reading not available - pdfplumber not installed"
            except Exception as e:
                return f"DOCUMENT CONTENT: Error reading PDF - {str(e)}"
        elif file_ext == '.docx':
            try:
                import docx
                doc = docx.Document(document.path)
                content = ""
                for paragraph in doc.paragraphs:
                    content += paragraph.text + "\n"
            except ImportError:
                return f"DOCUMENT CONTENT: DOCX reading not available - python-docx not installed"
            except Exception as e:
                return f"DOCUMENT CONTENT: Error reading DOCX - {str(e)}"
        else:
            # Try to read as text file
            try:
                with open(document.path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            except Exception as e:
                return f"DOCUMENT CONTENT: Error reading file - {str(e)}"
        
        # Clean and validate content
        if not content or not content.strip():
            return f"DOCUMENT CONTENT: No content available - file appears to be empty"
        
        # Truncate very long content for summary generation
        if len(content) > 50000:  # 50KB limit
            content = content[:50000] + "\n\n[Content truncated for processing...]"
        
        return content.strip()
        
    except Exception as e:
        logger.error(f"Failed to read document content for {document.filename}: {e}")
        return f"DOCUMENT CONTENT: Error reading file - {str(e)}"

class SummaryEngine:
    """Core AI engine for generating and managing document summaries"""
    
    def __init__(self):
        self.llm = None
    
    async def generate_summaries_from_documents(
        self, 
        documents: List[Document], 
        user_id: str,
        max_summaries: int = 8,
        min_docs_per_summary: int = 1,
        include_individual: bool = True,
        include_combined: bool = True,
        default_summary_type: str = "auto"
    ) -> List[Dict]:
        """
        Generate AI summaries from a collection of documents
        Uses different strategies based on document count and user preferences
        """
        if not documents:
            return []
        
        logger.info(f"Generating summaries for {len(documents)} documents (user: {user_id})")
        
        summaries = []
        
        # Strategy 1: Individual document summaries
        if include_individual and len(documents) > 0:
            individual_summaries = await self._generate_individual_summaries(
                documents, user_id, max_summaries
            )
            summaries.extend(individual_summaries)
        
        # Strategy 2: Combined summary for multiple documents
        if include_combined and len(documents) > 1:
            combined_summary = await self._generate_combined_summary(
                documents, user_id
            )
            if combined_summary:
                summaries.append(combined_summary)
        
        # Strategy 3: Thematic summaries based on document clustering
        if len(documents) >= 3 and default_summary_type == "auto":
            thematic_summaries = await self._generate_thematic_summaries(
                documents, user_id, max_summaries // 2
            )
            summaries.extend(thematic_summaries)
        
        # Sort by confidence and limit to max_summaries
        summaries.sort(key=lambda x: x['confidence_score'], reverse=True)
        return summaries[:max_summaries]
    
    async def _generate_individual_summaries(
        self, 
        documents: List[Document], 
        user_id: str,
        max_individual: int = 4
    ) -> List[Dict]:
        """Generate individual summaries for each document"""
        summaries = []
        
        # Limit to most recent documents if too many
        docs_to_process = documents[:max_individual] if len(documents) > max_individual else documents
        
        for doc in docs_to_process:
            try:
                summary = await self._generate_individual_summary(doc, user_id)
                if summary:
                    summaries.append(summary)
            except Exception as e:
                logger.error(f"Failed to generate individual summary for {doc.filename}: {e}")
                continue
        
        return summaries
    
    async def _generate_individual_summary(
        self, 
        document: Document, 
        user_id: str
    ) -> Optional[Dict]:
        """Generate a summary for a single document"""
        
        try:
            # Read the actual document content
            document_content = read_document_content(document)
            
            # Create a focused prompt for individual document summary with actual content
            prompt = f"""
            Generate a comprehensive summary for the following document:
            
            Document: {document.filename}
            Existing Summary: {document.summary or "No existing summary"}
            Tags: {', '.join(document.tags or [])}
            Demo Tags: {', '.join(document.demo_tags or [])}
            
            DOCUMENT CONTENT:
            {document_content}
            
            Please provide:
            1. A concise executive summary (2-3 sentences)
            2. Key points and findings
            3. Main themes and topics covered
            4. Important insights and takeaways
            5. Actionable recommendations if applicable
            
            Format the response in clear, professional language suitable for business use.
            Focus on the most valuable and actionable information from the document content.
            """
            
            # Generate summary using AI with default provider (can be enhanced to accept provider parameter)
            if not self.llm:
                self.llm = get_llm(provider="ollama")  # Default to ollama for auto-generated summaries
            
            response = await self.llm.chat(prompt=prompt)
            
            # Calculate confidence based on document content availability
            confidence = self._calculate_individual_confidence(document)
            
            # Create short title from filename
            title = self._create_individual_title(document.filename)
            
            return {
                'title': title,
                'description': f"Individual summary for {document.filename}",
                'summary_type': 'individual',
                'focus_keywords': document.tags or [],
                'document_ids': [document.id],
                'confidence_score': confidence,
                'document_count': 1,
                'summary_content': response.strip(),
                'content_length': len(response.strip())
            }
            
        except Exception as e:
            logger.error(f"Failed to generate individual summary for {document.filename}: {e}")
            return None
    
    async def _generate_combined_summary(
        self, 
        documents: List[Document], 
        user_id: str
    ) -> Optional[Dict]:
        """Generate a combined summary from multiple documents"""
        
        try:
            # Collect document information with actual content
            doc_info = []
            all_tags = set()
            
            for doc in documents:
                # Read actual document content
                document_content = read_document_content(doc)
                
                # Truncate content for combined summary to avoid overwhelming the prompt
                if len(document_content) > 2000:
                    content_preview = document_content[:2000] + "\n[Content truncated...]"
                else:
                    content_preview = document_content
                
                doc_info.append(f"- {doc.filename}:\n  Content: {content_preview}\n  Summary: {doc.summary or 'No summary available'}")
                
                if doc.tags:
                    all_tags.update(doc.tags)
                if doc.demo_tags:
                    all_tags.update(doc.demo_tags)
            
            # Create comprehensive prompt for combined summary with actual content
            prompt = f"""
            Generate a comprehensive combined summary that synthesizes insights from {len(documents)} documents:
            
            Documents with Content:
            {chr(10).join(doc_info)}
            
            Common themes: {', '.join(list(all_tags)[:10])}
            
            Please provide:
            1. Executive Overview: High-level synthesis of all documents
            2. Cross-Document Insights: Key patterns and themes across documents
            3. Collective Findings: Most important discoveries from the entire set
            4. Strategic Implications: What these documents mean together
            5. Recommended Actions: Next steps based on combined analysis
            
            Focus on creating a cohesive narrative that shows how all documents work together
            to provide comprehensive insights. Highlight synergies and complementary information.
            Base your analysis on the actual document content provided above.
            """
            
            # Generate summary using AI
            if not self.llm:
                self.llm = get_llm(provider="ollama")
            
            response = await self.llm.chat(prompt=prompt)
            
            # Calculate confidence based on document diversity and count
            confidence = self._calculate_combined_confidence(documents)
            
            return {
                'title': f"Combined Analysis ({len(documents)} Documents)",
                'description': f"Comprehensive summary combining insights from {len(documents)} documents",
                'summary_type': 'combined',
                'focus_keywords': list(all_tags)[:10],
                'document_ids': [doc.id for doc in documents],
                'confidence_score': confidence,
                'document_count': len(documents),
                'summary_content': response.strip(),
                'content_length': len(response.strip())
            }
            
        except Exception as e:
            logger.error(f"Failed to generate combined summary: {e}")
            return None
    
    async def _generate_thematic_summaries(
        self, 
        documents: List[Document], 
        user_id: str,
        max_thematic: int = 3
    ) -> List[Dict]:
        """Generate thematic summaries based on document clustering"""
        
        summaries = []
        
        try:
            # Extract themes and cluster documents
            theme_clusters = self._cluster_documents_by_themes(documents)
            
            # Generate summaries for top themes
            sorted_themes = sorted(
                theme_clusters.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )[:max_thematic]
            
            for theme, theme_docs in sorted_themes:
                if len(theme_docs) >= 2:  # Need at least 2 docs for thematic summary
                    thematic_summary = await self._generate_thematic_summary(
                        theme, theme_docs, user_id
                    )
                    if thematic_summary:
                        summaries.append(thematic_summary)
        
        except Exception as e:
            logger.error(f"Failed to generate thematic summaries: {e}")
        
        return summaries
    
    async def _generate_thematic_summary(
        self, 
        theme: str, 
        documents: List[Document], 
        user_id: str
    ) -> Optional[Dict]:
        """Generate a summary focused on a specific theme"""
        
        try:
            # Collect theme-relevant information
            doc_info = []
            for doc in documents:
                relevant_tags = [tag for tag in (doc.tags or []) if theme.lower() in tag.lower()]
                doc_info.append(f"- {doc.filename}: {doc.summary or 'No summary'} (Tags: {', '.join(relevant_tags)})")
            
            prompt = f"""
            Generate a thematic summary focused on "{theme}" from the following documents:
            
            {chr(10).join(doc_info)}
            
            Please provide:
            1. Theme Overview: What this theme represents across the documents
            2. Key Insights: Main findings related to this theme
            3. Supporting Evidence: Specific examples from the documents
            4. Implications: What this theme means for decision-making
            5. Recommendations: Actions related to this theme
            
            Focus specifically on the "{theme}" aspect and how it appears across the document set.
            """
            
            # Generate summary using AI
            if not self.llm:
                self.llm = get_llm(provider="ollama")
            
            response = await self.llm.chat(prompt=prompt)
            
            # Calculate confidence based on theme relevance
            confidence = self._calculate_thematic_confidence(theme, documents)
            
            return {
                'title': f"{theme.title()} Analysis",
                'description': f"Thematic analysis of {theme} across {len(documents)} documents",
                'summary_type': 'thematic',
                'focus_keywords': [theme],
                'document_ids': [doc.id for doc in documents],
                'confidence_score': confidence,
                'document_count': len(documents),
                'summary_content': response.strip(),
                'content_length': len(response.strip())
            }
            
        except Exception as e:
            logger.error(f"Failed to generate thematic summary for {theme}: {e}")
            return None
    
    def _cluster_documents_by_themes(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """Cluster documents by common themes"""
        
        theme_to_docs = defaultdict(list)
        
        for doc in documents:
            doc_themes = set()
            if doc.tags:
                doc_themes.update(doc.tags)
            if doc.demo_tags:
                doc_themes.update(doc.demo_tags)
            
            # Add document to each of its themes
            for theme in doc_themes:
                theme_to_docs[theme].append(doc)
        
        # Filter themes that appear in multiple documents
        filtered_themes = {
            theme: docs for theme, docs in theme_to_docs.items() 
            if len(docs) >= 2
        }
        
        return filtered_themes
    
    def _calculate_individual_confidence(self, document: Document) -> float:
        """Calculate confidence score for individual summary"""
        
        score = 0.5  # Base score
        
        # Boost for existing summary
        if document.summary:
            score += 0.2
        
        # Boost for tags
        if document.tags:
            score += min(len(document.tags) * 0.05, 0.2)
        
        # Boost for demo tags
        if document.demo_tags:
            score += min(len(document.demo_tags) * 0.05, 0.1)
        
        return min(score, 1.0)
    
    def _calculate_combined_confidence(self, documents: List[Document]) -> float:
        """Calculate confidence score for combined summary"""
        
        # Base score increases with document count
        base_score = min(0.3 + (len(documents) * 0.1), 0.8)
        
        # Boost for documents with summaries
        docs_with_summaries = sum(1 for doc in documents if doc.summary)
        summary_boost = (docs_with_summaries / len(documents)) * 0.2
        
        return min(base_score + summary_boost, 1.0)
    
    def _calculate_thematic_confidence(self, theme: str, documents: List[Document]) -> float:
        """Calculate confidence score for thematic summary"""
        
        # Base score
        score = 0.6
        
        # Boost for more documents with this theme
        score += min(len(documents) * 0.05, 0.3)
        
        # Boost for theme frequency across documents
        theme_mentions = 0
        for doc in documents:
            all_tags = (doc.tags or []) + (doc.demo_tags or [])
            theme_mentions += sum(1 for tag in all_tags if theme.lower() in tag.lower())
        
        frequency_boost = min(theme_mentions * 0.02, 0.1)
        score += frequency_boost
        
        return min(score, 1.0)
    
    def _create_individual_title(self, filename: str) -> str:
        """Create a clean title from filename"""
        
        # Remove extension and clean up
        name = filename.split('.')[0]
        
        # Limit length to ensure final title is <= 30 characters
        # Account for " Summary" (8 characters)
        max_name_length = 22
        if len(name) > max_name_length:
            name = name[:max_name_length-3] + "..."
        
        return f"{name} Summary"


class SummaryService:
    """Main service for managing AI summaries"""
    
    def __init__(self):
        self.engine = SummaryEngine()
    
    async def get_user_summaries(self, user_id: str, session: Session) -> List[Dict]:
        """Get all active summaries for a user's tenant"""
        
        # Get user's tenant_id for proper data sharing
        from app.database import User
        user = session.exec(select(User).where(User.id == user_id)).first()
        if not user or not user.tenant_id:
            return []
        
        statement = select(AISummary).where(
            and_(
                AISummary.tenant_id == user.tenant_id,
                AISummary.status.in_(['active', 'stale'])
            )
        ).order_by(AISummary.last_updated.desc())
        
        summaries = session.exec(statement).all()
        
        result = []
        for summary in summaries:
            result.append({
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
            })
        
        return result
    
    async def get_user_settings(self, user_id: str, session: Session) -> Dict:
        """Get summary settings for a user"""
        
        statement = select(SummarySettings).where(SummarySettings.owner_id == user_id)
        settings = session.exec(statement).first()
        
        if not settings:
            # Create default settings
            settings = SummarySettings(
                owner_id=user_id,
                auto_refresh=True,
                on_add="incremental",
                on_delete="auto_clean",
                change_threshold=15,
                max_summaries=8,
                min_documents_per_summary=1,
                default_summary_type="auto",
                include_individual=True,
                include_combined=True
            )
            session.add(settings)
            session.commit()
            session.refresh(settings)
        
        return {
            'autoRefresh': settings.auto_refresh,
            'onAdd': settings.on_add,
            'onDelete': settings.on_delete,
            'changeThreshold': settings.change_threshold,
            'maxSummaries': settings.max_summaries,
            'minDocumentsPerSummary': settings.min_documents_per_summary,
            'defaultSummaryType': settings.default_summary_type,
            'includeIndividual': settings.include_individual,
            'includeCombined': settings.include_combined
        }
    
    async def update_user_settings(
        self, 
        user_id: str, 
        settings_data: Dict, 
        session: Session
    ) -> Dict:
        """Update summary settings for a user"""
        
        statement = select(SummarySettings).where(SummarySettings.owner_id == user_id)
        settings = session.exec(statement).first()
        
        if not settings:
            settings = SummarySettings(owner_id=user_id)
            session.add(settings)
        
        # Update settings
        if 'autoRefresh' in settings_data:
            settings.auto_refresh = settings_data['autoRefresh']
        if 'onAdd' in settings_data:
            settings.on_add = settings_data['onAdd']
        if 'onDelete' in settings_data:
            settings.on_delete = settings_data['onDelete']
        if 'changeThreshold' in settings_data:
            settings.change_threshold = settings_data['changeThreshold']
        if 'maxSummaries' in settings_data:
            settings.max_summaries = settings_data['maxSummaries']
        if 'minDocumentsPerSummary' in settings_data:
            settings.min_documents_per_summary = settings_data['minDocumentsPerSummary']
        if 'defaultSummaryType' in settings_data:
            settings.default_summary_type = settings_data['defaultSummaryType']
        if 'includeIndividual' in settings_data:
            settings.include_individual = settings_data['includeIndividual']
        if 'includeCombined' in settings_data:
            settings.include_combined = settings_data['includeCombined']
        
        settings.updated_at = time.time()
        session.commit()
        session.refresh(settings)
        
        return await self.get_user_settings(user_id, session)
    
    async def generate_summaries_for_user(
        self, 
        user_id: str, 
        session: Session,
        generation_type: str = "manual",
        trigger_event: str = "manual_refresh"
    ) -> Dict:
        """Generate or regenerate summaries for a user"""
        
        start_time = time.time()
        
        try:
            # Get user settings
            settings = await self.get_user_settings(user_id, session)
            
            # Get user's tenant_id and documents from tenant
            from app.database import User
            user = session.exec(select(User).where(User.id == user_id)).first()
            if not user or not user.tenant_id:
                return {
                    'success': True,
                    'message': 'No tenant found for user',
                    'summaries': [],
                    'stats': {
                        'documentsProcessed': 0,
                        'summariesCreated': 0,
                        'summariesUpdated': 0,
                        'summariesDeleted': 0,
                        'processingTimeMs': 0
                    }
                }
            
            # Get documents from user's tenant for proper data sharing
            doc_statement = select(Document).where(Document.tenant_id == user.tenant_id)
            documents = session.exec(doc_statement).all()
            
            if not documents:
                return {
                    'success': True,
                    'message': 'No documents found',
                    'summaries': [],
                    'stats': {
                        'documentsProcessed': 0,
                        'summariesCreated': 0,
                        'summariesUpdated': 0,
                        'summariesDeleted': 0,
                        'processingTimeMs': 0
                    }
                }
            
            # Archive existing summaries
            existing_statement = select(AISummary).where(
                and_(
                    AISummary.owner_id == user_id,
                    AISummary.status == 'active'
                )
            )
            existing_summaries = session.exec(existing_statement).all()
            
            for summary in existing_summaries:
                summary.status = 'archived'
            
            # Generate new summaries
            new_summaries_data = await self.engine.generate_summaries_from_documents(
                documents,
                user_id,
                max_summaries=settings['maxSummaries'],
                min_docs_per_summary=settings['minDocumentsPerSummary'],
                include_individual=settings['includeIndividual'],
                include_combined=settings['includeCombined'],
                default_summary_type=settings['defaultSummaryType']
            )
            
            # Save new summaries to database
            created_summaries = []
            for summary_data in new_summaries_data:
                summary = AISummary(
                    owner_id=user_id,
                    tenant_id=user.tenant_id,  # Add tenant_id for proper isolation
                    title=summary_data['title'],
                    description=summary_data['description'],
                    summary_type=summary_data['summary_type'],
                    focus_keywords=summary_data['focus_keywords'],
                    document_ids=summary_data['document_ids'],
                    confidence_score=summary_data['confidence_score'],
                    document_count=summary_data['document_count'],
                    content_length=summary_data['content_length'],
                    summary_content=summary_data['summary_content'],
                    auto_generated=True,
                    generation_method=generation_type,
                    status='active'
                )
                session.add(summary)
                session.flush()  # Get the ID
                
                # Create document mappings
                for doc_id in summary_data['document_ids']:
                    mapping = DocumentSummaryMapping(
                        document_id=doc_id,
                        summary_id=summary.id,
                        owner_id=user_id,
                        relevance_score=0.8,  # Default relevance
                        contribution_weight=1.0 / len(summary_data['document_ids'])
                    )
                    session.add(mapping)
                
                created_summaries.append(summary)
            
            session.commit()
            
            # Record generation history
            processing_time = int((time.time() - start_time) * 1000)
            history = SummaryGenerationHistory(
                owner_id=user_id,
                generation_type=generation_type,
                trigger_event=trigger_event,
                documents_processed=len(documents),
                summaries_created=len(created_summaries),
                summaries_updated=0,
                summaries_deleted=len(existing_summaries),
                processing_time_ms=processing_time,
                success=True,
                meta_data={
                    'total_documents': len(documents),
                    'settings': settings
                }
            )
            session.add(history)
            session.commit()
            
            # Return results
            result_summaries = []
            for summary in created_summaries:
                result_summaries.append({
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
                    'contentLength': summary.content_length,
                    'content': summary.summary_content
                })
            
            return {
                'success': True,
                'message': f'Generated {len(created_summaries)} summaries',
                'summaries': result_summaries,
                'stats': {
                    'documentsProcessed': len(documents),
                    'summariesCreated': len(created_summaries),
                    'summariesUpdated': 0,
                    'summariesDeleted': len(existing_summaries),
                    'processingTimeMs': processing_time
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate summaries for user {user_id}: {e}")
            
            # Record failed generation
            processing_time = int((time.time() - start_time) * 1000)
            history = SummaryGenerationHistory(
                owner_id=user_id,
                generation_type=generation_type,
                trigger_event=trigger_event,
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e)
            )
            session.add(history)
            session.commit()
            
            return {
                'success': False,
                'message': f'Failed to generate summaries: {str(e)}',
                'summaries': [],
                'stats': {
                    'documentsProcessed': 0,
                    'summariesCreated': 0,
                    'summariesUpdated': 0,
                    'summariesDeleted': 0,
                    'processingTimeMs': processing_time
                }
            }
    
    async def handle_document_addition(
        self, 
        user_id: str, 
        new_document_ids: List[str], 
        session: Session
    ) -> Dict:
        """Handle summary updates when documents are added"""
        
        settings = await self.get_user_settings(user_id, session)
        
        if not settings['autoRefresh']:
            # Mark existing summaries as stale
            statement = select(AISummary).where(
                and_(
                    AISummary.owner_id == user_id,
                    AISummary.status == 'active'
                )
            )
            summaries = session.exec(statement).all()
            
            for summary in summaries:
                summary.status = 'stale'
            
            session.commit()
            
            return {
                'success': True,
                'message': 'Summaries marked as stale - manual refresh needed',
                'action': 'marked_stale'
            }
        
        # Auto-refresh based on settings
        if settings['onAdd'] == 'full':
            return await self.generate_summaries_for_user(
                user_id, session, 
                generation_type="full", 
                trigger_event="document_added"
            )
        elif settings['onAdd'] == 'incremental':
            # TODO: Implement incremental update logic
            # For now, fall back to full regeneration
            return await self.generate_summaries_for_user(
                user_id, session, 
                generation_type="incremental", 
                trigger_event="document_added"
            )
        else:  # manual
            return {
                'success': True,
                'message': 'Auto-refresh disabled - manual refresh needed',
                'action': 'no_action'
            }
    
    async def handle_document_deletion(
        self, 
        user_id: str, 
        deleted_document_ids: List[str], 
        session: Session
    ) -> Dict:
        """Handle summary updates when documents are deleted"""
        
        settings = await self.get_user_settings(user_id, session)
        
        # Remove document mappings
        mapping_statement = select(DocumentSummaryMapping).where(
            and_(
                DocumentSummaryMapping.owner_id == user_id,
                DocumentSummaryMapping.document_id.in_(deleted_document_ids)
            )
        )
        mappings = session.exec(mapping_statement).all()
        
        affected_summary_ids = set()
        for mapping in mappings:
            affected_summary_ids.add(mapping.summary_id)
            session.delete(mapping)
        
        # Update affected summaries
        summaries_updated = 0
        summaries_deleted = 0
        
        for summary_id in affected_summary_ids:
            summary_statement = select(AISummary).where(AISummary.id == summary_id)
            summary = session.exec(summary_statement).first()
            
            if summary:
                # Remove deleted documents from summary
                summary.document_ids = [
                    doc_id for doc_id in summary.document_ids 
                    if doc_id not in deleted_document_ids
                ]
                summary.document_count = len(summary.document_ids)
                
                # Check if summary still has enough documents
                if summary.document_count < settings['minDocumentsPerSummary']:
                    summary.status = 'archived'
                    summaries_deleted += 1
                else:
                    summary.last_updated = time.time()
                    if settings['onDelete'] == 'auto_clean':
                        summary.status = 'active'
                    else:
                        summary.status = 'stale'
                    summaries_updated += 1
        
        session.commit()
        
        return {
            'success': True,
            'message': f'Updated {summaries_updated} summaries, archived {summaries_deleted}',
            'stats': {
                'summariesUpdated': summaries_updated,
                'summariesDeleted': summaries_deleted,
                'documentsRemoved': len(deleted_document_ids)
            }
        }
    
    async def get_summary_status(self, user_id: str, session: Session) -> Dict:
        """Get the current status of summaries for a user"""
        
        statement = select(AISummary).where(AISummary.owner_id == user_id)
        summaries = session.exec(statement).all()
        
        status_counts = Counter([s.status for s in summaries])
        
        # Check if any summaries need updates
        needs_update = status_counts.get('stale', 0) > 0
        
        return {
            'totalSummaries': len(summaries),
            'activeSummaries': status_counts.get('active', 0),
            'staleSummaries': status_counts.get('stale', 0),
            'archivedSummaries': status_counts.get('archived', 0),
            'needsUpdate': needs_update,
            'lastGenerated': max([s.last_updated for s in summaries]).isoformat() if summaries else None
        }
    
    async def create_custom_summary(
        self,
        user_id: str,
        session: Session,
        title: str,
        description: Optional[str] = None,
        keywords: List[str] = None,
        focus_area: Optional[str] = None,
        provider: str = "gemini",
        model: str = "gemini-1.5-flash"
    ) -> Dict:
        """Create a custom AI summary with user-defined focus and keywords"""
        
        if not title.strip():
            return {
                'success': False,
                'message': 'Title is required for custom summary'
            }
        
        if keywords is None:
            keywords = []
        
        try:
            # Get user documents to find relevant ones for this summary
            doc_statement = select(Document).where(Document.owner_id == user_id)
            documents = session.exec(doc_statement).all()
            
            logger.info(f"Found {len(documents)} documents for user {user_id}")
            
            if not documents:
                # Check if there are any documents at all in the database for debugging
                all_docs_statement = select(Document)
                all_documents = session.exec(all_docs_statement).all()
                logger.warning(f"No documents found for user {user_id}. Total documents in database: {len(all_documents)}")
                
                return {
                    'success': False,
                    'message': f'No documents found for user {user_id}. Please upload documents first. Total documents in system: {len(all_documents)}'
                }
            
            # Find documents relevant to the custom summary
            relevant_docs = self._find_relevant_documents_for_summary(
                documents, keywords, title, description, focus_area
            )
            
            if not relevant_docs:
                # If no specific matches, use all documents
                relevant_docs = documents
            
            # Generate custom summary content
            summary_content = await self._generate_custom_summary_content(
                title, description, keywords, focus_area, relevant_docs, provider, model
            )
            
            # Create the custom summary
            summary = AISummary(
                owner_id=user_id,
                title=title,
                description=description or f"Custom summary about {title}",
                summary_type="custom",
                focus_keywords=keywords,
                document_ids=[doc.id for doc in relevant_docs],
                confidence_score=0.9,  # High confidence for user-created summaries
                document_count=len(relevant_docs),
                content_length=len(summary_content),
                summary_content=summary_content,
                auto_generated=False,  # This is user-created
                generation_method="custom",
                status='active'
            )
            
            session.add(summary)
            session.flush()  # Get the ID
            
            # Create document mappings
            for doc in relevant_docs:
                relevance_score = self._calculate_document_relevance_for_summary(
                    doc, keywords, title, focus_area
                )
                
                mapping = DocumentSummaryMapping(
                    document_id=doc.id,
                    summary_id=summary.id,
                    owner_id=user_id,
                    relevance_score=relevance_score,
                    contribution_weight=1.0 / len(relevant_docs)
                )
                session.add(mapping)
            
            session.commit()
            session.refresh(summary)
            
            # Record generation history
            history = SummaryGenerationHistory(
                owner_id=user_id,
                generation_type="custom",
                trigger_event="user_created",
                documents_processed=len(relevant_docs),
                summaries_created=1,
                summaries_updated=0,
                summaries_deleted=0,
                processing_time_ms=0,
                success=True,
                meta_data={
                    'custom_title': title,
                    'custom_keywords': keywords,
                    'focus_area': focus_area,
                    'provider': provider,
                    'model': model
                }
            )
            session.add(history)
            session.commit()
            
            # Return the created summary
            return {
                'success': True,
                'message': f'Custom summary "{title}" created successfully',
                'summary': {
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
            
        except Exception as e:
            logger.error(f"Failed to create custom summary for user {user_id}: {e}")
            return {
                'success': False,
                'message': f'Failed to create custom summary: {str(e)}'
            }
    
    async def _generate_custom_summary_content(
        self,
        title: str,
        description: Optional[str],
        keywords: List[str],
        focus_area: Optional[str],
        documents: List[Document],
        provider: str,
        model: str
    ) -> str:
        """Generate custom summary content using AI"""
        
        try:
            # Build comprehensive prompt for custom summary
            doc_info = []
            for doc in documents:
                doc_info.append(f"- {doc.filename}: {doc.summary or 'No summary available'}")
                if doc.tags:
                    doc_info.append(f"  Tags: {', '.join(doc.tags)}")
            
            keywords_str = ', '.join(keywords) if keywords else "general analysis"
            focus_str = f" with specific focus on {focus_area}" if focus_area else ""
            
            prompt = f"""
            Generate a custom summary titled "{title}" based on the following requirements:
            
            Description: {description or "Custom analysis of uploaded documents"}
            Keywords/Topics: {keywords_str}
            Focus Area: {focus_area or "General comprehensive analysis"}{focus_str}
            
            Documents to analyze:
            {chr(10).join(doc_info)}
            
            Please provide a comprehensive summary that includes:
            1. Executive Summary: Overview aligned with the specified focus
            2. Key Findings: Main insights related to the keywords and focus area
            3. Detailed Analysis: In-depth examination of relevant aspects
            4. Supporting Evidence: Specific examples from the documents
            5. Strategic Implications: What these findings mean
            6. Recommendations: Actionable next steps
            
            Focus specifically on the requested keywords ({keywords_str}) and ensure the analysis
            addresses the user's specific interests and requirements.
            
            Format the response using markdown with clear headings and professional structure.
            """
            
            # Use the LLM factory to generate content with the specified provider and model
            llm = get_llm(provider=provider, model=model)
            
            # Handle both sync and async LLM methods
            try:
                if hasattr(llm, 'chat'):
                    if asyncio.iscoroutinefunction(llm.chat):
                        response = await llm.chat(prompt=prompt)
                    else:
                        response = llm.chat(prompt=prompt)
                elif hasattr(llm, 'generate'):
                    if asyncio.iscoroutinefunction(llm.generate):
                        response = await llm.generate(prompt)
                    else:
                        response = llm.generate(prompt)
                else:
                    # Fallback to string representation
                    response = str(llm)
                
                return response.strip() if response else self._generate_fallback_summary_content(title, keywords, documents)
            except Exception as llm_error:
                logger.warning(f"LLM generation failed with {provider}/{model}: {llm_error}, using fallback")
                return self._generate_fallback_summary_content(title, keywords, documents)
            
        except Exception as e:
            logger.error(f"Failed to generate custom summary content: {e}")
            # Return fallback content
            return self._generate_fallback_summary_content(title, keywords, documents)
    
    def _find_relevant_documents_for_summary(
        self,
        documents: List[Document],
        keywords: List[str],
        title: str,
        description: Optional[str] = None,
        focus_area: Optional[str] = None
    ) -> List[Document]:
        """Find documents relevant to a custom summary"""
        
        if not keywords and not title and not focus_area:
            return documents
        
        # Create search terms from title, description, keywords, and focus area
        search_terms = set()
        
        # Add keywords
        search_terms.update([kw.lower().strip() for kw in keywords])
        
        # Extract terms from title
        title_words = [word.lower().strip() for word in title.split() if len(word) > 2]
        search_terms.update(title_words)
        
        # Extract terms from description
        if description:
            desc_words = [word.lower().strip() for word in description.split() if len(word) > 2]
            search_terms.update(desc_words)
        
        # Extract terms from focus area
        if focus_area:
            focus_words = [word.lower().strip() for word in focus_area.split() if len(word) > 2]
            search_terms.update(focus_words)
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'she', 'use', 'way', 'will', 'with', 'this', 'that', 'they',
            'have', 'from', 'know', 'want', 'been', 'good', 'much', 'some',
            'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make',
            'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'about'
        }
        
        search_terms = {term for term in search_terms if term not in stop_words and len(term) > 2}
        
        if not search_terms:
            return documents
        
        # Score documents based on relevance
        doc_scores = []
        
        for doc in documents:
            score = 0
            doc_text_fields = []
            
            # Collect all text from document
            if doc.tags:
                doc_text_fields.extend([tag.lower() for tag in doc.tags])
            if doc.demo_tags:
                doc_text_fields.extend([tag.lower() for tag in doc.demo_tags])
            if doc.summary:
                doc_text_fields.extend(doc.summary.lower().split())
            if doc.filename:
                doc_text_fields.extend(doc.filename.lower().split())
            
            # Calculate relevance score
            for term in search_terms:
                for field_text in doc_text_fields:
                    if term in field_text:
                        score += 1
            
            if score > 0:
                doc_scores.append((doc, score))
        
        # Sort by relevance score and return top documents
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return documents with any relevance, or all documents if none match
        relevant_docs = [doc for doc, score in doc_scores if score > 0]
        return relevant_docs if relevant_docs else documents
    
    def _calculate_document_relevance_for_summary(
        self,
        document: Document,
        keywords: List[str],
        title: str,
        focus_area: Optional[str] = None
    ) -> float:
        """Calculate how relevant a document is to a custom summary"""
        
        if not keywords and not title and not focus_area:
            return 0.5  # Default relevance
        
        # Create search terms
        search_terms = set()
        search_terms.update([kw.lower().strip() for kw in keywords])
        search_terms.update([word.lower().strip() for word in title.split() if len(word) > 2])
        if focus_area:
            search_terms.update([word.lower().strip() for word in focus_area.split() if len(word) > 2])
        
        # Get document text
        doc_text = []
        if document.tags:
            doc_text.extend([tag.lower() for tag in document.tags])
        if document.demo_tags:
            doc_text.extend([tag.lower() for tag in document.demo_tags])
        if document.summary:
            doc_text.extend(document.summary.lower().split())
        if document.filename:
            doc_text.extend(document.filename.lower().split())
        
        # Calculate matches
        matches = 0
        for term in search_terms:
            for text_item in doc_text:
                if term in text_item:
                    matches += 1
        
        # Calculate relevance score (0.0 to 1.0)
        if not search_terms:
            return 0.5
        
        relevance = min(matches / len(search_terms), 1.0)
        return max(relevance, 0.1)  # Minimum relevance of 0.1
    
    def _generate_fallback_summary_content(
        self,
        title: str,
        keywords: List[str],
        documents: List[Document]
    ) -> str:
        """Generate fallback content when AI generation fails"""
        
        keywords_str = ', '.join(keywords) if keywords else 'document analysis'
        
        return f"""# {title}

## Overview
This summary provides insights into {keywords_str.lower()} based on your document collection.

## Key Insights

### 1. Document Analysis
Your {len(documents)} document(s) contain valuable information related to {keywords_str.lower()} that can inform strategic decision-making.

### 2. Trends and Patterns
- Analysis reveals important patterns in your data
- Key themes emerge from the document collection
- Strategic opportunities are identified

### 3. Recommendations
- Focus on data-driven decision making
- Leverage insights for strategic planning
- Monitor key performance indicators
- Implement best practices from the analysis

## Document Summary
{chr(10).join([f"- **{doc.filename}**: {doc.summary or 'Analysis pending'}" for doc in documents[:5]])}
{f"- *...and {len(documents) - 5} more documents*" if len(documents) > 5 else ""}

## Next Steps
1. Review the detailed findings in your documents
2. Develop action plans based on the insights
3. Monitor progress and adjust strategies as needed
4. Continue to gather and analyze relevant data

## Conclusion
This summary highlights the importance of leveraging your document insights for strategic advantage and operational excellence.

*Note: This is a fallback summary. For detailed AI-generated insights, please ensure your AI provider is properly configured.*"""


# Global service instance
summary_service = SummaryService()
