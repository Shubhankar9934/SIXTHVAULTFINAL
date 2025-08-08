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
        """Generate a detailed, fact-oriented summary for a single document (1500-2000 words)"""
        
        try:
            # Read the actual document content
            document_content = read_document_content(document)
            
            # Create a comprehensive prompt for detailed individual document summary
            prompt = f"""
            Generate a comprehensive, detailed summary for the following document. The summary should be 1500-2000 words and provide an in-depth analysis based strictly on the facts and content present in the document.
            
            Document: {document.filename}
            Existing Summary: {document.summary or "No existing summary"}
            Tags: {', '.join(document.tags or [])}
            Demo Tags: {', '.join(document.demo_tags or [])}
            
            DOCUMENT CONTENT:
            {document_content}
            
            **CRITICAL REQUIREMENTS - STRICTLY FOLLOW THESE RULES**:
            - NEVER add percentages, statistics, or numerical data that are not explicitly present in the source document
            - NEVER fabricate or estimate potential ROI, growth rates, market share, conversion rates, or any quantitative metrics
            - NEVER use phrases like "potential X% increase", "estimated Y% improvement", "could result in Z% growth", "X-Y% reduction", "increase by X%"
            - NEVER create dummy percentages or made-up statistics to make the content sound more authoritative
            - Base ALL insights strictly on the actual content and data present in the provided document
            - If no specific metrics are provided in the document, focus on qualitative insights only
            - When referencing data, always ensure it comes directly from the source document
            - Use phrases like "based on the document", "according to the content", "the document shows" when making claims
            - If you want to suggest potential impact, use qualitative terms like "significant", "substantial", "notable", "considerable" instead of percentages
            - TARGET LENGTH: 1500-2000 words for comprehensive coverage
            - Provide detailed analysis with extensive factual support from the document
            
            **REQUIRED DETAILED STRUCTURE**:
            
            ## Executive Summary
            Provide a comprehensive overview of the document's main purpose, scope, and key findings based strictly on the content. Include the document's context, primary objectives, and most significant conclusions drawn from the actual material.
            
            ## Document Overview and Context
            - Document type, purpose, and intended audience based on content analysis
            - Scope and methodology (if mentioned in the document)
            - Background information and context provided in the document
            - Date, authorship, or organizational context if specified in the content
            
            ## Detailed Content Analysis
            ### Primary Findings and Facts
            - Comprehensive list of all major findings explicitly stated in the document
            - Detailed examination of data, research results, or conclusions presented
            - Specific facts, figures, and evidence directly quoted or referenced from the content
            - Methodological details if provided in the document
            
            ### Key Themes and Topics
            - In-depth analysis of main themes covered in the document
            - Detailed exploration of subtopics and supporting arguments
            - Connections between different sections or concepts within the document
            - Theoretical frameworks or models presented in the content
            
            ### Supporting Evidence and Documentation
            - Detailed review of evidence, case studies, or examples provided
            - Analysis of supporting data, charts, tables, or appendices mentioned
            - References to external sources or citations included in the document
            - Validation methods or quality assurance measures described
            
            ## Strategic Insights and Implications
            ### Business and Operational Implications
            - Detailed analysis of how findings impact business operations based on document content
            - Organizational implications explicitly discussed in the document
            - Process improvements or changes suggested in the content
            - Resource requirements or constraints mentioned in the document
            
            ### Market and Industry Context
            - Industry-specific insights provided in the document
            - Competitive landscape analysis if included in the content
            - Market trends or patterns identified in the document
            - Regulatory or compliance considerations mentioned
            
            ## Actionable Recommendations and Next Steps
            - Comprehensive list of all recommendations explicitly stated in the document
            - Detailed action items or implementation steps provided in the content
            - Priority levels or timelines mentioned in the document
            - Success metrics or evaluation criteria specified in the content
            - Risk mitigation strategies or considerations outlined in the document
            
            ## Conclusion and Key Takeaways
            - Summary of the most critical insights derived from the document
            - Long-term implications based on the content analysis
            - Areas requiring further investigation as mentioned in the document
            - Final assessment of the document's value and applicability
            
            **FORMATTING REQUIREMENTS**:
            - Use clear headings and subheadings as shown above
            - Include bullet points and numbered lists for clarity
            - Maintain professional, analytical tone throughout
            - Ensure each section meets the specified word count ranges
            - Ground every statement in actual document content
            - Use direct quotes where appropriate to support analysis
            - Provide comprehensive coverage while avoiding redundancy
            
            Focus on creating a thorough, fact-based analysis that serves as a comprehensive reference document for the original content. The summary should be detailed enough that a reader can understand the full scope and implications of the original document without needing to read it directly.
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
            
            **CRITICAL REQUIREMENTS - STRICTLY FOLLOW THESE RULES**:
            - NEVER add percentages, statistics, or numerical data that are not explicitly present in the source documents
            - NEVER fabricate or estimate potential ROI, growth rates, market share, conversion rates, or any quantitative metrics
            - NEVER use phrases like "potential X% increase", "estimated Y% improvement", "could result in Z% growth", "X-Y% reduction", "increase by X%"
            - NEVER create dummy percentages or made-up statistics to make the content sound more authoritative
            - Base ALL insights strictly on the actual content and data present in the provided documents
            - If no specific metrics are provided in the documents, focus on qualitative insights only
            - When referencing data, always ensure it comes directly from the source documents
            - Use phrases like "based on the documents", "according to the content", "the analysis shows" when making claims
            - If you want to suggest potential impact, use qualitative terms like "significant", "substantial", "notable", "considerable" instead of percentages
            
            Please provide:
            1. Executive Overview: High-level synthesis of all documents based strictly on document content
            2. Cross-Document Insights: Key patterns and themes across documents found in the source material
            3. Collective Findings: Most important discoveries from the entire set derived from actual content
            4. Strategic Implications: What these documents mean together based on provided information
            5. Recommended Actions: Next steps based on combined analysis from document content only
            
            Focus on creating a cohesive narrative that shows how all documents work together
            to provide comprehensive insights. Highlight synergies and complementary information.
            Base your analysis strictly on the actual document content provided above.
            Ground all insights in the actual document content and avoid any fabricated data.
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
            
            **CRITICAL REQUIREMENTS - STRICTLY FOLLOW THESE RULES**:
            - NEVER add percentages, statistics, or numerical data that are not explicitly present in the source documents
            - NEVER fabricate or estimate potential ROI, growth rates, market share, conversion rates, or any quantitative metrics
            - NEVER use phrases like "potential X% increase", "estimated Y% improvement", "could result in Z% growth", "X-Y% reduction", "increase by X%"
            - NEVER create dummy percentages or made-up statistics to make the content sound more authoritative
            - Base ALL insights strictly on the actual content and data present in the provided documents
            - If no specific metrics are provided in the documents, focus on qualitative insights only
            - When referencing data, always ensure it comes directly from the source documents
            - Use phrases like "based on the documents", "according to the content", "the analysis shows" when making claims
            - If you want to suggest potential impact, use qualitative terms like "significant", "substantial", "notable", "considerable" instead of percentages
            
            Please provide:
            1. Theme Overview: What this theme represents across the documents based strictly on document content
            2. Key Insights: Main findings related to this theme derived from actual document content
            3. Supporting Evidence: Specific examples directly from the documents
            4. Implications: What this theme means for decision-making based on provided information
            5. Recommendations: Actions related to this theme if mentioned in the documents
            
            Focus specifically on the "{theme}" aspect and how it appears across the document set.
            Ground all insights in the actual document content and avoid any fabricated data.
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
        """Get all active summaries for a user"""
        
        logger.info(f"Getting summaries for user {user_id}")
        
        # Get summaries directly by owner_id (no tenant requirement)
        statement = select(AISummary).where(
            and_(
                AISummary.owner_id == user_id,
                AISummary.status.in_(['active', 'stale'])
            )
        ).order_by(AISummary.last_updated.desc())
        
        summaries = session.exec(statement).all()
        logger.info(f"Found {len(summaries)} summaries in database for user {user_id}")
        
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
        
        logger.info(f"Returning {len(result)} formatted summaries for user {user_id}")
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
        model: str = "gemini-1.5-flash",
        document_ids: Optional[List[str]] = None
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
            # Get user's tenant_id and documents from tenant
            from app.database import User
            user = session.exec(select(User).where(User.id == user_id)).first()
            if not user or not user.tenant_id:
                return {
                    'success': False,
                    'message': 'No tenant found for user'
                }
            
            # If specific document IDs are provided, use only those documents
            if document_ids:
                logger.info(f"Using specific document IDs: {document_ids}")
                doc_statement = select(Document).where(
                    and_(
                        Document.id.in_(document_ids),
                        Document.tenant_id == user.tenant_id
                    )
                )
                documents = session.exec(doc_statement).all()
                
                if not documents:
                    return {
                        'success': False,
                        'message': f'No documents found with the specified IDs: {document_ids}'
                    }
                
                relevant_docs = documents
                logger.info(f"Found {len(relevant_docs)} documents from specified IDs")
            else:
                # Get all documents from user's tenant for proper data sharing
                doc_statement = select(Document).where(Document.tenant_id == user.tenant_id)
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
            summary_content = await self.generate_custom_summary_content(
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
        """Generate custom summary content using AI (private method)"""
        return await self.generate_custom_summary_content(
            title, description, keywords, focus_area, documents, provider, model
        )
    
    async def generate_custom_summary_content(
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
            Generate a comprehensive, detailed custom summary titled "{title}" based on the following requirements. The summary should be 1500-2000 words and provide an in-depth analysis based strictly on the facts and content present in the documents.
            
            Description: {description or "Custom analysis of uploaded documents"}
            Keywords/Topics: {keywords_str}
            Focus Area: {focus_area or "General comprehensive analysis"}{focus_str}
            
            Documents to analyze:
            {chr(10).join(doc_info)}
            
            **CRITICAL REQUIREMENTS - STRICTLY FOLLOW THESE RULES**:
            - NEVER add percentages, statistics, or numerical data that are not explicitly present in the source documents
            - NEVER fabricate or estimate potential ROI, growth rates, market share, conversion rates, or any quantitative metrics
            - NEVER use phrases like "potential X% increase", "estimated Y% improvement", "could result in Z% growth", "X-Y% reduction", "increase by X%"
            - NEVER create dummy percentages or made-up statistics to make the content sound more authoritative
            - Base ALL insights strictly on the actual content and data present in the provided documents
            - If no specific metrics are provided in the documents, focus on qualitative insights only
            - When referencing data, always ensure it comes directly from the source documents
            - Use phrases like "based on the documents", "according to the content", "the analysis shows" when making claims
            - If you want to suggest potential impact, use qualitative terms like "significant", "substantial", "notable", "considerable" instead of percentages
            - TARGET LENGTH: 1500-2000 words for comprehensive coverage
            - Provide detailed analysis with extensive factual support from the documents
            
            **REQUIRED DETAILED STRUCTURE**:
            
            ## Executive Summary
            Provide a comprehensive overview of the custom analysis focused on {keywords_str}. Include the scope of analysis, primary objectives related to the focus area, and most significant conclusions drawn from the document collection. Establish the context and importance of this analysis based strictly on the document content.
            
            ## Analysis Scope and Methodology
            - Define the scope of analysis based on the specified keywords and focus area
            - Document selection criteria and relevance to the focus area
            - Analytical approach used to examine the documents
            - Limitations or constraints identified in the available documentation
            
            ## Detailed Findings and Analysis
            ### Primary Insights Related to {keywords_str}
            - Comprehensive examination of all findings related to the specified keywords
            - Detailed analysis of patterns, trends, and themes identified in the documents
            - Specific evidence and data points directly extracted from the source materials
            - Cross-document correlations and connections related to the focus area
            
            ### Supporting Evidence and Documentation
            - Detailed review of supporting evidence from each relevant document
            - Analysis of case studies, examples, or specific instances mentioned
            - Examination of methodologies, frameworks, or approaches described
            - Assessment of data quality and reliability based on document content
            
            ### Thematic Analysis
            - In-depth exploration of major themes related to {focus_area or "the focus area"}
            - Identification of recurring concepts and their significance
            - Analysis of relationships between different aspects of the topic
            - Evaluation of consistency or contradictions across documents
            
            ## Strategic Implications and Context
            ### Business and Operational Impact
            - Detailed analysis of how findings impact operations based on document insights
            - Organizational implications explicitly discussed in the documents
            - Process improvements or strategic changes suggested in the content
            - Resource requirements, constraints, or opportunities identified
            
            ### Market and Industry Considerations
            - Industry-specific insights related to {keywords_str} found in the documents
            - Competitive landscape analysis if included in the source materials
            - Market trends, patterns, or dynamics identified in the documentation
            - Regulatory, compliance, or external factors mentioned in the documents
            
            ## Actionable Recommendations and Implementation
            - Comprehensive list of all recommendations explicitly stated in the documents
            - Detailed action items or implementation steps related to {keywords_str}
            - Priority levels, timelines, or sequencing mentioned in the source materials
            - Success metrics, evaluation criteria, or measurement approaches specified
            - Risk mitigation strategies or considerations outlined in the documents
            - Resource allocation or investment requirements identified
            
            ## Conclusion and Strategic Outlook
            - Summary of the most critical insights related to {keywords_str}
            - Long-term implications and strategic considerations based on the analysis
            - Areas requiring further investigation or additional documentation
            - Final assessment of the analysis value and its applicability to {focus_area or "the specified focus area"}
            - Key takeaways that decision-makers should prioritize
            
            **FORMATTING REQUIREMENTS**:
            - Use clear headings and subheadings as shown above
            - Include bullet points and numbered lists for clarity
            - Maintain professional, analytical tone throughout
            - Ensure each section meets the specified word count ranges
            - Ground every statement in actual document content
            - Use direct quotes where appropriate to support analysis
            - Provide comprehensive coverage while avoiding redundancy
            - Focus specifically on {keywords_str} and {focus_area or "the specified focus area"} throughout the analysis
            
            Focus on creating a thorough, fact-based analysis that serves as a comprehensive reference document for {keywords_str} and {focus_area or "the focus area"}. The summary should be detailed enough that a reader can understand the full scope and implications of the topic without needing to read the original documents directly.
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
