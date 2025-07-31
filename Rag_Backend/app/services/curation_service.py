"""
AI Curation Service - Intelligent Topic Generation and Management
Implements the hybrid approach for automatic curation with user control
"""

import time
import asyncio
from typing import List, Dict, Optional, Tuple, Set
from collections import Counter, defaultdict
from sqlmodel import Session, select
from sqlalchemy import and_, or_, func
import logging

from app.database import get_session
from app.models import (
    Document, AICuration, CurationSettings, 
    DocumentCurationMapping, CurationGenerationHistory
)
from app.services.llm_factory import get_llm
from app.deps import get_current_user

logger = logging.getLogger(__name__)

class CurationEngine:
    """Core AI engine for generating and managing document curations"""
    
    def __init__(self):
        self.llm = None
    
    async def generate_curations_from_documents(
        self, 
        documents: List[Document], 
        user_id: str,
        max_curations: int = 8,
        min_docs_per_curation: int = 2
    ) -> List[Dict]:
        """
        Generate AI curations from a collection of documents
        Uses a combination of keyword clustering and semantic analysis
        """
        if not documents:
            return []
        
        logger.info(f"Generating curations for {len(documents)} documents (user: {user_id})")
        
        # Step 1: Extract and analyze document features
        doc_features = self._extract_document_features(documents)
        
        # Step 2: Cluster documents by themes and topics
        clusters = self._cluster_documents_by_themes(doc_features, max_curations)
        
        # Step 3: Generate human-readable curation titles and descriptions
        curations = []
        for cluster_id, cluster_docs in clusters.items():
            if len(cluster_docs) >= min_docs_per_curation:
                curation = await self._generate_curation_from_cluster(
                    cluster_docs, doc_features, user_id
                )
                if curation:
                    curations.append(curation)
        
        # Step 4: Sort by confidence and limit to max_curations
        curations.sort(key=lambda x: x['confidence_score'], reverse=True)
        return curations[:max_curations]
    
    def _extract_document_features(self, documents: List[Document]) -> Dict[str, Dict]:
        """Extract features from documents for clustering"""
        features = {}
        
        for doc in documents:
            doc_id = doc.id
            
            # Combine all available text features
            all_keywords = []
            if doc.tags:
                all_keywords.extend(doc.tags)
            if doc.demo_tags:
                all_keywords.extend(doc.demo_tags)
            
            # Extract keywords from summary if available
            summary_keywords = []
            if doc.summary:
                summary_keywords = self._extract_keywords_from_text(doc.summary)
            
            features[doc_id] = {
                'document': doc,
                'keywords': all_keywords,
                'summary_keywords': summary_keywords,
                'all_terms': all_keywords + summary_keywords,
                'filename': doc.filename,
                'summary': doc.summary or ""
            }
        
        return features
    
    def _extract_keywords_from_text(self, text: str, max_keywords: int = 10) -> List[str]:
        """Simple keyword extraction from text"""
        if not text:
            return []
        
        # Simple approach: extract meaningful words
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 
            'did', 'she', 'use', 'way', 'will', 'with', 'this', 'that', 'they',
            'have', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some',
            'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make',
            'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'
        }
        
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Count frequency and return most common
        word_counts = Counter(meaningful_words)
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    def _cluster_documents_by_themes(
        self, 
        doc_features: Dict[str, Dict], 
        max_clusters: int = 8
    ) -> Dict[int, List[str]]:
        """Cluster documents by shared themes and keywords"""
        
        if not doc_features:
            return {}
        
        # Create keyword-to-documents mapping
        keyword_to_docs = defaultdict(set)
        for doc_id, features in doc_features.items():
            for keyword in features['all_terms']:
                keyword_to_docs[keyword].add(doc_id)
        
        # Find most significant keywords (appear in multiple documents)
        significant_keywords = {
            keyword: docs for keyword, docs in keyword_to_docs.items() 
            if len(docs) >= 2  # Keyword must appear in at least 2 documents
        }
        
        if not significant_keywords:
            # Fallback: create clusters based on individual document themes
            clusters = {}
            for i, doc_id in enumerate(doc_features.keys()):
                if i < max_clusters:
                    clusters[i] = [doc_id]
            return clusters
        
        # Sort keywords by document count (most shared first)
        sorted_keywords = sorted(
            significant_keywords.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        # Create clusters based on keyword overlap
        clusters = {}
        used_docs = set()
        cluster_id = 0
        
        for keyword, docs in sorted_keywords:
            if cluster_id >= max_clusters:
                break
                
            # Find documents not yet assigned to a cluster
            available_docs = docs - used_docs
            
            if len(available_docs) >= 2:  # Need at least 2 docs for a cluster
                clusters[cluster_id] = list(available_docs)
                used_docs.update(available_docs)
                cluster_id += 1
        
        # Add remaining documents to existing clusters or create new ones
        remaining_docs = set(doc_features.keys()) - used_docs
        for doc_id in remaining_docs:
            if cluster_id < max_clusters:
                clusters[cluster_id] = [doc_id]
                cluster_id += 1
            else:
                # Add to the smallest existing cluster
                if clusters:
                    smallest_cluster = min(clusters.keys(), key=lambda k: len(clusters[k]))
                    clusters[smallest_cluster].append(doc_id)
        
        return clusters
    
    async def _generate_curation_from_cluster(
        self, 
        doc_ids: List[str], 
        doc_features: Dict[str, Dict],
        user_id: str
    ) -> Optional[Dict]:
        """Generate a curation title and description for a document cluster"""
        
        if not doc_ids:
            return None
        
        # Collect all keywords and themes from the cluster
        all_keywords = []
        all_summaries = []
        filenames = []
        
        for doc_id in doc_ids:
            if doc_id in doc_features:
                features = doc_features[doc_id]
                all_keywords.extend(features['all_terms'])
                if features['summary']:
                    all_summaries.append(features['summary'])
                filenames.append(features['filename'])
        
        # Find most common keywords in this cluster
        keyword_counts = Counter(all_keywords)
        top_keywords = [kw for kw, count in keyword_counts.most_common(5)]
        
        if not top_keywords:
            return None
        
        # Generate title and description using AI
        try:
            title, description = await self._generate_curation_content(
                top_keywords, all_summaries, filenames
            )
            
            # Calculate confidence score based on keyword overlap and document count
            confidence = self._calculate_confidence_score(
                doc_ids, top_keywords, keyword_counts
            )
            
            return {
                'title': title,
                'description': description,
                'topic_keywords': top_keywords,
                'document_ids': doc_ids,
                'confidence_score': confidence,
                'document_count': len(doc_ids)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate curation content: {e}")
            # Fallback to simple title generation
            primary_keyword = top_keywords[0] if top_keywords else "Documents"
            return {
                'title': f"{primary_keyword.title()} Analysis",
                'description': f"Analysis of {len(doc_ids)} documents related to {primary_keyword}",
                'topic_keywords': top_keywords,
                'document_ids': doc_ids,
                'confidence_score': 0.5,
                'document_count': len(doc_ids)
            }
    
    async def _generate_curation_content(
        self, 
        keywords: List[str], 
        summaries: List[str], 
        filenames: List[str]
    ) -> Tuple[str, str]:
        """Use AI to generate curation title and description"""
        
        # Create prompt for AI
        keywords_str = ", ".join(keywords[:3])  # Use top 3 keywords
        files_str = ", ".join(filenames[:3])  # Show up to 3 filenames
        
        prompt = f"""
        Based on the following information, create a concise curation title and description:
        
        Key Topics: {keywords_str}
        Document Files: {files_str}
        Number of Documents: {len(summaries)}
        
        Generate:
        1. A clear, engaging title (max 50 characters)
        2. A brief description (max 150 characters)
        
        Focus on the main theme that connects these documents.
        Make it sound professional and insightful.
        
        Format your response as:
        Title: [your title]
        Description: [your description]
        """
        
        try:
            # Use the LLM factory to get a response
            if not self.llm:
                self.llm = get_llm(provider="ollama")
            response = await self.llm.chat(prompt=prompt)
            
            # Parse the response
            lines = response.strip().split('\n')
            title = "Document Analysis"
            description = f"Analysis of {len(summaries)} related documents"
            
            for line in lines:
                if line.startswith('Title:'):
                    title = line.replace('Title:', '').strip()[:50]
                elif line.startswith('Description:'):
                    description = line.replace('Description:', '').strip()[:150]
            
            return title, description
            
        except Exception as e:
            logger.error(f"AI generation failed: {e}")
            # Fallback to simple generation
            primary_topic = keywords[0] if keywords else "Documents"
            title = f"{primary_topic.title()} Insights"
            description = f"Key insights from {len(summaries)} documents about {primary_topic}"
            return title, description
    
    def _calculate_confidence_score(
        self, 
        doc_ids: List[str], 
        keywords: List[str], 
        keyword_counts: Counter
    ) -> float:
        """Calculate confidence score for a curation"""
        
        if not doc_ids or not keywords:
            return 0.0
        
        # Factors that increase confidence:
        # 1. More documents in cluster
        # 2. Higher keyword overlap
        # 3. More frequent keywords
        
        doc_count_score = min(len(doc_ids) / 5.0, 1.0)  # Max score at 5+ docs
        
        # Keyword frequency score
        total_keyword_occurrences = sum(keyword_counts.values())
        avg_keyword_frequency = total_keyword_occurrences / len(keywords) if keywords else 0
        keyword_score = min(avg_keyword_frequency / 3.0, 1.0)  # Max score at 3+ avg frequency
        
        # Keyword diversity score (more unique keywords = higher confidence)
        keyword_diversity = min(len(keywords) / 5.0, 1.0)  # Max score at 5+ keywords
        
        # Weighted combination
        confidence = (
            doc_count_score * 0.4 +
            keyword_score * 0.4 +
            keyword_diversity * 0.2
        )
        
        return round(confidence, 3)


class CurationService:
    """Main service for managing AI curations"""
    
    def __init__(self):
        self.engine = CurationEngine()
    
    async def get_user_curations(self, user_id: str, session: Session) -> List[Dict]:
        """Get all active curations for a user's tenant"""
        
        # Get user's tenant_id for proper data sharing
        from app.database import User
        user = session.exec(select(User).where(User.id == user_id)).first()
        if not user or not user.tenant_id:
            return []
        
        statement = select(AICuration).where(
            and_(
                AICuration.tenant_id == user.tenant_id,
                AICuration.status.in_(['active', 'stale'])
            )
        ).order_by(AICuration.last_updated.desc())
        
        curations = session.exec(statement).all()
        
        result = []
        for curation in curations:
            result.append({
                'id': curation.id,
                'title': curation.title,
                'description': curation.description,
                'keywords': curation.topic_keywords,
                'documentCount': curation.document_count,
                'confidence': curation.confidence_score,
                'status': curation.status,
                'lastUpdated': curation.last_updated.isoformat(),
                'autoGenerated': curation.auto_generated,
                'generationMethod': curation.generation_method
            })
        
        return result
    
    async def get_user_settings(self, user_id: str, session: Session) -> Dict:
        """Get curation settings for a user"""
        
        statement = select(CurationSettings).where(CurationSettings.owner_id == user_id)
        settings = session.exec(statement).first()
        
        if not settings:
            # Create default settings
            settings = CurationSettings(
                owner_id=user_id,
                auto_refresh=True,
                on_add="incremental",
                on_delete="auto_clean",
                change_threshold=15,
                max_curations=8,
                min_documents_per_curation=2
            )
            session.add(settings)
            session.commit()
            session.refresh(settings)
        
        return {
            'autoRefresh': settings.auto_refresh,
            'onAdd': settings.on_add,
            'onDelete': settings.on_delete,
            'changeThreshold': settings.change_threshold,
            'maxCurations': settings.max_curations,
            'minDocumentsPerCuration': settings.min_documents_per_curation
        }
    
    async def update_user_settings(
        self, 
        user_id: str, 
        settings_data: Dict, 
        session: Session
    ) -> Dict:
        """Update curation settings for a user"""
        
        statement = select(CurationSettings).where(CurationSettings.owner_id == user_id)
        settings = session.exec(statement).first()
        
        if not settings:
            settings = CurationSettings(owner_id=user_id)
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
        if 'maxCurations' in settings_data:
            settings.max_curations = settings_data['maxCurations']
        if 'minDocumentsPerCuration' in settings_data:
            settings.min_documents_per_curation = settings_data['minDocumentsPerCuration']
        
        settings.updated_at = time.time()
        session.commit()
        session.refresh(settings)
        
        return await self.get_user_settings(user_id, session)
    
    async def generate_curations_for_user(
        self, 
        user_id: str, 
        session: Session,
        generation_type: str = "manual",
        trigger_event: str = "manual_refresh"
    ) -> Dict:
        """Generate or regenerate curations for a user"""
        
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
                    'curations': [],
                    'stats': {
                        'documentsProcessed': 0,
                        'curationsCreated': 0,
                        'curationsUpdated': 0,
                        'curationsDeleted': 0,
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
                    'curations': [],
                    'stats': {
                        'documentsProcessed': 0,
                        'curationsCreated': 0,
                        'curationsUpdated': 0,
                        'curationsDeleted': 0,
                        'processingTimeMs': 0
                    }
                }
            
            # Archive existing curations
            existing_statement = select(AICuration).where(
                and_(
                    AICuration.owner_id == user_id,
                    AICuration.status == 'active'
                )
            )
            existing_curations = session.exec(existing_statement).all()
            
            for curation in existing_curations:
                curation.status = 'archived'
            
            # Generate new curations
            new_curations_data = await self.engine.generate_curations_from_documents(
                documents,
                user_id,
                max_curations=settings['maxCurations'],
                min_docs_per_curation=settings['minDocumentsPerCuration']
            )
            
            # Save new curations to database
            created_curations = []
            for curation_data in new_curations_data:
                # Get user's tenant_id for proper data sharing
                from app.database import User
                user = session.exec(select(User).where(User.id == user_id)).first()
                user_tenant_id = user.tenant_id if user else None
                
                curation = AICuration(
                    owner_id=user_id,
                    tenant_id=user_tenant_id,  # Add tenant_id for proper isolation
                    title=curation_data['title'],
                    description=curation_data['description'],
                    topic_keywords=curation_data['topic_keywords'],
                    document_ids=curation_data['document_ids'],
                    confidence_score=curation_data['confidence_score'],
                    document_count=curation_data['document_count'],
                    auto_generated=True,
                    generation_method=generation_type,
                    status='active'
                )
                session.add(curation)
                session.flush()  # Get the ID
                
                # Create document mappings
                for doc_id in curation_data['document_ids']:
                    mapping = DocumentCurationMapping(
                        document_id=doc_id,
                        curation_id=curation.id,
                        owner_id=user_id,
                        relevance_score=0.8,  # Default relevance
                        keywords_matched=curation_data['topic_keywords']
                    )
                    session.add(mapping)
                
                created_curations.append(curation)
            
            session.commit()
            
            # Record generation history
            processing_time = int((time.time() - start_time) * 1000)
            history = CurationGenerationHistory(
                owner_id=user_id,
                generation_type=generation_type,
                trigger_event=trigger_event,
                documents_processed=len(documents),
                curations_created=len(created_curations),
                curations_updated=0,
                curations_deleted=len(existing_curations),
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
            result_curations = []
            for curation in created_curations:
                result_curations.append({
                    'id': curation.id,
                    'title': curation.title,
                    'description': curation.description,
                    'keywords': curation.topic_keywords,
                    'documentCount': curation.document_count,
                    'confidence': curation.confidence_score,
                    'status': curation.status,
                    'lastUpdated': curation.last_updated.isoformat(),
                    'autoGenerated': curation.auto_generated
                })
            
            return {
                'success': True,
                'message': f'Generated {len(created_curations)} curations',
                'curations': result_curations,
                'stats': {
                    'documentsProcessed': len(documents),
                    'curationsCreated': len(created_curations),
                    'curationsUpdated': 0,
                    'curationsDeleted': len(existing_curations),
                    'processingTimeMs': processing_time
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate curations for user {user_id}: {e}")
            
            # Record failed generation
            processing_time = int((time.time() - start_time) * 1000)
            history = CurationGenerationHistory(
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
                'message': f'Failed to generate curations: {str(e)}',
                'curations': [],
                'stats': {
                    'documentsProcessed': 0,
                    'curationsCreated': 0,
                    'curationsUpdated': 0,
                    'curationsDeleted': 0,
                    'processingTimeMs': processing_time
                }
            }
    
    async def handle_document_addition(
        self, 
        user_id: str, 
        new_document_ids: List[str], 
        session: Session
    ) -> Dict:
        """Handle curation updates when documents are added"""
        
        settings = await self.get_user_settings(user_id, session)
        
        if not settings['autoRefresh']:
            # Mark existing curations as stale
            statement = select(AICuration).where(
                and_(
                    AICuration.owner_id == user_id,
                    AICuration.status == 'active'
                )
            )
            curations = session.exec(statement).all()
            
            for curation in curations:
                curation.status = 'stale'
            
            session.commit()
            
            return {
                'success': True,
                'message': 'Curations marked as stale - manual refresh needed',
                'action': 'marked_stale'
            }
        
        # Auto-refresh based on settings
        if settings['onAdd'] == 'full':
            return await self.generate_curations_for_user(
                user_id, session, 
                generation_type="full", 
                trigger_event="document_added"
            )
        elif settings['onAdd'] == 'incremental':
            # TODO: Implement incremental update logic
            # For now, fall back to full regeneration
            return await self.generate_curations_for_user(
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
        """Handle curation updates when documents are deleted"""
        
        settings = await self.get_user_settings(user_id, session)
        
        # Remove document mappings
        mapping_statement = select(DocumentCurationMapping).where(
            and_(
                DocumentCurationMapping.owner_id == user_id,
                DocumentCurationMapping.document_id.in_(deleted_document_ids)
            )
        )
        mappings = session.exec(mapping_statement).all()
        
        affected_curation_ids = set()
        for mapping in mappings:
            affected_curation_ids.add(mapping.curation_id)
            session.delete(mapping)
        
        # Update affected curations
        curations_updated = 0
        curations_deleted = 0
        
        for curation_id in affected_curation_ids:
            curation_statement = select(AICuration).where(AICuration.id == curation_id)
            curation = session.exec(curation_statement).first()
            
            if curation:
                # Remove deleted documents from curation
                curation.document_ids = [
                    doc_id for doc_id in curation.document_ids 
                    if doc_id not in deleted_document_ids
                ]
                curation.document_count = len(curation.document_ids)
                
                # Check if curation still has enough documents
                if curation.document_count < settings['minDocumentsPerCuration']:
                    curation.status = 'archived'
                    curations_deleted += 1
                else:
                    curation.last_updated = time.time()
                    if settings['onDelete'] == 'auto_clean':
                        curation.status = 'active'
                    else:
                        curation.status = 'stale'
                    curations_updated += 1
        
        session.commit()
        
        return {
            'success': True,
            'message': f'Updated {curations_updated} curations, archived {curations_deleted}',
            'stats': {
                'curationsUpdated': curations_updated,
                'curationsDeleted': curations_deleted,
                'documentsRemoved': len(deleted_document_ids)
            }
        }
    
    async def get_curation_status(self, user_id: str, session: Session) -> Dict:
        """Get the current status of curations for a user"""
        
        statement = select(AICuration).where(AICuration.owner_id == user_id)
        curations = session.exec(statement).all()
        
        status_counts = Counter([c.status for c in curations])
        
        # Check if any curations need updates
        needs_update = status_counts.get('stale', 0) > 0
        
        return {
            'totalCurations': len(curations),
            'activeCurations': status_counts.get('active', 0),
            'staleCurations': status_counts.get('stale', 0),
            'archivedCurations': status_counts.get('archived', 0),
            'needsUpdate': needs_update,
            'lastGenerated': max([c.last_updated for c in curations]).isoformat() if curations else None
        }
    
    async def create_custom_curation(
        self,
        user_id: str,
        session: Session,
        title: str,
        description: Optional[str] = None,
        keywords: List[str] = None,
        provider: str = "gemini",
        model: str = "gemini-1.5-flash"
    ) -> Dict:
        """Create a custom AI curation with user-defined topic and keywords"""
        
        if not title.strip():
            return {
                'success': False,
                'message': 'Title is required for custom curation'
            }
        
        if keywords is None:
            keywords = []
        
        try:
            # Get user documents to find relevant ones for this topic
            doc_statement = select(Document).where(Document.owner_id == user_id)
            documents = session.exec(doc_statement).all()
            
            if not documents:
                return {
                    'success': False,
                    'message': 'No documents found. Please upload documents first.'
                }
            
            # Find documents relevant to the custom topic
            relevant_docs = self._find_relevant_documents_for_topic(
                documents, keywords, title, description
            )
            
            if not relevant_docs:
                # If no specific matches, use all documents
                relevant_docs = documents
            
            # Create the custom curation
            curation = AICuration(
                owner_id=user_id,
                title=title,
                description=description or f"Custom curation about {title}",
                topic_keywords=keywords,
                document_ids=[doc.id for doc in relevant_docs],
                confidence_score=0.9,  # High confidence for user-created curations
                document_count=len(relevant_docs),
                auto_generated=False,  # This is user-created
                generation_method="custom",
                status='active'
            )
            
            session.add(curation)
            session.flush()  # Get the ID
            
            # Create document mappings
            for doc in relevant_docs:
                relevance_score = self._calculate_document_relevance_for_topic(
                    doc, keywords, title
                )
                
                mapping = DocumentCurationMapping(
                    document_id=doc.id,
                    curation_id=curation.id,
                    owner_id=user_id,
                    relevance_score=relevance_score,
                    keywords_matched=keywords
                )
                session.add(mapping)
            
            session.commit()
            session.refresh(curation)
            
            # Record generation history
            history = CurationGenerationHistory(
                owner_id=user_id,
                generation_type="custom",
                trigger_event="user_created",
                documents_processed=len(relevant_docs),
                curations_created=1,
                curations_updated=0,
                curations_deleted=0,
                processing_time_ms=0,
                success=True,
                meta_data={
                    'custom_title': title,
                    'custom_keywords': keywords,
                    'provider': provider,
                    'model': model
                }
            )
            session.add(history)
            session.commit()
            
            # Return the created curation
            return {
                'success': True,
                'message': f'Custom curation "{title}" created successfully',
                'curation': {
                    'id': curation.id,
                    'title': curation.title,
                    'description': curation.description,
                    'keywords': curation.topic_keywords,
                    'documentCount': curation.document_count,
                    'confidence': curation.confidence_score,
                    'status': curation.status,
                    'lastUpdated': curation.last_updated.isoformat(),
                    'autoGenerated': curation.auto_generated,
                    'generationMethod': curation.generation_method
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to create custom curation for user {user_id}: {e}")
            return {
                'success': False,
                'message': f'Failed to create custom curation: {str(e)}'
            }
    
    def _find_relevant_documents_for_topic(
        self,
        documents: List[Document],
        keywords: List[str],
        title: str,
        description: Optional[str] = None
    ) -> List[Document]:
        """Find documents relevant to a custom topic"""
        
        if not keywords and not title:
            return documents
        
        # Create search terms from title, description, and keywords
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
    
    def _calculate_document_relevance_for_topic(
        self,
        document: Document,
        keywords: List[str],
        title: str
    ) -> float:
        """Calculate how relevant a document is to a custom topic"""
        
        if not keywords and not title:
            return 0.5  # Default relevance
        
        # Create search terms
        search_terms = set()
        search_terms.update([kw.lower().strip() for kw in keywords])
        search_terms.update([word.lower().strip() for word in title.split() if len(word) > 2])
        
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


# Global service instance
curation_service = CurationService()
