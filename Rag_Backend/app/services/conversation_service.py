# app/services/conversation_service.py
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlmodel import Session, select, and_, or_, desc, func
from app.models import Conversation, Message, ConversationSettings, User
from app.database import get_session, engine
import re

class ConversationService:
    """Service for managing conversations and messages"""
    
    def __init__(self):
        pass
    
    def create_conversation(
        self, 
        user_id: str, 
        tenant_id: Optional[str] = None,
        title: Optional[str] = None,
        auto_generate_title: bool = True
    ) -> Conversation:
        """Create a new conversation"""
        with Session(engine) as session:
            conversation = Conversation(
                owner_id=user_id,
                tenant_id=tenant_id,
                title=title or "New Conversation",
                message_count=0,
                is_archived=False,
                is_pinned=False,
                tags=[],
                total_tokens_used=0,
                document_ids_referenced=[]
            )
            
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            
            return conversation
    
    def get_user_conversations(
        self, 
        user_id: str, 
        tenant_id: Optional[str] = None,
        include_archived: bool = False,
        limit: int = 50,
        offset: int = 0,
        search_query: Optional[str] = None
    ) -> List[Conversation]:
        """Get user's conversations with optional filtering"""
        with Session(engine) as session:
            query = select(Conversation).where(
                and_(
                    Conversation.owner_id == user_id,
                    Conversation.tenant_id == tenant_id if tenant_id else True
                )
            )
            
            if not include_archived:
                query = query.where(Conversation.is_archived == False)
            
            if search_query:
                # Search in conversation titles
                query = query.where(
                    Conversation.title.ilike(f"%{search_query}%")
                )
            
            # Order by pinned first, then by updated_at desc
            query = query.order_by(
                desc(Conversation.is_pinned),
                desc(Conversation.updated_at)
            ).offset(offset).limit(limit)
            
            conversations = session.exec(query).all()
            return list(conversations)
    
    def get_conversation_by_id(
        self, 
        conversation_id: str, 
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Optional[Conversation]:
        """Get a specific conversation by ID"""
        with Session(engine) as session:
            query = select(Conversation).where(
                and_(
                    Conversation.id == conversation_id,
                    Conversation.owner_id == user_id,
                    Conversation.tenant_id == tenant_id if tenant_id else True
                )
            )
            
            conversation = session.exec(query).first()
            return conversation
    
    def update_conversation(
        self,
        conversation_id: str,
        user_id: str,
        tenant_id: Optional[str] = None,
        title: Optional[str] = None,
        is_archived: Optional[bool] = None,
        is_pinned: Optional[bool] = None,
        tags: Optional[List[str]] = None
    ) -> Optional[Conversation]:
        """Update conversation properties"""
        with Session(engine) as session:
            conversation = session.exec(
                select(Conversation).where(
                    and_(
                        Conversation.id == conversation_id,
                        Conversation.owner_id == user_id,
                        Conversation.tenant_id == tenant_id if tenant_id else True
                    )
                )
            ).first()
            
            if not conversation:
                return None
            
            if title is not None:
                conversation.title = title
            if is_archived is not None:
                conversation.is_archived = is_archived
            if is_pinned is not None:
                conversation.is_pinned = is_pinned
            if tags is not None:
                conversation.tags = tags
            
            conversation.updated_at = datetime.utcnow()
            
            session.add(conversation)
            session.commit()
            session.refresh(conversation)
            
            return conversation
    
    def delete_conversation(
        self,
        conversation_id: str,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> bool:
        """Delete a conversation and all its messages"""
        with Session(engine) as session:
            # First delete all messages
            messages_query = select(Message).join(Conversation).where(
                and_(
                    Conversation.id == conversation_id,
                    Conversation.owner_id == user_id,
                    Conversation.tenant_id == tenant_id if tenant_id else True
                )
            )
            messages = session.exec(messages_query).all()
            for message in messages:
                session.delete(message)
            
            # Then delete the conversation
            conversation = session.exec(
                select(Conversation).where(
                    and_(
                        Conversation.id == conversation_id,
                        Conversation.owner_id == user_id,
                        Conversation.tenant_id == tenant_id if tenant_id else True
                    )
                )
            ).first()
            
            if conversation:
                session.delete(conversation)
                session.commit()
                return True
            
            return False
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        user_id: str,
        tenant_id: Optional[str] = None,
        provider_used: Optional[str] = None,
        model_used: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        sources: Optional[List[Dict]] = None,
        context_mode: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tokens_used: Optional[int] = None,
        response_time_ms: Optional[int] = None,
        curation_id: Optional[str] = None,
        summary_id: Optional[str] = None
    ) -> Optional[Message]:
        """Add a message to a conversation"""
        with Session(engine) as session:
            # Verify conversation exists and belongs to user
            conversation = session.exec(
                select(Conversation).where(
                    and_(
                        Conversation.id == conversation_id,
                        Conversation.owner_id == user_id,
                        Conversation.tenant_id == tenant_id if tenant_id else True
                    )
                )
            ).first()
            
            if not conversation:
                return None
            
            # Create message
            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                provider_used=provider_used,
                model_used=model_used,
                document_ids=document_ids or [],
                sources=sources or [],
                context_mode=context_mode,
                message_metadata=metadata or {},
                tokens_used=tokens_used,
                response_time_ms=response_time_ms,
                curation_id=curation_id,
                summary_id=summary_id
            )
            
            session.add(message)
            
            # Update conversation metadata
            conversation.message_count += 1
            conversation.updated_at = datetime.utcnow()
            
            if tokens_used:
                conversation.total_tokens_used += tokens_used
            
            if provider_used and not conversation.primary_provider:
                conversation.primary_provider = provider_used
            
            if model_used and not conversation.primary_model:
                conversation.primary_model = model_used
            
            if document_ids:
                # Add new document IDs to referenced list
                existing_ids = set(conversation.document_ids_referenced)
                new_ids = set(document_ids)
                conversation.document_ids_referenced = list(existing_ids.union(new_ids))
            
            # Auto-generate title from first user message if needed
            if (conversation.message_count == 1 and 
                role == "user" and 
                conversation.title == "New Conversation"):
                conversation.title = self._generate_title_from_content(content)
            
            session.add(conversation)
            session.commit()
            session.refresh(message)
            
            return message
    
    def get_conversation_messages(
        self,
        conversation_id: str,
        user_id: str,
        tenant_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a conversation"""
        with Session(engine) as session:
            # Verify conversation belongs to user
            conversation = session.exec(
                select(Conversation).where(
                    and_(
                        Conversation.id == conversation_id,
                        Conversation.owner_id == user_id,
                        Conversation.tenant_id == tenant_id if tenant_id else True
                    )
                )
            ).first()
            
            if not conversation:
                return []
            
            query = select(Message).where(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at).offset(offset).limit(limit)
            
            messages = session.exec(query).all()
            return list(messages)
    
    def search_conversations(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        query: str = "",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search conversations by content"""
        with Session(engine) as session:
            # Search in conversation titles and message content
            conversations_query = select(Conversation).where(
                and_(
                    Conversation.owner_id == user_id,
                    Conversation.tenant_id == tenant_id if tenant_id else True,
                    or_(
                        Conversation.title.ilike(f"%{query}%"),
                        Conversation.id.in_(
                            select(Message.conversation_id).where(
                                Message.content.ilike(f"%{query}%")
                            )
                        )
                    )
                )
            ).order_by(desc(Conversation.updated_at)).limit(limit)
            
            conversations = session.exec(conversations_query).all()
            
            results = []
            for conv in conversations:
                # Get matching messages for context
                matching_messages = session.exec(
                    select(Message).where(
                        and_(
                            Message.conversation_id == conv.id,
                            Message.content.ilike(f"%{query}%")
                        )
                    ).limit(3)
                ).all()
                
                results.append({
                    "conversation": conv,
                    "matching_messages": list(matching_messages)
                })
            
            return results
    
    def get_conversation_settings(
        self,
        user_id: str
    ) -> ConversationSettings:
        """Get user's conversation settings"""
        with Session(engine) as session:
            settings = session.exec(
                select(ConversationSettings).where(
                    ConversationSettings.owner_id == user_id
                )
            ).first()
            
            if not settings:
                # Create default settings
                settings = ConversationSettings(
                    owner_id=user_id,
                    auto_save_conversations=True,
                    auto_generate_titles=True,
                    max_conversations=100,
                    auto_archive_after_days=30,
                    default_provider="gemini",
                    default_model="gemini-1.5-flash",
                    save_curation_conversations=True,
                    save_summary_conversations=True
                )
                session.add(settings)
                session.commit()
                session.refresh(settings)
            
            return settings
    
    def update_conversation_settings(
        self,
        user_id: str,
        **kwargs
    ) -> ConversationSettings:
        """Update user's conversation settings"""
        with Session(engine) as session:
            settings = session.exec(
                select(ConversationSettings).where(
                    ConversationSettings.owner_id == user_id
                )
            ).first()
            
            if not settings:
                settings = ConversationSettings(owner_id=user_id)
            
            # Update provided fields
            for key, value in kwargs.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
            
            settings.updated_at = datetime.utcnow()
            
            session.add(settings)
            session.commit()
            session.refresh(settings)
            
            return settings
    
    def cleanup_old_conversations(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        days_old: int = 30
    ) -> int:
        """Archive old conversations based on settings"""
        with Session(engine) as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            conversations = session.exec(
                select(Conversation).where(
                    and_(
                        Conversation.owner_id == user_id,
                        Conversation.tenant_id == tenant_id if tenant_id else True,
                        Conversation.updated_at < cutoff_date,
                        Conversation.is_archived == False,
                        Conversation.is_pinned == False
                    )
                )
            ).all()
            
            count = 0
            for conv in conversations:
                conv.is_archived = True
                conv.updated_at = datetime.utcnow()
                session.add(conv)
                count += 1
            
            session.commit()
            return count
    
    def get_conversation_stats(
        self,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get conversation statistics for user"""
        with Session(engine) as session:
            # Total conversations
            total_conversations = session.exec(
                select(func.count(Conversation.id)).where(
                    and_(
                        Conversation.owner_id == user_id,
                        Conversation.tenant_id == tenant_id if tenant_id else True
                    )
                )
            ).first()
            
            # Active conversations
            active_conversations = session.exec(
                select(func.count(Conversation.id)).where(
                    and_(
                        Conversation.owner_id == user_id,
                        Conversation.tenant_id == tenant_id if tenant_id else True,
                        Conversation.is_archived == False
                    )
                )
            ).first()
            
            # Total messages
            total_messages = session.exec(
                select(func.count(Message.id)).join(Conversation).where(
                    and_(
                        Conversation.owner_id == user_id,
                        Conversation.tenant_id == tenant_id if tenant_id else True
                    )
                )
            ).first()
            
            # Total tokens used
            total_tokens = session.exec(
                select(func.sum(Conversation.total_tokens_used)).where(
                    and_(
                        Conversation.owner_id == user_id,
                        Conversation.tenant_id == tenant_id if tenant_id else True
                    )
                )
            ).first()
            
            return {
                "total_conversations": total_conversations or 0,
                "active_conversations": active_conversations or 0,
                "archived_conversations": (total_conversations or 0) - (active_conversations or 0),
                "total_messages": total_messages or 0,
                "total_tokens_used": total_tokens or 0
            }
    
    def _generate_title_from_content(self, content: str, max_length: int = 50) -> str:
        """Generate a conversation title from the first message content"""
        # Clean the content
        clean_content = re.sub(r'[^\w\s]', '', content)
        words = clean_content.split()
        
        if not words:
            return "New Conversation"
        
        # Take first few words that fit within max_length
        title_words = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > max_length:
                break
            title_words.append(word)
            current_length += len(word) + 1
        
        title = " ".join(title_words)
        
        if len(title) < len(content):
            title += "..."
        
        return title or "New Conversation"

# Global instance
conversation_service = ConversationService()
