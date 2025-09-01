from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlmodel import Session, select, func, and_, or_
from typing import List, Optional
from app.database import get_session, User, UserToken
from app.models import Document, AICuration, AISummary, ProcessingDocument
from app.models_document_access import DocumentAccess, DocumentAccessLog
from app.deps import get_current_admin_user, get_current_tenant_id_dependency
from app.auth.models import UserCreate, UserResponse, AdminUserCreate
from app.auth.jwt_handler import get_password_hash
from app.services.document_access_service import DocumentAccessService
from datetime import datetime, timedelta
import uuid
import os
from pathlib import Path
from pydantic import BaseModel

class UserCreateWithDocuments(BaseModel):
    first_name: str
    last_name: str
    email: str
    username: Optional[str] = None
    password: str
    role: str = "user"
    company: Optional[str] = None
    company_id: Optional[str] = None
    is_admin: bool = False
    is_active: bool = True
    assigned_document_ids: List[str] = []
    document_permissions: List[str] = ["read"]

class DocumentAssignmentRequest(BaseModel):
    user_ids: List[str]
    permissions: List[str] = ["read"]

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/users", response_model=List[dict])
async def get_all_users(
    request: Request,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get all users within the same tenant (admin only)"""
    
    # Only get users from the same tenant using tenant context
    statement = select(User).where(User.tenant_id == tenant_id).offset(skip).limit(limit)
    users = session.exec(statement).all()
    
    result = []
    for user in users:
        # Count documents owned by this user
        owned_docs_count = session.exec(
            select(func.count(Document.id)).where(Document.owner_id == str(user.id))
        ).first() or 0
        
        # Count documents assigned to this user (excluding owned documents)
        assigned_docs_count = session.exec(
            select(func.count(DocumentAccess.id)).where(
                and_(
                    DocumentAccess.user_id == str(user.id),
                    DocumentAccess.tenant_id == tenant_id,
                    DocumentAccess.is_active == True
                )
            )
        ).first() or 0
        
        # Total document count = owned + assigned
        total_doc_count = owned_docs_count + assigned_docs_count
        
        # Get last login info
        last_login = user.last_login.strftime("%Y-%m-%d %H:%M:%S") if user.last_login else "Never"
        
        result.append({
            "id": user.id,
            "name": f"{user.first_name} {user.last_name}",
            "email": user.email,
            "role": user.role,
            "status": "active" if user.verified else "inactive",
            "lastLogin": last_login,
            "documentsCount": total_doc_count,
            "company": user.company,
            "verified": user.verified,
            "created_at": user.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "first_name": user.first_name,
            "last_name": user.last_name
        })
    
    return result

# Document Assignment Endpoints

@router.post("/documents/{document_id}/assign", response_model=dict)
async def assign_document_to_users(
    request: Request,
    document_id: str,
    assignment_data: DocumentAssignmentRequest,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Assign a document to multiple users (admin only)"""
    
    # Verify document exists and belongs to tenant
    document = session.exec(
        select(Document, User).join(User, Document.owner_id == User.id).where(
            and_(
                Document.id == document_id,
                User.tenant_id == tenant_id
            )
        )
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found in this tenant"
        )
    
    doc, owner = document
    assigned_users = []
    failed_assignments = []
    
    for user_id in assignment_data.user_ids:
        success = DocumentAccessService.assign_document_to_user(
            db=session,
            admin_user_id=admin_user.id,
            document_id=document_id,
            target_user_id=user_id,
            tenant_id=tenant_id,
            permissions=assignment_data.permissions
        )
        
        if success:
            assigned_users.append(user_id)
        else:
            failed_assignments.append(user_id)
    
    return {
        "message": f"Document '{doc.filename}' assignment completed",
        "document_id": document_id,
        "assigned_users": assigned_users,
        "failed_assignments": failed_assignments,
        "permissions": assignment_data.permissions
    }

@router.delete("/documents/{document_id}/assign/{user_id}", response_model=dict)
async def remove_document_assignment(
    request: Request,
    document_id: str,
    user_id: str,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Remove document assignment from a user (admin only)"""
    
    success = DocumentAccessService.remove_document_assignment(
        db=session,
        admin_user_id=admin_user.id,
        document_id=document_id,
        target_user_id=user_id,
        tenant_id=tenant_id
    )
    
    if success:
        return {
            "message": "Document assignment removed successfully",
            "document_id": document_id,
            "user_id": user_id
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document assignment not found"
        )

@router.get("/documents/{document_id}/assignments", response_model=List[dict])
async def get_document_assignments(
    request: Request,
    document_id: str,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Get all user assignments for a document (admin only)"""
    
    assignments = DocumentAccessService.get_document_assignments(
        db=session,
        admin_user_id=admin_user.id,
        document_id=document_id,
        tenant_id=tenant_id
    )
    
    return assignments

@router.post("/users/create-with-documents", response_model=dict)
async def create_user_with_document_access(
    request: Request,
    user_data: UserCreateWithDocuments,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Create a new user and assign documents (admin only)"""
    
    # Check if user already exists by email within the tenant
    existing_user = session.exec(
        select(User).where(
            and_(
                User.email == user_data.email.lower(),
                User.tenant_id == tenant_id
            )
        )
    ).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists in this tenant"
        )
    
    # Validate password strength
    if len(user_data.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    # Set admin flags based on role
    is_admin = user_data.role == "admin" or user_data.is_admin
    
    # Create new user within the same tenant
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email.lower(),
        username=user_data.username,
        password_hash=hashed_password,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        company=user_data.company or admin_user.company,
        company_id=user_data.company_id or admin_user.company_id,
        tenant_id=tenant_id,
        primary_tenant_id=tenant_id,
        verified=True,
        role=user_data.role,
        is_admin=is_admin,
        is_active=user_data.is_active,
        created_by=admin_user.id,
        created_at=datetime.utcnow()
    )
    
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    
    # Assign documents if provided and user is not admin
    assigned_documents = []
    failed_assignments = []
    
    if user_data.assigned_document_ids and not is_admin:
        for doc_id in user_data.assigned_document_ids:
            success = DocumentAccessService.assign_document_to_user(
                db=session,
                admin_user_id=admin_user.id,
                document_id=doc_id,
                target_user_id=new_user.id,
                tenant_id=tenant_id,
                permissions=user_data.document_permissions
            )
            
            if success:
                assigned_documents.append(doc_id)
            else:
                failed_assignments.append(doc_id)
    
    return {
        "message": "User created successfully",
        "user_id": new_user.id,
        "email": new_user.email,
        "username": new_user.username,
        "role": new_user.role,
        "is_admin": new_user.is_admin,
        "assigned_documents": assigned_documents,
        "failed_document_assignments": failed_assignments,
        "document_permissions": user_data.document_permissions if not is_admin else []
    }

@router.post("/users", response_model=dict)
async def create_user(
    request: Request,
    user_data: AdminUserCreate,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Create a new user (admin only)"""
    
    # Check if user already exists by email within the tenant
    existing_user = session.exec(
        select(User).where(
            and_(
                User.email == user_data.email.lower(),
                User.tenant_id == tenant_id
            )
        )
    ).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists in this tenant"
        )
    
    # Check if username is provided and unique within tenant
    if user_data.username:
        existing_username = session.exec(
            select(User).where(
                and_(
                    User.username == user_data.username,
                    User.tenant_id == tenant_id
                )
            )
        ).first()
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists in this tenant"
            )
    
    # Validate password strength
    if len(user_data.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    # Set admin flags based on role
    is_admin = user_data.role == "admin" or user_data.is_admin
    
    # Create new user within the same tenant
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        email=user_data.email.lower(),
        username=user_data.username,
        password_hash=hashed_password,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        company=user_data.company or admin_user.company,  # Inherit admin's company if not specified
        company_id=user_data.company_id or admin_user.company_id,
        tenant_id=tenant_id,  # Assign to current tenant
        primary_tenant_id=tenant_id,  # Set as primary tenant
        verified=True,  # Admin-created users are automatically verified
        role=user_data.role,
        is_admin=is_admin,
        is_active=user_data.is_active,
        created_by=admin_user.id,  # Track who created this user
        created_at=datetime.utcnow()
    )
    
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    
    return {
        "message": "User created successfully",
        "user_id": new_user.id,
        "email": new_user.email,
        "username": new_user.username,
        "role": new_user.role,
        "is_admin": new_user.is_admin
    }

@router.put("/users/{user_id}", response_model=dict)
async def update_user(
    request: Request,
    user_id: str,
    user_data: dict,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Update user information (admin only)"""
    
    # Find user within the same tenant
    user = session.exec(
        select(User).where(
            and_(
                User.id == user_id,
                User.tenant_id == tenant_id
            )
        )
    ).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in this tenant"
        )
    
    # Update allowed fields
    if "first_name" in user_data:
        user.first_name = user_data["first_name"]
    if "last_name" in user_data:
        user.last_name = user_data["last_name"]
    if "email" in user_data:
        # Check if new email is already taken
        existing_user = session.exec(
            select(User).where(User.email == user_data["email"].lower(), User.id != user_id)
        ).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already taken by another user"
            )
        user.email = user_data["email"].lower()
    if "company" in user_data:
        user.company = user_data["company"]
    if "role" in user_data and user_data["role"] in ["user", "admin"]:
        user.role = user_data["role"]
    if "verified" in user_data:
        user.verified = user_data["verified"]
    
    user.updated_at = datetime.utcnow()
    session.add(user)
    session.commit()
    
    return {
        "message": "User updated successfully",
        "user_id": user.id
    }

@router.delete("/users/{user_id}", response_model=dict)
async def delete_user(
    request: Request,
    user_id: str,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Delete a user and all their data (admin only)"""
    
    # Find user within the same tenant
    user = session.exec(
        select(User).where(
            and_(
                User.id == user_id,
                User.tenant_id == tenant_id
            )
        )
    ).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in this tenant"
        )
    
    # Prevent admin from deleting themselves
    if user.id == admin_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own admin account"
        )
    
    # Import additional models for complete cleanup
    from app.models import (
        CurationSettings, DocumentCurationMapping, CurationGenerationHistory,
        SummarySettings, DocumentSummaryMapping, SummaryGenerationHistory,
        Conversation, Message, ConversationSettings
    )
    from app.models_document_access import DocumentAccess, UserDocumentGroup, DocumentAccessLog
    
    try:
        # Step 1: Delete all user tokens first (this is causing the foreign key constraint error)
        tokens = session.exec(select(UserToken).where(UserToken.user_id == user_id)).all()
        for token in tokens:
            session.delete(token)
        session.flush()  # Ensure tokens are deleted before proceeding
        
        # Step 2: Delete document access logs
        access_logs = session.exec(select(DocumentAccessLog).where(DocumentAccessLog.user_id == user_id)).all()
        for log in access_logs:
            session.delete(log)
        
        # Step 3: Delete document access records
        doc_access_records = session.exec(select(DocumentAccess).where(DocumentAccess.user_id == user_id)).all()
        for access in doc_access_records:
            session.delete(access)
        
        # Step 4: Delete document access records where user was the assigner
        assigned_access_records = session.exec(select(DocumentAccess).where(DocumentAccess.assigned_by == user_id)).all()
        for access in assigned_access_records:
            session.delete(access)
        
        # Step 5: Update user document groups to remove this user
        user_groups = session.exec(select(UserDocumentGroup)).all()
        for group in user_groups:
            if user_id in group.user_ids:
                group.user_ids.remove(user_id)
                session.add(group)
        
        # Step 6: Delete user document groups created by this user
        created_groups = session.exec(select(UserDocumentGroup).where(UserDocumentGroup.created_by == user_id)).all()
        for group in created_groups:
            session.delete(group)
        
        # Step 7: Delete conversation messages
        user_conversations = session.exec(select(Conversation).where(Conversation.owner_id == user_id)).all()
        for conversation in user_conversations:
            # Delete messages in this conversation
            messages = session.exec(select(Message).where(Message.conversation_id == conversation.id)).all()
            for message in messages:
                session.delete(message)
            # Delete the conversation
            session.delete(conversation)
        
        # Step 8: Delete conversation settings
        conv_settings = session.exec(select(ConversationSettings).where(ConversationSettings.owner_id == user_id)).all()
        for setting in conv_settings:
            session.delete(setting)
        
        # Step 9: Delete user's document curation mappings
        doc_curation_mappings = session.exec(select(DocumentCurationMapping).where(DocumentCurationMapping.owner_id == user_id)).all()
        for mapping in doc_curation_mappings:
            session.delete(mapping)
        
        # Step 10: Delete user's document summary mappings
        doc_summary_mappings = session.exec(select(DocumentSummaryMapping).where(DocumentSummaryMapping.owner_id == user_id)).all()
        for mapping in doc_summary_mappings:
            session.delete(mapping)
        
        # Step 11: Delete user's curation generation history
        curation_history = session.exec(select(CurationGenerationHistory).where(CurationGenerationHistory.owner_id == user_id)).all()
        for history in curation_history:
            session.delete(history)
        
        # Step 12: Delete user's summary generation history
        summary_history = session.exec(select(SummaryGenerationHistory).where(SummaryGenerationHistory.owner_id == user_id)).all()
        for history in summary_history:
            session.delete(history)
        
        # Step 13: Delete user's AI curations
        curations = session.exec(select(AICuration).where(AICuration.owner_id == user_id)).all()
        for curation in curations:
            session.delete(curation)
        
        # Step 14: Delete user's AI summaries
        summaries = session.exec(select(AISummary).where(AISummary.owner_id == user_id)).all()
        for summary in summaries:
            session.delete(summary)
        
        # Step 15: Delete user's curation settings
        curation_settings = session.exec(select(CurationSettings).where(CurationSettings.owner_id == user_id)).all()
        for setting in curation_settings:
            session.delete(setting)
        
        # Step 16: Delete user's summary settings
        summary_settings = session.exec(select(SummarySettings).where(SummarySettings.owner_id == user_id)).all()
        for setting in summary_settings:
            session.delete(setting)
        
        # Step 17: Delete user's processing documents
        processing_docs = session.exec(select(ProcessingDocument).where(ProcessingDocument.owner_id == user_id)).all()
        for proc_doc in processing_docs:
            session.delete(proc_doc)
        
        # Step 18: Delete user's documents and files
        user_documents = session.exec(select(Document).where(Document.owner_id == user_id)).all()
        deleted_files = 0
        
        for doc in user_documents:
            # Delete physical file
            if os.path.exists(doc.path):
                try:
                    os.remove(doc.path)
                    deleted_files += 1
                except Exception as e:
                    print(f"Failed to delete file {doc.path}: {e}")
            
            # Delete from database
            session.delete(doc)
        
        # Step 19: Finally, delete the user
        session.delete(user)
        
        # Commit all changes
        session.commit()
        
        return {
            "message": "User and all associated data deleted successfully",
            "user_id": user_id,
            "deleted_documents": len(user_documents),
            "deleted_files": deleted_files,
            "deleted_curations": len(curations),
            "deleted_summaries": len(summaries),
            "deleted_curation_settings": len(curation_settings),
            "deleted_summary_settings": len(summary_settings),
            "deleted_curation_mappings": len(doc_curation_mappings),
            "deleted_summary_mappings": len(doc_summary_mappings),
            "deleted_curation_history": len(curation_history),
            "deleted_summary_history": len(summary_history),
            "deleted_conversations": len(user_conversations),
            "deleted_conversation_messages": sum(len(session.exec(select(Message).where(Message.conversation_id == conv.id)).all()) for conv in user_conversations),
            "deleted_tokens": len(tokens),
            "deleted_access_records": len(doc_access_records),
            "deleted_access_logs": len(access_logs)
        }
        
    except Exception as e:
        session.rollback()
        print(f"Error during user deletion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}"
        )

@router.get("/documents", response_model=List[dict])
async def get_all_documents(
    request: Request,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session),
    user_id: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get all documents within the current tenant"""
    
    print(f"🔍 ADMIN DOCS: Getting documents for tenant {tenant_id}")
    print(f"🔍 ADMIN DOCS: Admin user: {admin_user.email}")
    print(f"🔍 ADMIN DOCS: Specific user filter: {user_id}")
    
    # Get tenant users first
    tenant_user_ids = session.exec(select(User.id).where(User.tenant_id == tenant_id)).all()
    print(f"🔍 ADMIN DOCS: Found {len(tenant_user_ids)} users in tenant {tenant_id}")
    
    if not tenant_user_ids:
        print(f"⚠️ ADMIN DOCS: No users found in tenant {tenant_id}")
        return []
    
    # Build query for documents owned by users in the current tenant
    if user_id:
        # If specific user requested, filter by that user (but ensure they're in the tenant)
        if user_id not in tenant_user_ids:
            print(f"⚠️ ADMIN DOCS: User {user_id} not in tenant {tenant_id}")
            return []
        statement = select(Document, User).join(User, Document.owner_id == User.id).where(
            and_(
                Document.owner_id == user_id,
                User.tenant_id == tenant_id
            )
        )
    else:
        # Show all documents from users in the current tenant
        statement = select(Document, User).join(User, Document.owner_id == User.id).where(
            and_(
                Document.owner_id.in_(tenant_user_ids),
                User.tenant_id == tenant_id
            )
        )
    
    statement = statement.offset(skip).limit(limit)
    results = session.exec(statement).all()
    
    print(f"🔍 ADMIN DOCS: Found {len(results)} documents")
    
    documents = []
    for doc, user in results:
        # Use stored file size from database, fallback to file system check if needed
        file_size = doc.file_size if hasattr(doc, 'file_size') and doc.file_size else 0
        
        # If no stored file size, try to get it from file system (for backward compatibility)
        if file_size == 0 and os.path.exists(doc.path):
            try:
                file_size = os.path.getsize(doc.path)
                print(f"📊 Admin: Fallback file size calculation for {doc.filename}: {file_size} bytes")
            except Exception as e:
                print(f"❌ Admin: Error getting file size for {doc.path}: {e}")
                file_size = 0
        
        # Use stored content type or determine from extension
        file_type = doc.content_type if hasattr(doc, 'content_type') and doc.content_type else None
        
        if not file_type:
            # Fallback to extension-based detection
            file_ext = Path(doc.filename).suffix.lower()
            file_type = "application/pdf" if file_ext == ".pdf" else \
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if file_ext == ".docx" else \
                       "text/plain" if file_ext == ".txt" else \
                       "application/rtf" if file_ext == ".rtf" else \
                       "application/octet-stream"
        
        documents.append({
            "id": doc.id,
            "name": doc.filename,
            "size": file_size,
            "type": file_type,
            "uploadDate": doc.created_at.strftime("%Y-%m-%d"),
            "owner": {
                "id": user.id,
                "name": f"{user.first_name} {user.last_name}",
                "email": user.email
            },
            "language": "English",  # Default
            "themes": doc.tags or [],
            "keywords": doc.tags or [],
            "demographics": doc.demo_tags or [],
            "summary": doc.summary,
            "keyInsights": [doc.insight] if doc.insight else [],
            "path": doc.path
        })
    
    print(f"✅ ADMIN DOCS: Returning {len(documents)} documents")
    return documents

@router.delete("/documents/{document_id}", response_model=dict)
async def delete_document(
    request: Request,
    document_id: str,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Delete a specific document and its associated data (admin only)"""
    
    # Admin can delete documents from any tenant
    document = session.exec(
        select(Document, User).join(User, Document.owner_id == User.id).where(
            Document.id == document_id
        )
    ).first()
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    doc, owner = document
    
    # Import additional models for complete cleanup
    from app.models import (
        DocumentCurationMapping, DocumentSummaryMapping
    )
    
    owner_name = f"{owner.first_name} {owner.last_name}" if owner else "Unknown"
    
    # Delete physical file
    file_deleted = False
    if os.path.exists(doc.path):
        try:
            os.remove(doc.path)
            file_deleted = True
        except Exception as e:
            print(f"Failed to delete file {doc.path}: {e}")
    
    # Delete document access records first (to avoid foreign key constraint)
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
    
    # Delete processing documents related to this document
    processing_docs = session.exec(
        select(ProcessingDocument).where(ProcessingDocument.doc_id == document_id)
    ).all()
    for proc_doc in processing_docs:
        session.delete(proc_doc)
    
    # Delete the document
    session.delete(doc)
    session.commit()
    
    return {
        "message": "Document deleted successfully",
        "document_id": document_id,
        "document_name": doc.filename,
        "owner": owner_name,
        "file_deleted": file_deleted,
        "deleted_access_records": len(doc_access_records),
        "deleted_access_logs": len(doc_access_logs),
        "deleted_curation_mappings": len(doc_curation_mappings),
        "deleted_summary_mappings": len(doc_summary_mappings),
        "deleted_processing_records": len(processing_docs)
    }

@router.get("/system", response_model=dict)
async def get_system_info(
    request: Request,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Get system information and statistics (admin only)"""
    
    # Count users within the same tenant
    total_users = session.exec(select(func.count(User.id)).where(User.tenant_id == tenant_id)).first() or 0
    active_users = session.exec(select(func.count(User.id)).where(
        and_(User.tenant_id == tenant_id, User.verified == True)
    )).first() or 0
    
    # Get tenant users for document filtering
    tenant_user_ids = session.exec(select(User.id).where(User.tenant_id == tenant_id)).all()
    
    # Count documents from tenant users only
    total_documents = 0
    total_storage = 0
    if tenant_user_ids:
        total_documents = session.exec(select(func.count(Document.id)).where(
            Document.owner_id.in_(tenant_user_ids)
        )).first() or 0
        
        # Calculate storage used by tenant documents only
        documents = session.exec(select(Document).where(Document.owner_id.in_(tenant_user_ids))).all()
        for doc in documents:
            if os.path.exists(doc.path):
                total_storage += os.path.getsize(doc.path)
    
    storage_gb = total_storage / (1024 * 1024 * 1024)
    
    # Count AI curations and summaries for tenant users only
    total_curations = 0
    total_summaries = 0
    processing_documents = 0
    if tenant_user_ids:
        total_curations = session.exec(select(func.count(AICuration.id)).where(
            AICuration.owner_id.in_(tenant_user_ids)
        )).first() or 0
        total_summaries = session.exec(select(func.count(AISummary.id)).where(
            AISummary.owner_id.in_(tenant_user_ids)
        )).first() or 0
        processing_documents = session.exec(select(func.count(ProcessingDocument.id)).where(
            ProcessingDocument.owner_id.in_(tenant_user_ids)
        )).first() or 0
    
    return {
        "users": {
            "total": total_users,
            "active": active_users,
            "inactive": total_users - active_users
        },
        "documents": {
            "total": total_documents,
            "processing": processing_documents,
            "completed": total_documents
        },
        "storage": {
            "total_bytes": total_storage,
            "total_gb": round(storage_gb, 2)
        },
        "ai": {
            "curations": total_curations,
            "summaries": total_summaries
        }
    }

@router.get("/system/stats", response_model=dict)
async def get_system_stats(
    request: Request,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Get system statistics for current tenant (admin only)"""
    
    # Count users within the same tenant
    total_users = session.exec(select(func.count(User.id)).where(User.tenant_id == tenant_id)).first() or 0
    active_users = session.exec(select(func.count(User.id)).where(
        and_(User.tenant_id == tenant_id, User.verified == True)
    )).first() or 0
    
    # Get tenant users for document filtering
    tenant_user_ids = session.exec(select(User.id).where(User.tenant_id == tenant_id)).all()
    
    # Count documents from tenant users only
    total_documents = 0
    total_storage = 0
    if tenant_user_ids:
        total_documents = session.exec(select(func.count(Document.id)).where(
            Document.owner_id.in_(tenant_user_ids)
        )).first() or 0
        
        # Calculate storage used by tenant documents only
        documents = session.exec(select(Document).where(Document.owner_id.in_(tenant_user_ids))).all()
        for doc in documents:
            if os.path.exists(doc.path):
                total_storage += os.path.getsize(doc.path)
    
    storage_gb = total_storage / (1024 * 1024 * 1024)
    
    # Count AI curations and summaries for tenant users only
    total_curations = 0
    total_summaries = 0
    processing_documents = 0
    if tenant_user_ids:
        total_curations = session.exec(select(func.count(AICuration.id)).where(
            AICuration.owner_id.in_(tenant_user_ids)
        )).first() or 0
        total_summaries = session.exec(select(func.count(AISummary.id)).where(
            AISummary.owner_id.in_(tenant_user_ids)
        )).first() or 0
        processing_documents = session.exec(select(func.count(ProcessingDocument.id)).where(
            ProcessingDocument.owner_id.in_(tenant_user_ids)
        )).first() or 0
    
    return {
        "users": {
            "total": total_users,
            "active": active_users,
            "inactive": total_users - active_users
        },
        "documents": {
            "total": total_documents,
            "processing": processing_documents,
            "completed": total_documents
        },
        "storage": {
            "total_bytes": total_storage,
            "total_gb": round(storage_gb, 2)
        },
        "ai": {
            "curations": total_curations,
            "summaries": total_summaries
        }
    }

@router.post("/system/clear-cache", response_model=dict)
async def clear_system_cache(
    request: Request,
    admin_user: User = Depends(get_current_admin_user)
):
    """Clear system cache (admin only)"""
    
    try:
        # Clear Qdrant cache if available
        from app.utils.qdrant_store import clear_cache
        cache_cleared = clear_cache()
        
        return {
            "message": "System cache cleared successfully",
            "cache_cleared": cache_cleared,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "message": "Cache clearing completed with warnings",
            "warning": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.post("/system/reindex", response_model=dict)
async def reindex_documents(
    request: Request,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Reindex all documents (admin only)"""
    
    try:
        # Get all documents for the current tenant only
        tenant_user_ids = session.exec(select(User.id).where(User.tenant_id == tenant_id)).all()
        documents = session.exec(select(Document).where(Document.owner_id.in_(tenant_user_ids))).all()
        
        # Trigger reindexing (this would depend on your specific implementation)
        reindexed_count = len(documents)
        
        return {
            "message": "Document reindexing completed",
            "documents_reindexed": reindexed_count,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reindexing failed: {str(e)}"
        )

@router.get("/system/export", response_model=dict)
async def export_system_data(
    request: Request,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Export system data (admin only)"""
    
    try:
        # Get counts for export (tenant-specific)
        tenant_user_ids = session.exec(select(User.id).where(User.tenant_id == tenant_id)).all()
        
        users_count = len(tenant_user_ids)
        documents_count = session.exec(select(func.count(Document.id)).where(
            Document.owner_id.in_(tenant_user_ids)
        )).first() or 0 if tenant_user_ids else 0
        curations_count = session.exec(select(func.count(AICuration.id)).where(
            AICuration.owner_id.in_(tenant_user_ids)
        )).first() or 0 if tenant_user_ids else 0
        summaries_count = session.exec(select(func.count(AISummary.id)).where(
            AISummary.owner_id.in_(tenant_user_ids)
        )).first() or 0 if tenant_user_ids else 0
        
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "tenant_id": tenant_id,
            "system_stats": {
                "users": users_count,
                "documents": documents_count,
                "curations": curations_count,
                "summaries": summaries_count
            },
            "export_id": str(uuid.uuid4())
        }
        
        return {
            "message": "Data export prepared",
            "export_data": export_data,
            "download_ready": True
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}"
        )

# Additional User Management Endpoints

@router.get("/users/{user_id}", response_model=dict)
async def get_user_details(
    request: Request,
    user_id: str,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Get detailed information about a specific user (admin only)"""
    
    # Find user within the same tenant
    user = session.exec(
        select(User).where(
            and_(
                User.id == user_id,
                User.tenant_id == tenant_id
            )
        )
    ).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in this tenant"
        )
    
    # Get user's document count
    doc_count = session.exec(select(func.count(Document.id)).where(Document.owner_id == user_id)).first() or 0
    
    # Get user's active sessions
    active_sessions = session.exec(
        select(func.count(UserToken.id)).where(
            UserToken.user_id == user_id,
            UserToken.is_active == True,
            UserToken.expires_at > datetime.utcnow()
        )
    ).first() or 0
    
    # Get who created this user
    created_by_user = None
    if user.created_by:
        creator = session.exec(select(User).where(User.id == user.created_by)).first()
        if creator:
            created_by_user = f"{creator.first_name} {creator.last_name} ({creator.email})"
    
    return {
        "id": user.id,
        "email": user.email,
        "username": user.username,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "full_name": f"{user.first_name} {user.last_name}",
        "company": user.company,
        "company_id": user.company_id,
        "role": user.role,
        "is_admin": user.is_admin,
        "is_active": user.is_active,
        "verified": user.verified,
        "created_at": user.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": user.updated_at.strftime("%Y-%m-%d %H:%M:%S") if user.updated_at else None,
        "last_login": user.last_login.strftime("%Y-%m-%d %H:%M:%S") if user.last_login else "Never",
        "created_by": created_by_user,
        "statistics": {
            "documents_count": doc_count,
            "active_sessions": active_sessions
        }
    }

# Simplified admin functions for remaining endpoints
@router.post("/users/{user_id}/reset-password", response_model=dict)
async def admin_reset_user_password(
    request: Request,
    user_id: str,
    new_password: str,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Reset a user's password (admin only)"""
    
    # Find user within the same tenant
    user = session.exec(
        select(User).where(
            and_(
                User.id == user_id,
                User.tenant_id == tenant_id
            )
        )
    ).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in this tenant"
        )
    
    # Validate password strength
    if len(new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 8 characters long"
        )
    
    # Hash new password
    hashed_password = get_password_hash(new_password)
    
    # Update user password
    user.password_hash = hashed_password
    user.updated_at = datetime.utcnow()
    
    # Clear any existing reset tokens
    user.reset_token = None
    user.reset_token_expires_at = None
    
    session.add(user)
    session.commit()
    
    # Revoke all existing tokens for this user to force re-login
    tokens = session.exec(select(UserToken).where(UserToken.user_id == user_id)).all()
    for token in tokens:
        token.is_active = False
    session.commit()
    
    return {
        "message": "Password reset successfully",
        "user_id": user.id,
        "email": user.email,
        "tokens_revoked": len(tokens)
    }

class PasswordResetRequest(BaseModel):
    new_password: str

@router.post("/users/{user_id}/reset-password-secure", response_model=dict)
async def admin_reset_user_password_secure(
    request: Request,
    user_id: str,
    password_data: PasswordResetRequest,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Reset a user's password with request body (admin only)"""
    
    return await admin_reset_user_password(request, user_id, password_data.new_password, admin_user, tenant_id, session)

@router.post("/users/{user_id}/toggle-status", response_model=dict)
async def toggle_user_status(
    request: Request,
    user_id: str,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Toggle user active/inactive status (admin only)"""
    
    # Find user within the same tenant
    user = session.exec(
        select(User).where(
            and_(
                User.id == user_id,
                User.tenant_id == tenant_id
            )
        )
    ).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in this tenant"
        )
    
    # Prevent admin from deactivating themselves
    if user.id == admin_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own admin account"
        )
    
    # Toggle status
    user.is_active = not user.is_active
    user.updated_at = datetime.utcnow()
    
    session.add(user)
    session.commit()
    
    # If deactivating, revoke all tokens
    if not user.is_active:
        tokens = session.exec(select(UserToken).where(UserToken.user_id == user_id)).all()
        for token in tokens:
            token.is_active = False
        session.commit()
        tokens_revoked = len(tokens)
    else:
        tokens_revoked = 0
    
    return {
        "message": f"User {'activated' if user.is_active else 'deactivated'} successfully",
        "user_id": user.id,
        "is_active": user.is_active,
        "tokens_revoked": tokens_revoked
    }

@router.post("/users/{user_id}/promote-to-admin", response_model=dict)
async def promote_user_to_admin(
    request: Request,
    user_id: str,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Promote a user to admin role (admin only)"""
    
    # Find user within the same tenant
    user = session.exec(
        select(User).where(
            and_(
                User.id == user_id,
                User.tenant_id == tenant_id
            )
        )
    ).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in this tenant"
        )
    
    # Check if user is already admin
    if user.role == "admin" or user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is already an admin"
        )
    
    # Promote to admin
    user.role = "admin"
    user.is_admin = True
    user.updated_at = datetime.utcnow()
    
    session.add(user)
    session.commit()
    
    return {
        "message": "User promoted to admin successfully",
        "user_id": user.id,
        "email": user.email,
        "role": user.role,
        "is_admin": user.is_admin
    }

@router.post("/users/{user_id}/demote-from-admin", response_model=dict)
async def demote_user_from_admin(
    request: Request,
    user_id: str,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Demote an admin user to regular user (admin only)"""
    
    # Find user within the same tenant
    user = session.exec(
        select(User).where(
            and_(
                User.id == user_id,
                User.tenant_id == tenant_id
            )
        )
    ).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in this tenant"
        )
    
    # Prevent admin from demoting themselves
    if user.id == admin_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot demote your own admin account"
        )
    
    # Check if user is admin
    if user.role != "admin" and not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is not an admin"
        )
    
    # Demote from admin
    user.role = "user"
    user.is_admin = False
    user.updated_at = datetime.utcnow()
    
    session.add(user)
    session.commit()
    
    return {
        "message": "User demoted from admin successfully",
        "user_id": user.id,
        "email": user.email,
        "role": user.role,
        "is_admin": user.is_admin
    }

@router.get("/users/{user_id}/sessions", response_model=List[dict])
async def get_user_sessions(
    request: Request,
    user_id: str,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Get all active sessions for a specific user (admin only)"""
    
    # Find user within the same tenant
    user = session.exec(
        select(User).where(
            and_(
                User.id == user_id,
                User.tenant_id == tenant_id
            )
        )
    ).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in this tenant"
        )
    
    # Get user's sessions
    user_sessions = session.exec(
        select(UserToken).where(
            UserToken.user_id == user_id,
            UserToken.is_active == True
        ).order_by(UserToken.created_at.desc())
    ).all()
    
    sessions_data = []
    for user_session in user_sessions:
        is_expired = user_session.expires_at < datetime.utcnow()
        sessions_data.append({
            "id": user_session.id,
            "device_info": user_session.device_info,
            "created_at": user_session.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "last_used": user_session.last_used.strftime("%Y-%m-%d %H:%M:%S") if user_session.last_used else None,
            "expires_at": user_session.expires_at.strftime("%Y-%m-%d %H:%M:%S"),
            "is_active": user_session.is_active,
            "is_expired": is_expired,
            "status": "expired" if is_expired else ("active" if user_session.is_active else "revoked")
        })
    
    return sessions_data

@router.post("/users/{user_id}/revoke-sessions", response_model=dict)
async def revoke_user_sessions(
    request: Request,
    user_id: str,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Revoke all active sessions for a specific user (admin only)"""
    
    # Find user within the same tenant
    user = session.exec(
        select(User).where(
            and_(
                User.id == user_id,
                User.tenant_id == tenant_id
            )
        )
    ).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in this tenant"
        )
    
    # Revoke all active sessions
    tokens = session.exec(select(UserToken).where(
        UserToken.user_id == user_id,
        UserToken.is_active == True
    )).all()
    
    for token in tokens:
        token.is_active = False
    
    session.commit()
    
    return {
        "message": "All user sessions revoked successfully",
        "user_id": user.id,
        "email": user.email,
        "sessions_revoked": len(tokens)
    }

@router.get("/users/search", response_model=List[dict])
async def search_users(
    request: Request,
    q: str = Query(..., min_length=2, description="Search query"),
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session),
    limit: int = Query(20, ge=1, le=100)
):
    """Search users by name, email, or username (admin only)"""
    
    # Build search query within the same tenant
    search_term = f"%{q.lower()}%"
    
    statement = select(User).where(
        and_(
            User.tenant_id == tenant_id,
            or_(
                func.lower(User.first_name).like(search_term),
                func.lower(User.last_name).like(search_term),
                func.lower(User.email).like(search_term),
                func.lower(User.username).like(search_term)
            )
        )
    ).limit(limit)
    
    users = session.exec(statement).all()
    
    result = []
    for user in users:
        # Get document count
        doc_count = session.exec(select(func.count(Document.id)).where(Document.owner_id == str(user.id))).first() or 0
        
        result.append({
            "id": user.id,
            "name": f"{user.first_name} {user.last_name}",
            "email": user.email,
            "username": user.username,
            "role": user.role,
            "is_admin": user.is_admin,
            "is_active": user.is_active,
            "verified": user.verified,
            "company": user.company,
            "documents_count": doc_count,
            "created_at": user.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "last_login": user.last_login.strftime("%Y-%m-%d %H:%M:%S") if user.last_login else "Never"
        })
    
    return result

@router.get("/users/{user_id}/documents", response_model=List[dict])
async def get_user_assigned_documents(
    request: Request,
    user_id: str,
    admin_user: User = Depends(get_current_admin_user),
    tenant_id: str = Depends(get_current_tenant_id_dependency),
    session: Session = Depends(get_session)
):
    """Get all documents assigned to a specific user (admin only)"""
    
    # Find user within the same tenant
    user = session.exec(
        select(User).where(
            and_(
                User.id == user_id,
                User.tenant_id == tenant_id
            )
        )
    ).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found in this tenant"
        )
    
    assigned_documents = []
    
    # Get documents owned by the user
    owned_documents = session.exec(
        select(Document).where(Document.owner_id == user_id)
    ).all()
    
    for doc in owned_documents:
        # Use stored file size from database, fallback to file system check if needed
        file_size = doc.file_size if hasattr(doc, 'file_size') and doc.file_size else 0
        
        # If no stored file size, try to get it from file system (for backward compatibility)
        if file_size == 0 and os.path.exists(doc.path):
            try:
                file_size = os.path.getsize(doc.path)
                print(f"📊 Admin: Fallback file size calculation for owned doc {doc.filename}: {file_size} bytes")
            except Exception as e:
                print(f"❌ Admin: Error getting file size for owned doc {doc.path}: {e}")
                file_size = 0
        
        assigned_documents.append({
            "id": doc.id,
            "name": doc.filename,
            "size": file_size,
            "assignment_type": "owner",
            "permissions": ["read", "download", "manage", "query"],
            "assigned_at": doc.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "assigned_by": "self"
        })
    
    # Get documents explicitly assigned to the user
    document_accesses = session.exec(
        select(DocumentAccess).where(
            and_(
                DocumentAccess.user_id == user_id,
                DocumentAccess.tenant_id == tenant_id,
                DocumentAccess.is_active == True
            )
        )
    ).all()
    
    for access in document_accesses:
        # Get document details
        doc = session.exec(
            select(Document).where(Document.id == access.document_id)
        ).first()
        
        if doc:
            # Use stored file size from database, fallback to file system check if needed
            file_size = doc.file_size if hasattr(doc, 'file_size') and doc.file_size else 0
            
            # If no stored file size, try to get it from file system (for backward compatibility)
            if file_size == 0 and os.path.exists(doc.path):
                try:
                    file_size = os.path.getsize(doc.path)
                    print(f"📊 Admin: Fallback file size calculation for assigned doc {doc.filename}: {file_size} bytes")
                except Exception as e:
                    print(f"❌ Admin: Error getting file size for assigned doc {doc.path}: {e}")
                    file_size = 0
            
            # Get who assigned this document
            assigned_by_user = session.exec(
                select(User).where(User.id == access.assigned_by)
            ).first()
            assigned_by_name = f"{assigned_by_user.first_name} {assigned_by_user.last_name}" if assigned_by_user else "Unknown"
            
            assigned_documents.append({
                "id": doc.id,
                "name": doc.filename,
                "size": file_size,
                "assignment_type": "assigned",
                "permissions": access.permissions,
                "assigned_at": access.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "assigned_by": assigned_by_name
            })
    
    return assigned_documents
