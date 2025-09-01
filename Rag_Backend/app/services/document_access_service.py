from sqlmodel import Session, select, and_, or_
from typing import List, Optional, Dict, Any
from app.database import User
from app.models import Document
from app.models_document_access import DocumentAccess, DocumentAccessLog
from app.tenant_models import TenantUser
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DocumentAccessService:
    """
    Service to handle document access control and permissions within tenants
    """
    
    @staticmethod
    def can_user_access_document(
        db: Session, 
        user_id: str, 
        document_id: str, 
        tenant_id: str,
        permission_type: str = "read"
    ) -> bool:
        """
        Check if a user can access a specific document
        
        Args:
            db: Database session
            user_id: User ID
            document_id: Document ID
            tenant_id: Tenant ID
            permission_type: Type of permission (read, download, manage)
        
        Returns:
            bool: True if user has access, False otherwise
        """
        try:
            # Get user info
            user = db.exec(select(User).where(User.id == user_id)).first()
            if not user:
                return False
            
            # Get document info
            document = db.exec(
                select(Document).where(
                    and_(
                        Document.id == document_id,
                        Document.tenant_id == tenant_id
                    )
                )
            ).first()
            
            if not document:
                return False
            
            # Admin users in the same tenant can access all documents
            if user.is_admin and user.tenant_id == tenant_id:
                return True
            
            # Document owner can always access their documents
            if document.owner_id == user_id:
                return True
            
            # For non-admin users, check if document is explicitly assigned
            document_access = db.exec(
                select(DocumentAccess).where(
                    and_(
                        DocumentAccess.document_id == document_id,
                        DocumentAccess.user_id == user_id,
                        DocumentAccess.tenant_id == tenant_id,
                        DocumentAccess.is_active == True
                    )
                )
            ).first()
            
            if document_access:
                # Check if user has the required permission
                return permission_type in document_access.permissions
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking document access: {e}")
            return False
    
    @staticmethod
    def get_user_accessible_documents(
        db: Session, 
        user_id: str, 
        tenant_id: str,
        permission_type: str = "read"
    ) -> List[Document]:
        """
        Get all documents a user can access within their tenant
        
        Args:
            db: Database session
            user_id: User ID
            tenant_id: Tenant ID
            permission_type: Type of permission (read, download, manage)
        
        Returns:
            List[Document]: List of accessible documents
        """
        try:
            # Get user info
            user = db.exec(select(User).where(User.id == user_id)).first()
            if not user:
                return []
            
            # Admin users can access all documents in their tenant
            if user.is_admin and user.tenant_id == tenant_id:
                documents = db.exec(
                    select(Document).where(Document.tenant_id == tenant_id)
                ).all()
                return list(documents)
            
            # For non-admin users, get documents they own AND documents assigned to them
            accessible_document_ids = set()
            
            # 1. Documents they own
            owned_documents = db.exec(
                select(Document).where(
                    and_(
                        Document.owner_id == user_id,
                        Document.tenant_id == tenant_id
                    )
                )
            ).all()
            
            for doc in owned_documents:
                accessible_document_ids.add(doc.id)
            
            # 2. Documents explicitly assigned to them
            assigned_accesses = db.exec(
                select(DocumentAccess).where(
                    and_(
                        DocumentAccess.user_id == user_id,
                        DocumentAccess.tenant_id == tenant_id,
                        DocumentAccess.is_active == True
                    )
                )
            ).all()
            
            for access in assigned_accesses:
                # Check if user has the required permission
                if permission_type in access.permissions:
                    accessible_document_ids.add(access.document_id)
            
            # Get all accessible documents
            if accessible_document_ids:
                documents = db.exec(
                    select(Document).where(
                        and_(
                            Document.id.in_(accessible_document_ids),
                            Document.tenant_id == tenant_id
                        )
                    )
                ).all()
                return list(documents)
            else:
                return []
            
        except Exception as e:
            logger.error(f"Error getting user accessible documents: {e}")
            return []
    
    @staticmethod
    def assign_document_to_user(
        db: Session,
        admin_user_id: str,
        document_id: str,
        target_user_id: str,
        tenant_id: str,
        permissions: List[str] = None
    ) -> bool:
        """
        Assign a document to a user (admin only)
        
        Args:
            db: Database session
            admin_user_id: Admin user ID performing the assignment
            document_id: Document ID to assign
            target_user_id: User ID to assign document to
            tenant_id: Tenant ID
            permissions: List of permissions (read, download, manage)
        
        Returns:
            bool: True if assignment successful, False otherwise
        """
        try:
            # Verify admin user
            admin_user = db.exec(select(User).where(User.id == admin_user_id)).first()
            if not admin_user or not admin_user.is_admin or admin_user.tenant_id != tenant_id:
                logger.warning(f"Unauthorized document assignment attempt by user {admin_user_id}")
                return False
            
            # Verify document exists and belongs to tenant
            document = db.exec(
                select(Document).where(
                    and_(
                        Document.id == document_id,
                        Document.tenant_id == tenant_id
                    )
                )
            ).first()
            
            if not document:
                logger.warning(f"Document {document_id} not found in tenant {tenant_id}")
                return False
            
            # Verify target user exists and belongs to tenant
            target_user = db.exec(
                select(User).where(
                    and_(
                        User.id == target_user_id,
                        User.tenant_id == tenant_id
                    )
                )
            ).first()
            
            if not target_user:
                logger.warning(f"Target user {target_user_id} not found in tenant {tenant_id}")
                return False
            
            # Check if assignment already exists
            existing_access = db.exec(
                select(DocumentAccess).where(
                    and_(
                        DocumentAccess.document_id == document_id,
                        DocumentAccess.user_id == target_user_id,
                        DocumentAccess.tenant_id == tenant_id
                    )
                )
            ).first()
            
            if existing_access:
                # Update existing assignment
                existing_access.permissions = permissions or ["read"]
                existing_access.is_active = True
                existing_access.updated_at = datetime.utcnow()
                db.add(existing_access)
            else:
                # Create new assignment
                new_access = DocumentAccess(
                    document_id=document_id,
                    user_id=target_user_id,
                    tenant_id=tenant_id,
                    assigned_by=admin_user_id,
                    permissions=permissions or ["read"]
                )
                db.add(new_access)
            
            db.commit()
            logger.info(f"Document assignment: {document_id} -> {target_user_id} by admin {admin_user_id}")
            logger.info(f"Permissions: {permissions or ['read']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error assigning document to user: {e}")
            return False
    
    @staticmethod
    def remove_document_assignment(
        db: Session,
        admin_user_id: str,
        document_id: str,
        target_user_id: str,
        tenant_id: str
    ) -> bool:
        """
        Remove document assignment from a user (admin only)
        
        Args:
            db: Database session
            admin_user_id: Admin user ID performing the removal
            document_id: Document ID to remove assignment for
            target_user_id: User ID to remove assignment from
            tenant_id: Tenant ID
        
        Returns:
            bool: True if removal successful, False otherwise
        """
        try:
            # Verify admin user
            admin_user = db.exec(select(User).where(User.id == admin_user_id)).first()
            if not admin_user or not admin_user.is_admin or admin_user.tenant_id != tenant_id:
                logger.warning(f"Unauthorized document assignment removal attempt by user {admin_user_id}")
                return False
            
            # Find and deactivate the assignment
            document_access = db.exec(
                select(DocumentAccess).where(
                    and_(
                        DocumentAccess.document_id == document_id,
                        DocumentAccess.user_id == target_user_id,
                        DocumentAccess.tenant_id == tenant_id
                    )
                )
            ).first()
            
            if document_access:
                document_access.is_active = False
                document_access.updated_at = datetime.utcnow()
                db.add(document_access)
                db.commit()
                logger.info(f"Document assignment removed: {document_id} from {target_user_id} by admin {admin_user_id}")
                return True
            else:
                logger.warning(f"No assignment found to remove: {document_id} from {target_user_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error removing document assignment: {e}")
            return False
    
    @staticmethod
    def get_document_assignments(
        db: Session,
        admin_user_id: str,
        document_id: str,
        tenant_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get all user assignments for a document (admin only)
        
        Args:
            db: Database session
            admin_user_id: Admin user ID requesting the assignments
            document_id: Document ID
            tenant_id: Tenant ID
        
        Returns:
            List[Dict]: List of user assignments with permissions
        """
        try:
            # Verify admin user
            admin_user = db.exec(select(User).where(User.id == admin_user_id)).first()
            if not admin_user or not admin_user.is_admin or admin_user.tenant_id != tenant_id:
                logger.warning(f"Unauthorized document assignments request by user {admin_user_id}")
                return []
            
            # Verify document exists and belongs to tenant
            document = db.exec(
                select(Document).where(
                    and_(
                        Document.id == document_id,
                        Document.tenant_id == tenant_id
                    )
                )
            ).first()
            
            if not document:
                return []
            
            # TODO: Implement document assignment table query
            # For now, return document owner as the only assignment
            owner = db.exec(select(User).where(User.id == document.owner_id)).first()
            if owner:
                return [{
                    "user_id": owner.id,
                    "email": owner.email,
                    "first_name": owner.first_name,
                    "last_name": owner.last_name,
                    "permissions": ["read", "download", "manage"],
                    "assignment_type": "owner",
                    "assigned_at": document.created_at
                }]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting document assignments: {e}")
            return []
    
    @staticmethod
    def get_user_document_permissions(
        db: Session,
        user_id: str,
        document_id: str,
        tenant_id: str
    ) -> List[str]:
        """
        Get specific permissions a user has for a document
        
        Args:
            db: Database session
            user_id: User ID
            document_id: Document ID
            tenant_id: Tenant ID
        
        Returns:
            List[str]: List of permissions (read, download, manage)
        """
        try:
            # Get user info
            user = db.exec(select(User).where(User.id == user_id)).first()
            if not user:
                return []
            
            # Get document info
            document = db.exec(
                select(Document).where(
                    and_(
                        Document.id == document_id,
                        Document.tenant_id == tenant_id
                    )
                )
            ).first()
            
            if not document:
                return []
            
            # Admin users have all permissions
            if user.is_admin and user.tenant_id == tenant_id:
                return ["read", "download", "manage"]
            
            # Document owner has all permissions
            if document.owner_id == user_id:
                return ["read", "download", "manage"]
            
            # TODO: Check document assignment table for specific permissions
            # For now, non-owners have no permissions
            return []
            
        except Exception as e:
            logger.error(f"Error getting user document permissions: {e}")
            return []
    
    @staticmethod
    def can_user_upload_documents(
        db: Session,
        user_id: str,
        tenant_id: str
    ) -> bool:
        """
        Check if a user can upload documents (admin only)
        
        Args:
            db: Database session
            user_id: User ID
            tenant_id: Tenant ID
        
        Returns:
            bool: True if user can upload, False otherwise
        """
        try:
            user = db.exec(select(User).where(User.id == user_id)).first()
            if not user:
                return False
            
            # Only admin users can upload documents
            return user.is_admin and user.tenant_id == tenant_id
            
        except Exception as e:
            logger.error(f"Error checking upload permissions: {e}")
            return False
    
    @staticmethod
    def can_user_manage_users(
        db: Session,
        user_id: str,
        tenant_id: str
    ) -> bool:
        """
        Check if a user can manage other users (admin only)
        
        Args:
            db: Database session
            user_id: User ID
            tenant_id: Tenant ID
        
        Returns:
            bool: True if user can manage users, False otherwise
        """
        try:
            user = db.exec(select(User).where(User.id == user_id)).first()
            if not user:
                return False
            
            # Only admin users can manage other users
            return user.is_admin and user.tenant_id == tenant_id
            
        except Exception as e:
            logger.error(f"Error checking user management permissions: {e}")
            return False
