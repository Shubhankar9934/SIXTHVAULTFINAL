from datetime import datetime, timedelta
from typing import Optional, List
from sqlmodel import Session, select
from sqlalchemy import text
from app.database import User
from app.auth.jwt_handler import verify_token, ACCESS_TOKEN_EXPIRE_MINUTES
import uuid

class TokenModel:
    """Data model for token operations"""
    def __init__(self, id: str, user_id: str, token: str, expires_at: datetime,
                 created_at: datetime, is_active: bool, device_info: Optional[str] = None,
                 last_used: Optional[datetime] = None):
        self.id = id
        self.user_id = user_id
        self.token = token
        self.expires_at = expires_at
        self.created_at = created_at
        self.is_active = is_active
        self.device_info = device_info
        self.last_used = last_used

class TokenService:
    """Service for managing user tokens"""
    
    @staticmethod
    async def store_token(db: Session, user_id: str, token: str, 
                         device_info: Optional[str] = None) -> TokenModel:
        """Store a new token in the database"""
        expires_at = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        token_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        
        # Insert token
        insert_query = text("""
        INSERT INTO user_tokens (id, user_id, token, expires_at, device_info, created_at, is_active)
        VALUES (:id, :user_id, :token, :expires_at, :device_info, :created_at, :is_active)
        """)

        # Fetch the inserted token
        select_query = text("""
        SELECT id, user_id, token, expires_at, created_at, is_active, device_info, last_used
        FROM user_tokens
        WHERE id = :id
        """)
        
        # Execute insert
        db.execute(insert_query, {
            "id": token_id,
            "user_id": user_id,
            "token": token,
            "expires_at": expires_at,
            "device_info": device_info,
            "created_at": created_at,
            "is_active": True
        })
        db.commit()

        # Fetch inserted record
        result = db.execute(select_query, {"id": token_id}).fetchone()
        
        if result:
            id, user_id_db, token_db, expires_at_db, created_at_db, is_active_db, device_info_db, last_used_db = result
            
            # Convert datetime strings if needed
            if isinstance(expires_at_db, str):
                try:
                    expires_at_db = datetime.fromisoformat(expires_at_db.replace('Z', '+00:00'))
                except ValueError:
                    expires_at_db = datetime.fromisoformat(expires_at_db)
            
            if isinstance(created_at_db, str):
                try:
                    created_at_db = datetime.fromisoformat(created_at_db.replace('Z', '+00:00'))
                except ValueError:
                    created_at_db = datetime.fromisoformat(created_at_db)
            
            if isinstance(last_used_db, str):
                try:
                    last_used_db = datetime.fromisoformat(last_used_db.replace('Z', '+00:00'))
                except ValueError:
                    last_used_db = datetime.fromisoformat(last_used_db)
            
            return TokenModel(
                id=id,
                user_id=user_id_db,
                token=token_db,
                expires_at=expires_at_db,
                created_at=created_at_db,
                is_active=bool(is_active_db),
                device_info=device_info_db,
                last_used=last_used_db
            )
        else:
            # Fallback to creating TokenModel with original values
            return TokenModel(
                id=token_id,
                user_id=user_id,
                token=token,
                expires_at=expires_at,
                created_at=created_at,
                is_active=True,
                device_info=device_info,
                last_used=None
            )

    @staticmethod
    async def validate_token(db: Session, token: str) -> Optional[str]:
        """Validate a token and return the user_id if valid"""
        try:
            # First verify the JWT signature and expiry
            payload = verify_token(token)
            if not payload:
                print(f"Token validation failed: Invalid JWT signature or expired")
                return None
                
            query = text("""
            SELECT user_id, expires_at, is_active
            FROM user_tokens
            WHERE token = :token
            """)
            
            result = db.execute(query, {
                "token": token
            }).fetchone()
            
            if not result:
                print(f"Token validation failed: Token not found in database")
                return None
                
            user_id, expires_at, is_active = result
            current_time = datetime.utcnow()
            
            # Check if token is active
            if not is_active:
                print(f"Token validation failed: Token is inactive for user {user_id}")
                return None
            
            # Convert expires_at to datetime if it's a string (SQLite returns strings)
            if isinstance(expires_at, str):
                try:
                    expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                except ValueError:
                    # Try parsing without timezone info
                    expires_at = datetime.fromisoformat(expires_at)
            
            # Check if token is expired
            if expires_at <= current_time:
                print(f"Token validation failed: Token expired at {expires_at} for user {user_id}")
                # Clean up expired token
                await TokenService.revoke_token(db, token)
                return None
            
            # Update last_used timestamp
            update_query = text("""
            UPDATE user_tokens
            SET last_used = :current_time
            WHERE token = :token
            """)
            
            db.execute(update_query, {
                "current_time": current_time,
                "token": token
            })
            db.commit()
            
            print(f"Token validation successful for user {user_id}")
            return user_id
            
        except Exception as e:
            print(f"Token validation error: {e}")
            return None

    @staticmethod
    async def revoke_token(db: Session, token: str) -> bool:
        """Revoke a specific token"""
        query = text("""
        UPDATE user_tokens
        SET is_active = FALSE
        WHERE token = :token
        AND is_active = TRUE
        """)
        
        result = db.execute(query, {"token": token})
        db.commit()
        
        return result.rowcount > 0

    @staticmethod
    async def revoke_all_tokens(db: Session, user_id: str) -> bool:
        """Revoke all tokens for a user"""
        query = text("""
        UPDATE user_tokens
        SET is_active = FALSE
        WHERE user_id = :user_id
        AND is_active = TRUE
        """)
        
        result = db.execute(query, {"user_id": user_id})
        db.commit()
        
        return result.rowcount > 0

    @staticmethod
    async def cleanup_expired_tokens(db: Session) -> int:
        """Remove expired tokens from the database"""
        query = text("""
        DELETE FROM user_tokens
        WHERE expires_at < :current_time
        OR (is_active = FALSE AND created_at < :old_tokens)
        """)
        
        result = db.execute(query, {
            "current_time": datetime.utcnow(),
            "old_tokens": datetime.utcnow() - timedelta(days=30)  # Keep revoked tokens for 30 days
        })
        db.commit()
        
        return result.rowcount

    @staticmethod
    async def get_active_sessions(db: Session, user_id: str) -> List[TokenModel]:
        """Get all active sessions for a user"""
        query = text("""
        SELECT id, user_id, token, expires_at, created_at, is_active, device_info, last_used
        FROM user_tokens
        WHERE user_id = :user_id
        AND is_active = TRUE
        AND expires_at > :current_time
        ORDER BY created_at DESC
        """)
        
        results = db.execute(query, {
            "user_id": user_id,
            "current_time": datetime.utcnow()
        }).fetchall()
        
        # Convert string dates to datetime objects for TokenModel
        token_models = []
        for row in results:
            id, user_id, token, expires_at, created_at, is_active, device_info, last_used = row
            
            # Convert expires_at if it's a string
            if isinstance(expires_at, str):
                try:
                    expires_at = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
                except ValueError:
                    expires_at = datetime.fromisoformat(expires_at)
            
            # Convert created_at if it's a string
            if isinstance(created_at, str):
                try:
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except ValueError:
                    created_at = datetime.fromisoformat(created_at)
            
            # Convert last_used if it's a string
            if isinstance(last_used, str):
                try:
                    last_used = datetime.fromisoformat(last_used.replace('Z', '+00:00'))
                except ValueError:
                    last_used = datetime.fromisoformat(last_used)
            
            token_models.append(TokenModel(
                id=id,
                user_id=user_id,
                token=token,
                expires_at=expires_at,
                created_at=created_at,
                is_active=bool(is_active),
                device_info=device_info,
                last_used=last_used
            ))
        
        return token_models
