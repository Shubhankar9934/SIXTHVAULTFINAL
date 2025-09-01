import jwt
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
import os

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET", "fd12a34bc56d78e9fa0b12cd34ef56ab78cd90ef12ab34cd56ef78ab90cd12ef")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token with tenant and role context"""
    to_encode = data.copy()
    now = datetime.utcnow()
    
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Ensure required fields are present
    required_fields = ["sub", "email"]
    for field in required_fields:
        if field not in to_encode:
            raise ValueError(f"Missing required field: {field}")
    
    to_encode.update({
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
        "type": "access_token"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

def extract_user_id_from_token(token: str) -> Optional[int]:
    """Extract user ID from JWT token"""
    payload = verify_token(token)
    if payload:
        return payload.get("sub")
    return None

def extract_tenant_id_from_token(token: str) -> Optional[str]:
    """Extract tenant ID from JWT token"""
    payload = verify_token(token)
    if payload:
        return payload.get("tenant_id")
    return None

def extract_user_role_from_token(token: str) -> Optional[str]:
    """Extract user role from JWT token"""
    payload = verify_token(token)
    if payload:
        return payload.get("role")
    return None

def extract_user_context_from_token(token: str) -> Optional[dict]:
    """Extract complete user context from JWT token"""
    payload = verify_token(token)
    if payload:
        user_context = {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "tenant_id": payload.get("tenant_id"),
            "role": payload.get("role"),
            "is_admin": payload.get("is_admin", False)
        }
        print(f"🔍 JWT CONTEXT: Extracted user context from token:")
        print(f"   - User ID: {user_context['user_id']}")
        print(f"   - Email: {user_context['email']}")
        print(f"   - Tenant ID: {user_context['tenant_id']}")
        print(f"   - Role: {user_context['role']}")
        print(f"   - Is Admin: {user_context['is_admin']}")
        return user_context
    return None
