from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    email: str
    username: Optional[str] = None
    first_name: str
    last_name: str
    company: Optional[str] = None
    company_id: Optional[str] = None
    tenant_id: Optional[str] = None

class UserCreate(UserBase):
    password: str
    role: Optional[str] = "user"
    is_admin: Optional[bool] = False
    is_active: Optional[bool] = True

class AdminUserCreate(BaseModel):
    email: str
    username: Optional[str] = None
    first_name: str
    last_name: str
    password: str
    company: Optional[str] = None
    company_id: Optional[str] = None
    tenant_id: Optional[str] = None
    role: Optional[str] = "user"
    is_admin: Optional[bool] = False
    is_active: Optional[bool] = True

class UserLogin(BaseModel):
    email: str
    password: str

class UserResponse(UserBase):
    id: str
    verified: bool
    role: str
    is_admin: bool
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

class Token(BaseModel):
    access_token: str
    user: UserResponse

class ForgotPasswordRequest(BaseModel):
    email: str

class ValidateResetTokenRequest(BaseModel):
    token: str

class VerifyResetCodeRequest(BaseModel):
    email: str
    verification_code: str

class ResetPasswordRequest(BaseModel):
    token: str
    password: str
