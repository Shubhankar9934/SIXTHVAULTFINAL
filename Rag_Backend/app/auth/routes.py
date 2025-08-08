from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select, func
from datetime import datetime, timedelta
from app.database import get_session
from pydantic import BaseModel
from typing import List
from app.config import settings
from app.database import User, TempUser
from app.auth.models import (
    UserCreate, UserLogin, UserResponse, Token,
    ForgotPasswordRequest, ResetPasswordRequest, ValidateResetTokenRequest
)
from app.auth.jwt_handler import verify_password, get_password_hash, create_access_token
from app.deps import get_current_user
from app.auth.token_service import TokenService
from fastapi import Request
from datetime import datetime, timedelta
import random
import string
import secrets
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.email_service import EmailService
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

router = APIRouter(prefix="/auth", tags=["authentication"])

async def cleanup_expired_registrations(db: Session):
    """Cleanup expired temporary registrations"""
    try:
        # Find and delete all expired temp registrations
        expired = db.exec(
            select(TempUser).where(
                TempUser.verification_code_expires_at < datetime.utcnow()
            )
        ).all()
        
        for temp_user in expired:
            db.delete(temp_user)
        
        if expired:
            db.commit()
            print(f"Cleaned up {len(expired)} expired registration(s)")
    except Exception as e:
        print(f"Failed to cleanup expired registrations: {e}")
        db.rollback()

@router.post("/register", response_model=dict)
async def register(user_data: UserCreate, db: Session = Depends(get_session)):
    """Register a new user"""
    # Run cleanup task for expired registrations
    await cleanup_expired_registrations(db)
    
    email = user_data.email.lower()
    
    # Check if a verified user already exists
    existing_user = db.exec(select(User).where(User.email == email)).first()
    if existing_user and existing_user.verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check for existing temporary registration
    existing_temp = db.exec(select(TempUser).where(TempUser.email == email)).first()
    if existing_temp:
        # If there's an existing temp registration, delete it
        db.delete(existing_temp)
        db.commit()
    
    # Generate verification code
    verification_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    
    # Create temporary user
    hashed_password = get_password_hash(user_data.password)
    temp_user = TempUser(
        email=email,
        password_hash=hashed_password,
        first_name=user_data.first_name,
        last_name=user_data.last_name,
        company=user_data.company,
        verification_code=verification_code,
        verification_code_expires_at=datetime.utcnow() + timedelta(hours=24)
    )
    
    try:
        # Delete any unverified user with the same email
        unverified_user = db.exec(select(User).where(
            (User.email == email) & (User.verified == False)
        )).first()
        if unverified_user:
            db.delete(unverified_user)
        
        # Save temporary user
        db.add(temp_user)
        db.commit()
        db.refresh(temp_user)
        
        # Send verification email
        email_result = await EmailService.sendVerificationEmail(
            temp_user.email,
            temp_user.first_name,
            verification_code
        )
        
        print(f"Email service result: {email_result}")
        
        # Always return verification code in development mode
        is_development = not settings.resend_api_key or email_result.get("simulated", False)
        
        return {
            "message": "User registered successfully. Please check your email for verification.",
            "temp_id": temp_user.id,
            "email": temp_user.email,
            "verification_code": verification_code if is_development else None
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user. Please try again."
        )

@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, request: Request, db: Session = Depends(get_session)):
    """Login user and return JWT token"""
    # Find user by email
    user = db.exec(select(User).where(User.email == credentials.email.lower())).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Verify password
    if not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Check if user is verified
    if not user.verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please verify your email before logging in"
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.add(user)
    db.commit()
    
    # Create access token with role and tenant
    access_token = create_access_token(data={
        "sub": str(user.id), 
        "email": user.email,
        "role": user.role,
        "tenant_id": user.tenant_id
    })
    
    # Store token in database
    device_info = f"{request.headers.get('user-agent', 'Unknown')} - {request.client.host}"
    await TokenService.store_token(db, str(user.id), access_token, device_info)
    
    # Prepare user response
    user_response = UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        company=user.company,
        company_id=user.company_id,
        verified=user.verified,
        role=user.role,
        is_admin=user.is_admin,
        is_active=user.is_active,
        created_at=user.created_at,
        last_login=user.last_login
    )
    
    return Token(access_token=access_token, user=user_response)

class VerificationRequest(BaseModel):
    email: str
    verification_code: str

@router.post("/verify")
async def verify_email(request: VerificationRequest, db: Session = Depends(get_session)):
    """Verify user email with verification code"""
    email = request.email.lower().strip()
    
    # Sanitize and validate input verification code
    input_code = request.verification_code.strip().upper() if request.verification_code else ""
    
    # Remove any non-alphanumeric characters (spaces, dashes, etc.)
    input_code = ''.join(c for c in input_code if c.isalnum())
    
    # Validate code format
    if not input_code or len(input_code) != 6:
        print(f"Verification failed - Invalid code format. Input: '{request.verification_code}', Sanitized: '{input_code}', Length: {len(input_code)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification code must be exactly 6 characters (letters and numbers only)"
        )
    
    # Check temp_users first
    temp_user = db.exec(select(TempUser).where(TempUser.email == email)).first()
    if not temp_user:
        print(f"Verification failed - No temp user found for email: {email}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Registration not found. Please register again."
        )
    
    # Check if verification code has expired
    if temp_user.verification_code_expires_at < datetime.utcnow():
        print(f"Verification failed - Code expired for email: {email}. Expired at: {temp_user.verification_code_expires_at}, Current time: {datetime.utcnow()}")
        # Delete expired temp user
        db.delete(temp_user)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification code has expired. Please register again."
        )

    # Sanitize stored code as well (ensure consistency)
    stored_code = temp_user.verification_code.strip().upper()
    
    # Debug logging (remove in production)
    print(f"Verification attempt for email: {email}")
    print(f"Original input: '{request.verification_code}'")
    print(f"Sanitized input: '{input_code}'")
    print(f"Stored code: '{stored_code}'")
    print(f"Codes match: {stored_code == input_code}")
    
    if stored_code != input_code:
        print(f"Verification failed - Code mismatch for email: {email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification code. Please check the code and try again."
        )
    
    print(f"Verification successful for email: {email}")
    
    try:
        # Import Tenant model
        from app.database import Tenant
        
        # Create a new tenant for this user (each verified account gets its own tenant)
        tenant_name = temp_user.company or f"{temp_user.first_name} {temp_user.last_name}'s Organization"
        new_tenant = Tenant(
            name=tenant_name,
            created_at=datetime.utcnow(),
            is_active=True
        )
        
        db.add(new_tenant)
        db.commit()
        db.refresh(new_tenant)
        
        # Create permanent user as tenant admin
        new_user = User(
            email=temp_user.email,
            password_hash=temp_user.password_hash,
            first_name=temp_user.first_name,
            last_name=temp_user.last_name,
            company=temp_user.company,
            tenant_id=new_tenant.id,  # Link to their own tenant
            verified=True,
            role="admin",  # Tenant admin
            is_admin=True,  # Admin within their tenant
            is_active=True,
            created_at=temp_user.created_at,
            updated_at=datetime.utcnow(),
            created_by=None  # Self-registered users have no creator
        )
        
        # Update tenant owner
        new_tenant.owner_id = new_user.id
        
        # Add permanent user and remove temp user
        db.add(new_user)
        db.delete(temp_user)
        db.commit()
        db.refresh(new_user)
        
        try:
            # Send welcome email
            await EmailService.sendWelcomeEmail(new_user.email, new_user.first_name)
        except Exception as e:
            print(f"Failed to send welcome email: {e}")
            # Don't fail verification if welcome email fails
        
        return {"message": "Email verified successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify email. Please try again."
        )

@router.post("/resend-verification", response_model=dict)
async def resend_verification(email: str, db: Session = Depends(get_session)):
    """Resend verification code"""
    email = email.lower()
    temp_user = db.exec(select(TempUser).where(TempUser.email == email)).first()
    
    if not temp_user:
        # Return success even if registration doesn't exist to prevent email enumeration
        return {"message": "If a pending registration exists, a new verification code has been sent."}
    
    # Generate new verification code
    verification_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    temp_user.verification_code = verification_code
    temp_user.verification_code_expires_at = datetime.utcnow() + timedelta(hours=24)
    temp_user.updated_at = datetime.utcnow()
    
    try:
        db.add(temp_user)
        db.commit()
        
        # Send new verification email
        await EmailService.sendVerificationEmail(
            temp_user.email,
            temp_user.first_name,
            verification_code
        )
        
        return {"message": "New verification code sent successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to resend verification code. Please try again."
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        company=current_user.company,
        company_id=current_user.company_id,
        verified=current_user.verified,
        role=current_user.role,
        is_admin=current_user.is_admin,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )

@router.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, db: Session = Depends(get_session)):
    """Send password reset verification code"""
    try:
        # Find user by email
        user = db.exec(select(User).where(User.email == request.email.lower())).first()
        
        # Always return success to prevent email enumeration attacks
        if not user:
            return {"message": "If an account with that email exists, we have sent a verification code."}
        
        # Generate verification code
        verification_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        reset_expires = datetime.utcnow() + timedelta(minutes=15)  # 15 minutes expiry
        
        # Update user with verification code
        user.reset_token = verification_code  # Reuse reset_token field for verification code
        user.reset_token_expires_at = reset_expires
        user.updated_at = datetime.utcnow()
        
        db.add(user)
        db.commit()
        
        # Send verification code email
        try:
            await send_password_reset_verification_email(user.email, user.first_name, verification_code)
        except Exception as email_error:
            print(f"Failed to send verification email: {email_error}")
            # Don't reveal email sending failures
        
        return {"message": "If an account with that email exists, we have sent a verification code."}
        
    except Exception as e:
        print(f"Forgot password error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

class VerifyResetCodeRequest(BaseModel):
    email: str
    verification_code: str

@router.post("/verify-reset-code")
async def verify_reset_code(request: VerifyResetCodeRequest, db: Session = Depends(get_session)):
    """Verify password reset code and return reset token"""
    try:
        email = request.email.lower()
        
        # Find user by email
        user = db.exec(select(User).where(User.email == email)).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Check if verification code exists and is not expired
        if not user.reset_token or not user.reset_token_expires_at:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No verification code found. Please request a new one."
            )
        
        if user.reset_token_expires_at < datetime.utcnow():
            # Clear expired code
            user.reset_token = None
            user.reset_token_expires_at = None
            db.add(user)
            db.commit()
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Verification code has expired. Please request a new one."
            )
        
        # Sanitize and validate input verification code
        input_code = request.verification_code.strip().upper() if request.verification_code else ""
        
        # Remove any non-alphanumeric characters
        input_code = ''.join(c for c in input_code if c.isalnum())
        
        # Validate code format
        if not input_code or len(input_code) != 6:
            print(f"Password reset verification failed - Invalid code format. Input: '{request.verification_code}', Sanitized: '{input_code}'")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Verification code must be exactly 6 characters (letters and numbers only)"
            )
        
        # Sanitize stored code as well
        stored_code = user.reset_token.strip().upper()
        
        # Debug logging
        print(f"Password reset verification attempt for email: {email}")
        print(f"Original input: '{request.verification_code}'")
        print(f"Sanitized input: '{input_code}'")
        print(f"Stored code: '{stored_code}'")
        print(f"Codes match: {stored_code == input_code}")
        
        # Verify the code
        if stored_code != input_code:
            print(f"Password reset verification failed - Code mismatch for email: {email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification code. Please check the code and try again."
            )
        
        print(f"Password reset verification successful for email: {email}")
        
        # Generate a new secure reset token for password change
        reset_token = secrets.token_urlsafe(32)
        reset_expires = datetime.utcnow() + timedelta(minutes=15)  # 15 minutes for password reset
        
        # Update user with new reset token
        user.reset_token = reset_token
        user.reset_token_expires_at = reset_expires
        user.updated_at = datetime.utcnow()
        
        db.add(user)
        db.commit()
        
        return {
            "message": "Verification code verified successfully",
            "reset_token": reset_token
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Verify reset code error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/validate-reset-token")
async def validate_reset_token(request: ValidateResetTokenRequest, db: Session = Depends(get_session)):
    """Validate password reset token"""
    try:
        # Find user by reset token
        user = db.exec(select(User).where(User.reset_token == request.token)).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token"
            )
        
        # Check if token is expired
        if not user.reset_token_expires_at or user.reset_token_expires_at < datetime.utcnow():
            # Clear expired token
            user.reset_token = None
            user.reset_token_expires_at = None
            db.add(user)
            db.commit()
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reset token has expired"
            )
        
        return {
            "message": "Token is valid",
            "email": user.email
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Validate reset token error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/reset-password")
async def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_session)):
    """Reset user password with token"""
    try:
        # Validate password strength
        if len(request.password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        
        # Find user by reset token
        user = db.exec(select(User).where(User.reset_token == request.token)).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token"
            )
        
        # Check if token is expired
        if not user.reset_token_expires_at or user.reset_token_expires_at < datetime.utcnow():
            # Clear expired token
            user.reset_token = None
            user.reset_token_expires_at = None
            db.add(user)
            db.commit()
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Reset token has expired"
            )
        
        # Hash new password
        new_password_hash = get_password_hash(request.password)
        
        # Update user password and clear reset token
        user.password_hash = new_password_hash
        user.reset_token = None
        user.reset_token_expires_at = None
        user.updated_at = datetime.utcnow()
        
        db.add(user)
        db.commit()
        
        # Send password change notification
        try:
            await send_password_change_notification(user.email, user.first_name)
        except Exception as email_error:
            print(f"Failed to send password change notification: {email_error}")
            # Don't fail the request if email fails
        
        return {"message": "Password reset successful"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Reset password error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_session),
    request: Request = None
):
    """Logout user by revoking the current token"""
    try:
        auth_header = request.headers.get('Authorization') if request else None
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            revoked = await TokenService.revoke_token(db, token)
            if revoked:
                print(f"Successfully revoked token for user {current_user.id}")
            else:
                print(f"Token was already revoked or not found for user {current_user.id}")
        else:
            print(f"No authorization header found for logout request from user {current_user.id}")
            
        # Also cleanup any expired tokens for this user
        await TokenService.cleanup_expired_tokens(db)
        
        return {"message": "Logged out successfully"}
    except Exception as e:
        print(f"Logout error for user {current_user.id}: {e}")
        # Still return success to prevent client-side issues
        return {"message": "Logged out successfully"}

@router.post("/logout-all")
async def logout_all(current_user: User = Depends(get_current_user), db: Session = Depends(get_session)):
    """Logout from all devices by revoking all active tokens"""
    await TokenService.revoke_all_tokens(db, str(current_user.id))
    return {"message": "Logged out from all devices"}

@router.get("/sessions", response_model=List[dict])
async def get_active_sessions(current_user: User = Depends(get_current_user), db: Session = Depends(get_session)):
    """Get all active sessions for the current user"""
    try:
        sessions = await TokenService.get_active_sessions(db, str(current_user.id))
        return [
            {
                "id": session.id,
                "device_info": session.device_info,
                "created_at": session.created_at,
                "last_used": session.last_used,
                "expires_at": session.expires_at
            }
            for session in sessions
        ]
    except Exception as e:
        print(f"Failed to get active sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get active sessions"
        )

async def send_password_reset_verification_email(email: str, first_name: str, verification_code: str):
    """Send password reset verification code email"""
    try:
        # Send verification email using EmailService
        email_result = await EmailService.sendPasswordResetVerificationEmail(
            email,
            first_name,
            verification_code
        )
        
        print(f"Password reset verification email result: {email_result}")
        
        # For demo purposes, also print the verification code
        print("=== PASSWORD RESET VERIFICATION EMAIL ===")
        print(f"To: {email}")
        print(f"Subject: Reset Your SIXTHVAULT Password - Verification Code")
        print(f"Verification Code: {verification_code}")
        print(f"Hello {first_name}, use the code above to reset your password.")
        print("This code expires in 15 minutes.")
        print("=== END PASSWORD RESET VERIFICATION EMAIL ===")
        
        return email_result
    except Exception as e:
        print(f"Failed to send password reset verification email: {e}")
        # Fallback to console logging
        print("=== PASSWORD RESET VERIFICATION EMAIL FALLBACK ===")
        print(f"To: {email}")
        print(f"Verification Code: {verification_code}")
        print("=== END FALLBACK ===")
        return {"success": True, "simulated": True}

async def send_password_change_notification(email: str, first_name: str):
    """Send password change notification email"""
    # For demo purposes, just print the notification
    print("=== PASSWORD CHANGE NOTIFICATION ===")
    print(f"To: {email}")
    print(f"Subject: Password Changed Successfully - SIXTHVAULT")
    print(f"Hello {first_name}, your password has been changed successfully.")
    print(f"Changed on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("If you didn't make this change, contact support immediately.")
    print("=== END PASSWORD CHANGE NOTIFICATION ===")
