import json
import httpx
from typing import Optional, Dict, Any
from datetime import datetime
from app.config import settings

class EmailService:
    RESEND_API_URL = "https://api.resend.com/emails"

    @staticmethod
    def _get_api_key() -> str:
        """Get Resend API key from settings"""
        api_key = settings.resend_api_key
        if not api_key:
            raise ValueError("RESEND_API_KEY is not configured in settings")
        return api_key

    @staticmethod
    def _get_base_url() -> str:
        """Get base URL"""
        return "http://localhost:3000"  # Frontend URL

    @staticmethod
    async def _send_email(to: str, subject: str, html_content: str, text_content: str) -> Dict[str, Any]:
        """Send email using Resend API"""
        try:
            api_key = EmailService._get_api_key()
            print(f"Sending email with Resend API key: {api_key[:4]}...{api_key[-4:]}")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "from": f"SIXTHVAULT <verify@{settings.email_domain}>",
                "to": [to],
                "subject": subject,
                "html": html_content,
                "text": text_content,
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    EmailService.RESEND_API_URL,
                    headers=headers,
                    json=payload
                )
                
                if not response.is_success:
                    error_data = response.json()
                    print(f"Resend API Error Response - Status: {response.status_code}")
                    print(f"Response Headers: {dict(response.headers)}")
                    print(f"Response Body: {error_data}")
                    
                    # Check for testing limitation error
                    if "testing emails to your own email address" in error_data.get("message", ""):
                        print("=== RESEND TESTING MODE ===")
                        print(f"You can only send emails to your verified email address during testing.")
                        print(f"Email would be sent to: {to}")
                        print(f"Subject: {subject}")
                        print("=== END RESEND TESTING MODE ===")
                        return {
                            "success": True,
                            "simulated": True,
                            "messageId": f"resend-test-{datetime.utcnow().timestamp()}",
                            "message": "Email simulated due to Resend testing limitations",
                        }
                    
                    raise Exception(f"Resend API error: {error_data.get('message', response.text)}")
                
                result = response.json()
                print(f"Email sent successfully via Resend: {result.get('id')}")
                return {
                    "success": True,
                    "messageId": result.get("id"),
                    "simulated": False,
                }
                
        except Exception as e:
            print("Failed to send email:", str(e))
            
            # Fallback to console logging
            print("=== EMAIL SERVICE FALLBACK ===")
            print(f"Email would be sent to: {to}")
            print(f"Subject: {subject}")
            print("=== END FALLBACK ===")
            
            return {
                "success": True,
                "messageId": f"fallback-{datetime.utcnow().timestamp()}",
                "simulated": True,
                "message": "Email simulated due to service error",
            }

    @staticmethod
    async def sendVerificationEmail(to: str, first_name: str, verification_code: str) -> Dict[str, Any]:
        """Send verification email to user"""
        base_url = EmailService._get_base_url()
        verify_url = f"{base_url}/verify?email={to}"
        
        html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Verify Your Account</title>
            </head>
            <body style="margin: 0; padding: 0; font-family: Arial, sans-serif; background-color: #f5f5f5;">
              <div style="max-width: 600px; margin: 0 auto; background-color: white; padding: 40px 20px;">
                <!-- Header -->
                <div style="text-align: center; margin-bottom: 40px;">
                  <h1 style="color: #1f2937; font-size: 28px; margin: 0; font-weight: bold;">
                    SIXTHVAULT
                  </h1>
                  <p style="color: #6b7280; margin: 10px 0 0 0;">AI-Powered Document Intelligence</p>
                </div>

                <!-- Main Content -->
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 12px; text-align: center; margin-bottom: 30px;">
                  <h2 style="color: white; font-size: 24px; margin: 0 0 20px 0;">
                    Welcome to SIXTHVAULT, {first_name}!
                  </h2>
                  <p style="color: #e5e7eb; margin: 0 0 30px 0; font-size: 16px;">
                    Please verify your email address to activate your account
                  </p>
                  
                  <!-- Verification Code -->
                  <div style="background-color: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <p style="color: #374151; margin: 0 0 10px 0; font-size: 14px; font-weight: bold;">
                      Your Verification Code:
                    </p>
                    <div style="font-size: 32px; font-weight: bold; color: #1f2937; letter-spacing: 8px; font-family: monospace;">
                      {verification_code}
                    </div>
                  </div>

                  <!-- Verify Button -->
                  <a href="{verify_url}" 
                     style="display: inline-block; background-color: #10b981; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; margin-top: 20px;">
                    Verify Account
                  </a>
                </div>

                <!-- Instructions -->
                <div style="background-color: #f9fafb; padding: 20px; border-radius: 8px; margin-bottom: 30px;">
                  <h3 style="color: #374151; margin: 0 0 15px 0; font-size: 18px;">How to verify:</h3>
                  <ol style="color: #6b7280; margin: 0; padding-left: 20px;">
                    <li style="margin-bottom: 8px;">Click the "Verify Account" button above, or</li>
                    <li style="margin-bottom: 8px;">Copy the 6-digit code and paste it on the verification page</li>
                    <li>Your account will be activated immediately</li>
                  </ol>
                </div>

                <!-- Footer -->
                <div style="text-align: center; padding-top: 30px; border-top: 1px solid #e5e7eb;">
                  <p style="color: #9ca3af; font-size: 14px; margin: 0 0 10px 0;">
                    This verification code expires in 24 hours.
                  </p>
                  <p style="color: #9ca3af; font-size: 12px; margin: 0;">
                    If you didn't create an account with SIXTHVAULT, please ignore this email.
                  </p>
                </div>
              </div>
            </body>
            </html>
        """
        
        text_content = f"""
Welcome to SIXTHVAULT, {first_name}!

Your verification code is: {verification_code}

Please visit {verify_url} to verify your account.

This code expires in 24 hours.

If you didn't create an account with SIXTHVAULT, please ignore this email.
        """
        
        result = await EmailService._send_email(
            to=to,
            subject="Verify Your SIXTHVAULT Account",
            html_content=html_content,
            text_content=text_content
        )
        
        if result["success"]:
            result["verificationCode"] = verification_code
        
        return result

    @staticmethod
    async def sendWelcomeEmail(to: str, first_name: str) -> Dict[str, Any]:
        """Send welcome email to verified user"""
        base_url = EmailService._get_base_url()
        vault_url = f"{base_url}/vault"
        
        html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Welcome to SIXTHVAULT</title>
            </head>
            <body style="margin: 0; padding: 0; font-family: Arial, sans-serif; background-color: #f5f5f5;">
              <div style="max-width: 600px; margin: 0 auto; background-color: white; padding: 40px 20px;">
                <!-- Header -->
                <div style="text-align: center; margin-bottom: 40px;">
                  <h1 style="color: #1f2937; font-size: 28px; margin: 0; font-weight: bold;">
                    SIXTHVAULT
                  </h1>
                  <p style="color: #6b7280; margin: 10px 0 0 0;">AI-Powered Document Intelligence</p>
                </div>

                <!-- Welcome Message -->
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); padding: 30px; border-radius: 12px; text-align: center; margin-bottom: 30px;">
                  <h2 style="color: white; font-size: 24px; margin: 0 0 15px 0;">
                    🎉 Welcome {first_name}!
                  </h2>
                  <p style="color: #d1fae5; margin: 0; font-size: 16px;">
                    Your SIXTHVAULT account is now active and ready to use!
                  </p>
                </div>

                <!-- Get Started -->
                <div style="text-align: center; margin-bottom: 30px;">
                  <a href="{vault_url}" 
                     style="display: inline-block; background-color: #3b82f6; color: white; padding: 15px 30px; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 16px;">
                    Start Using SIXTHVAULT
                  </a>
                </div>

                <!-- Support -->
                <div style="text-align: center; padding-top: 30px; border-top: 1px solid #e5e7eb;">
                  <p style="color: #6b7280; font-size: 14px; margin: 0 0 10px 0;">
                    Need help? We're here to support you on your AI journey.
                  </p>
                  <p style="color: #9ca3af; font-size: 12px; margin: 0;">
                    © 2024 SIXTHVAULT. All rights reserved.
                  </p>
                </div>
              </div>
            </body>
            </html>
        """
        
        text_content = f"""
Welcome to SIXTHVAULT, {first_name}!

Your account is now active and ready to use!

Get started: {vault_url}

Need help? We're here to support you on your AI journey.
        """
        
        return await EmailService._send_email(
            to=to,
            subject="Welcome to SIXTHVAULT! Your Account is Active",
            html_content=html_content,
            text_content=text_content
        )

    @staticmethod
    async def sendPasswordResetVerificationEmail(to: str, first_name: str, verification_code: str) -> Dict[str, Any]:
        """Send password reset verification code email"""
        base_url = EmailService._get_base_url()
        
        html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Reset Your Password</title>
            </head>
            <body style="margin: 0; padding: 0; font-family: Arial, sans-serif; background-color: #f5f5f5;">
              <div style="max-width: 600px; margin: 0 auto; background-color: white; padding: 40px 20px;">
                <!-- Header -->
                <div style="text-align: center; margin-bottom: 40px;">
                  <h1 style="color: #1f2937; font-size: 28px; margin: 0; font-weight: bold;">
                    SIXTHVAULT
                  </h1>
                  <p style="color: #6b7280; margin: 10px 0 0 0;">AI-Powered Document Intelligence</p>
                </div>

                <!-- Main Content -->
                <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 30px; border-radius: 12px; text-align: center; margin-bottom: 30px;">
                  <h2 style="color: white; font-size: 24px; margin: 0 0 20px 0;">
                    🔐 Password Reset Request
                  </h2>
                  <p style="color: #fecaca; margin: 0 0 30px 0; font-size: 16px;">
                    Hi {first_name}, we received a request to reset your password
                  </p>
                  
                  <!-- Verification Code -->
                  <div style="background-color: white; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <p style="color: #374151; margin: 0 0 10px 0; font-size: 14px; font-weight: bold;">
                      Your Verification Code:
                    </p>
                    <div style="font-size: 32px; font-weight: bold; color: #dc2626; letter-spacing: 8px; font-family: monospace;">
                      {verification_code}
                    </div>
                  </div>

                  <p style="color: #fecaca; margin: 20px 0 0 0; font-size: 14px;">
                    Enter this code on the password reset page to continue
                  </p>
                </div>

                <!-- Security Notice -->
                <div style="background-color: #fef3c7; padding: 20px; border-radius: 8px; margin-bottom: 30px; border-left: 4px solid #f59e0b;">
                  <h3 style="color: #92400e; margin: 0 0 15px 0; font-size: 18px;">🛡️ Security Notice</h3>
                  <ul style="color: #92400e; margin: 0; padding-left: 20px;">
                    <li style="margin-bottom: 8px;">This verification code expires in 15 minutes</li>
                    <li style="margin-bottom: 8px;">The code can only be used once</li>
                    <li style="margin-bottom: 8px;">If you didn't request this, please ignore this email</li>
                    <li>Your password will remain unchanged until you complete the reset process</li>
                  </ul>
                </div>

                <!-- Instructions -->
                <div style="background-color: #f9fafb; padding: 20px; border-radius: 8px; margin-bottom: 30px;">
                  <h4 style="color: #374151; margin: 0 0 10px 0; font-size: 16px;">How to reset your password:</h4>
                  <ol style="color: #6b7280; margin: 0; padding-left: 20px;">
                    <li style="margin-bottom: 8px;">Enter the 6-digit code above on the password reset page</li>
                    <li style="margin-bottom: 8px;">Create a new secure password</li>
                    <li>Sign in with your new password</li>
                  </ol>
                </div>

                <!-- Footer -->
                <div style="text-align: center; padding-top: 30px; border-top: 1px solid #e5e7eb;">
                  <p style="color: #9ca3af; font-size: 14px; margin: 0 0 10px 0;">
                    This password reset was requested on {datetime.utcnow().strftime('%Y-%m-%d at %H:%M:%S')} UTC
                  </p>
                  <p style="color: #9ca3af; font-size: 12px; margin: 0;">
                    If you didn't request this password reset, please contact our security team immediately.
                  </p>
                </div>
              </div>
            </body>
            </html>
        """
        
        text_content = f"""
SIXTHVAULT - Password Reset Request

Hi {first_name}, we received a request to reset your password.

Your verification code is: {verification_code}

SECURITY NOTICE:
- This verification code expires in 15 minutes
- The code can only be used once
- If you didn't request this, please ignore this email
- Your password will remain unchanged until you complete the reset process

How to reset your password:
1. Enter the 6-digit code above on the password reset page
2. Create a new secure password
3. Sign in with your new password

This password reset was requested on {datetime.utcnow().strftime('%Y-%m-%d at %H:%M:%S')} UTC

If you didn't request this password reset, please contact our security team immediately.
        """
        
        result = await EmailService._send_email(
            to=to,
            subject="Reset Your SIXTHVAULT Password - Verification Code",
            html_content=html_content,
            text_content=text_content
        )
        
        if result["success"]:
            result["verificationCode"] = verification_code
        
        return result
