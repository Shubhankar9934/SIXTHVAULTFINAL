import asyncio
import sys
import os
sys.path.append('.')
from lib.email_service import EmailService

async def test_verification_email():
    print('🧪 Testing Verification Email with Workaround')
    print('=' * 50)
    
    try:
        result = await EmailService.sendVerificationEmail(
            'test@company.com',
            'Test User', 
            'ABC123'
        )
        
        print(f'✅ Result: {result}')
        
        if result.get('redirected'):
            print(f'📧 Email redirected from: {result.get("original_recipient")}')
            print(f'📧 Email sent to: {result.get("actual_recipient")}')
            print('🎯 VERIFICATION EMAIL WORKAROUND WORKING!')
        elif result.get('simulated'):
            print('⚠️ Email was simulated')
        else:
            print('📧 Direct delivery successful')
            
    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == "__main__":
    asyncio.run(test_verification_email())
