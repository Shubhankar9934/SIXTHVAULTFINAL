import asyncio
import sys
import os
sys.path.append('.')
from lib.email_service import EmailService

async def test_verified_domain():
    print('🧪 Testing Email with Verified Domain (sixth-vault.com)')
    print('=' * 60)
    
    test_cases = [
        {
            'name': 'Company Email Test',
            'email': 'shubhankar.kumar@sapienplus.ai',
            'description': 'Testing direct delivery to company domain'
        },
        {
            'name': 'Gmail Test',
            'email': 'test@gmail.com',
            'description': 'Testing direct delivery to Gmail'
        },
        {
            'name': 'Outlook Test',
            'email': 'test@outlook.com',
            'description': 'Testing direct delivery to Outlook'
        }
    ]
    
    for test_case in test_cases:
        print(f'\n📧 {test_case["name"]}')
        print(f'   Target: {test_case["email"]}')
        print(f'   Description: {test_case["description"]}')
        
        try:
            result = await EmailService._send_email(
                to=test_case['email'],
                subject=f'SIXTHVAULT Test: {test_case["name"]}',
                html_content=f'''
                    <h2>SIXTHVAULT Email Test</h2>
                    <p>This is a test email to verify direct delivery with verified domain.</p>
                    <p><strong>Test:</strong> {test_case["name"]}</p>
                    <p><strong>Target:</strong> {test_case["email"]}</p>
                    <p><strong>Domain:</strong> sixth-vault.com (verified)</p>
                    <p><strong>Time:</strong> {asyncio.get_event_loop().time()}</p>
                ''',
                text_content=f'''
SIXTHVAULT Email Test

This is a test email to verify direct delivery with verified domain.

Test: {test_case["name"]}
Target: {test_case["email"]}
Domain: sixth-vault.com (verified)
Time: {asyncio.get_event_loop().time()}
                '''
            )
            
            print(f'   ✅ Result: {result}')
            
            if result.get('redirected'):
                print(f'   📧 Redirected from: {result.get("original_recipient")}')
                print(f'   📧 Sent to: {result.get("actual_recipient")}')
                print('   ⚠️  Still using workaround (domain config may need restart)')
            elif result.get('simulated'):
                print(f'   ⚠️  Simulated: {result.get("message")}')
            else:
                print('   🎯 DIRECT DELIVERY SUCCESSFUL!')
                print(f'   📧 Email sent directly to: {test_case["email"]}')
                
        except Exception as e:
            print(f'   ❌ Error: {e}')
        
        print('-' * 50)
    
    print('\n🎯 EXPECTED WITH VERIFIED DOMAIN:')
    print('1. Direct delivery to all email addresses')
    print('2. No redirection needed')
    print('3. No 403 errors')
    print('4. Emails delivered to actual recipients')
    print('=' * 60)

if __name__ == "__main__":
    asyncio.run(test_verified_domain())
