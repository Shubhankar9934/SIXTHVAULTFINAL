/**
 * SIXTHVAULT Email Workaround Test
 * 
 * This script tests the new email workaround that redirects emails to the verified address
 * while preserving the original recipient information in the email content.
 */

const API_BASE_URL = 'http://localhost:8000';

async function testEmailWorkaround() {
    console.log('🧪 TESTING EMAIL WORKAROUND');
    console.log('=' .repeat(50));
    
    // Test cases with different email domains
    const testCases = [
        {
            name: 'Company Email Test',
            email: 'shubhankar.kumar@sapienplus.ai',
            subject: 'SIXTHVAULT Analysis: Test Company Email',
            content: 'This is a test email to verify the workaround for company domains.'
        },
        {
            name: 'Gmail Test',
            email: 'test@gmail.com',
            subject: 'SIXTHVAULT Analysis: Test Gmail',
            content: 'This is a test email to verify the workaround for Gmail domains.'
        },
        {
            name: 'Outlook Test',
            email: 'test@outlook.com',
            subject: 'SIXTHVAULT Analysis: Test Outlook',
            content: 'This is a test email to verify the workaround for Outlook domains.'
        }
    ];
    
    for (const testCase of testCases) {
        console.log(`\n📧 Testing: ${testCase.name}`);
        console.log(`   Target: ${testCase.email}`);
        
        try {
            const response = await fetch(`${API_BASE_URL}/email/send`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    to: testCase.email,
                    subject: testCase.subject,
                    html_content: `
                        <h2>SIXTHVAULT Email Test</h2>
                        <p>${testCase.content}</p>
                        <p><strong>Original Target:</strong> ${testCase.email}</p>
                        <p><strong>Test Time:</strong> ${new Date().toISOString()}</p>
                    `,
                    text_content: `
SIXTHVAULT Email Test

${testCase.content}

Original Target: ${testCase.email}
Test Time: ${new Date().toISOString()}
                    `
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                console.log(`   ✅ SUCCESS: ${result.messageId}`);
                if (result.redirected) {
                    console.log(`   📧 Redirected from: ${result.original_recipient}`);
                    console.log(`   📧 Sent to: ${result.actual_recipient}`);
                    console.log(`   🎯 WORKAROUND ACTIVE: Email delivered to verified address`);
                } else {
                    console.log(`   📧 Direct delivery successful`);
                }
                if (result.simulated) {
                    console.log(`   ⚠️  Simulated: ${result.message}`);
                }
            } else {
                console.log(`   ❌ FAILED: ${result.message || 'Unknown error'}`);
            }
            
        } catch (error) {
            console.log(`   ❌ ERROR: ${error.message}`);
        }
        
        // Wait between tests
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    console.log('\n' + '=' .repeat(50));
    console.log('🎯 EXPECTED BEHAVIOR:');
    console.log('1. All emails should return success=true');
    console.log('2. Emails should be redirected to sapien.cloud1@gmail.com');
    console.log('3. Original recipient info should be preserved in email content');
    console.log('4. Subject should be prefixed with [FOR: original@email.com]');
    console.log('5. Check sapien.cloud1@gmail.com inbox for actual emails');
    console.log('=' .repeat(50));
}

// Run the test
testEmailWorkaround().catch(console.error);
