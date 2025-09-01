// Test script to verify email functionality
async function testEmailFunctionality() {
  console.log("🧪 Testing SIXTHVAULT Email Functionality")
  console.log("=" * 50)

  // Test 1: Simulated Email (default behavior)
  console.log("\n📧 Test 1: Simulated Email")
  try {
    const response1 = await fetch('http://localhost:3000/api/send-email', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        to: 'test@example.com',
        subject: 'Test Simulated Email',
        text: 'This is a test email in simulation mode.',
        html: '<p>This is a test email in simulation mode.</p>',
        useBackendService: false
      })
    })

    const result1 = await response1.json()
    console.log("✅ Simulated Email Result:", result1)
  } catch (error) {
    console.log("❌ Simulated Email Error:", error)
  }

  // Test 2: Actual Email via Backend Service
  console.log("\n📨 Test 2: Actual Email via Backend Service")
  try {
    const response2 = await fetch('http://localhost:3000/api/send-email', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        to: 'shubhankarbittu9934@gmail.com',
        subject: 'Test Actual Email from SIXTHVAULT',
        text: 'This is a test email sent via the Resend service.',
        html: '<p>This is a test email sent via the <strong>Resend service</strong>.</p>',
        useBackendService: true
      })
    })

    const result2 = await response2.json()
    console.log("✅ Actual Email Result:", result2)
  } catch (error) {
    console.log("❌ Actual Email Error:", error)
  }

  // Test 3: Direct Backend Email Service
  console.log("\n🔧 Test 3: Direct Backend Email Service")
  try {
    const response3 = await fetch('http://localhost:8000/email/send', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        to: 'shubhankarbittu9934@gmail.com',
        subject: 'Direct Backend Test Email',
        html_content: '<p>This is a direct test of the backend email service.</p>',
        text_content: 'This is a direct test of the backend email service.'
      })
    })

    const result3 = await response3.json()
    console.log("✅ Direct Backend Result:", result3)
  } catch (error) {
    console.log("❌ Direct Backend Error:", error)
  }

  console.log("\n🎉 Email functionality tests completed!")
}

// Run the test
testEmailFunctionality()
