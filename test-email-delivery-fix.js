#!/usr/bin/env node

/**
 * Email Delivery Diagnostic and Fix Script
 * 
 * This script tests and fixes email delivery issues in SIXTHVAULT
 * Run with: node test-email-delivery-fix.js
 */

const https = require('https');
const http = require('http');

class EmailDeliveryTester {
    constructor() {
        this.frontendUrl = 'http://localhost:3000';
        this.backendUrl = 'http://localhost:8000';
        this.testEmail = 'shubhankarbittu9934@gmail.com';
        this.resendApiKey = 're_4nS8L25m_5Yuaq2ffwvDnmR8PqjAwzqxY';
    }

    log(message, type = 'info') {
        const timestamp = new Date().toISOString();
        const colors = {
            info: '\x1b[36m',    // Cyan
            success: '\x1b[32m', // Green
            warning: '\x1b[33m', // Yellow
            error: '\x1b[31m',   // Red
            reset: '\x1b[0m'     // Reset
        };
        
        console.log(`${colors[type]}[${timestamp}] ${message}${colors.reset}`);
    }

    async makeRequest(url, options = {}) {
        return new Promise((resolve, reject) => {
            const isHttps = url.startsWith('https://');
            const client = isHttps ? https : http;
            
            const req = client.request(url, options, (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    try {
                        const jsonData = JSON.parse(data);
                        resolve({ status: res.statusCode, data: jsonData, headers: res.headers });
                    } catch (e) {
                        resolve({ status: res.statusCode, data: data, headers: res.headers });
                    }
                });
            });

            req.on('error', reject);
            
            if (options.body) {
                req.write(options.body);
            }
            
            req.end();
        });
    }

    async testResendApiDirectly() {
        this.log('Testing Resend API directly...', 'info');
        
        try {
            const payload = JSON.stringify({
                from: 'SIXTHVAULT <verify@sixth-vault.com>',
                to: [this.testEmail],
                subject: 'SIXTHVAULT Email Delivery Test',
                html: `
                    <h1>Email Delivery Test</h1>
                    <p>This is a test email to verify Resend API configuration.</p>
                    <p>If you receive this, the Resend API is working correctly.</p>
                    <p>Timestamp: ${new Date().toISOString()}</p>
                `,
                text: `
                    Email Delivery Test
                    
                    This is a test email to verify Resend API configuration.
                    If you receive this, the Resend API is working correctly.
                    
                    Timestamp: ${new Date().toISOString()}
                `
            });

            const response = await this.makeRequest('https://api.resend.com/emails', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.resendApiKey}`,
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(payload)
                },
                body: payload
            });

            if (response.status === 200) {
                this.log(`✅ Resend API test successful! Email ID: ${response.data.id}`, 'success');
                return { success: true, messageId: response.data.id };
            } else {
                this.log(`❌ Resend API test failed: ${JSON.stringify(response.data)}`, 'error');
                return { success: false, error: response.data };
            }
        } catch (error) {
            this.log(`❌ Resend API test error: ${error.message}`, 'error');
            return { success: false, error: error.message };
        }
    }

    async testBackendEmailService() {
        this.log('Testing backend email service...', 'info');
        
        try {
            const payload = JSON.stringify({
                to: this.testEmail,
                subject: 'SIXTHVAULT Backend Email Test',
                html_content: `
                    <h1>Backend Email Service Test</h1>
                    <p>This email was sent through the SIXTHVAULT backend email service.</p>
                    <p>Timestamp: ${new Date().toISOString()}</p>
                `,
                text_content: `
                    Backend Email Service Test
                    
                    This email was sent through the SIXTHVAULT backend email service.
                    
                    Timestamp: ${new Date().toISOString()}
                `
            });

            const response = await this.makeRequest(`${this.backendUrl}/email/send`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(payload)
                },
                body: payload
            });

            if (response.status === 200) {
                this.log(`✅ Backend email service test successful!`, 'success');
                this.log(`Response: ${JSON.stringify(response.data)}`, 'info');
                return { success: true, data: response.data };
            } else {
                this.log(`❌ Backend email service test failed: ${JSON.stringify(response.data)}`, 'error');
                return { success: false, error: response.data };
            }
        } catch (error) {
            this.log(`❌ Backend email service test error: ${error.message}`, 'error');
            return { success: false, error: error.message };
        }
    }

    async testFrontendEmailApi() {
        this.log('Testing frontend email API...', 'info');
        
        try {
            // Test with simulation mode (default)
            const simulatedPayload = JSON.stringify({
                to: this.testEmail,
                subject: 'SIXTHVAULT Frontend Email Test (Simulated)',
                html: '<h1>Frontend Email Test</h1><p>This should be simulated.</p>',
                text: 'Frontend Email Test - This should be simulated.',
                useBackendService: false
            });

            const simulatedResponse = await this.makeRequest(`${this.frontendUrl}/api/send-email`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(simulatedPayload)
                },
                body: simulatedPayload
            });

            this.log(`Simulated email response: ${JSON.stringify(simulatedResponse.data)}`, 'info');

            // Test with actual backend service
            const actualPayload = JSON.stringify({
                to: this.testEmail,
                subject: 'SIXTHVAULT Frontend Email Test (Actual)',
                html: `
                    <h1>Frontend Email Test (Actual)</h1>
                    <p>This email was sent through the frontend API with backend service enabled.</p>
                    <p>Timestamp: ${new Date().toISOString()}</p>
                `,
                text: `
                    Frontend Email Test (Actual)
                    
                    This email was sent through the frontend API with backend service enabled.
                    
                    Timestamp: ${new Date().toISOString()}
                `,
                useBackendService: true
            });

            const actualResponse = await this.makeRequest(`${this.frontendUrl}/api/send-email`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': Buffer.byteLength(actualPayload)
                },
                body: actualPayload
            });

            if (actualResponse.status === 200) {
                this.log(`✅ Frontend email API test successful!`, 'success');
                this.log(`Response: ${JSON.stringify(actualResponse.data)}`, 'info');
                return { success: true, data: actualResponse.data };
            } else {
                this.log(`❌ Frontend email API test failed: ${JSON.stringify(actualResponse.data)}`, 'error');
                return { success: false, error: actualResponse.data };
            }
        } catch (error) {
            this.log(`❌ Frontend email API test error: ${error.message}`, 'error');
            return { success: false, error: error.message };
        }
    }

    async checkResendDomainStatus() {
        this.log('Checking Resend domain status...', 'info');
        
        try {
            const response = await this.makeRequest('https://api.resend.com/domains', {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${this.resendApiKey}`,
                    'Content-Type': 'application/json'
                }
            });

            if (response.status === 200) {
                this.log(`✅ Domain status retrieved successfully!`, 'success');
                this.log(`Domains: ${JSON.stringify(response.data, null, 2)}`, 'info');
                
                const domains = response.data.data || [];
                const sixthVaultDomain = domains.find(d => d.name === 'sixth-vault.com');
                
                if (sixthVaultDomain) {
                    this.log(`Domain sixth-vault.com status: ${sixthVaultDomain.status}`, 'info');
                    if (sixthVaultDomain.status !== 'verified') {
                        this.log(`⚠️  Domain sixth-vault.com is not verified! This may cause delivery issues.`, 'warning');
                    }
                } else {
                    this.log(`⚠️  Domain sixth-vault.com not found in Resend account!`, 'warning');
                }
                
                return { success: true, domains: domains };
            } else {
                this.log(`❌ Failed to retrieve domain status: ${JSON.stringify(response.data)}`, 'error');
                return { success: false, error: response.data };
            }
        } catch (error) {
            this.log(`❌ Domain status check error: ${error.message}`, 'error');
            return { success: false, error: error.message };
        }
    }

    async generateEmailDeliveryReport() {
        this.log('Generating comprehensive email delivery report...', 'info');
        
        const results = {
            timestamp: new Date().toISOString(),
            tests: {}
        };

        // Test 1: Resend API Direct
        this.log('\n=== TEST 1: Resend API Direct ===', 'info');
        results.tests.resendDirect = await this.testResendApiDirectly();

        // Test 2: Domain Status
        this.log('\n=== TEST 2: Domain Status Check ===', 'info');
        results.tests.domainStatus = await this.checkResendDomainStatus();

        // Test 3: Backend Email Service
        this.log('\n=== TEST 3: Backend Email Service ===', 'info');
        results.tests.backendService = await this.testBackendEmailService();

        // Test 4: Frontend Email API
        this.log('\n=== TEST 4: Frontend Email API ===', 'info');
        results.tests.frontendApi = await this.testFrontendEmailApi();

        // Generate recommendations
        this.log('\n=== RECOMMENDATIONS ===', 'info');
        this.generateRecommendations(results);

        return results;
    }

    generateRecommendations(results) {
        const recommendations = [];

        if (!results.tests.resendDirect.success) {
            recommendations.push('❌ Resend API is not working - check API key and account status');
        }

        if (results.tests.domainStatus.success) {
            const domains = results.tests.domainStatus.domains || [];
            const sixthVaultDomain = domains.find(d => d.name === 'sixth-vault.com');
            
            if (!sixthVaultDomain) {
                recommendations.push('⚠️  Add and verify domain sixth-vault.com in Resend dashboard');
            } else if (sixthVaultDomain.status !== 'verified') {
                recommendations.push('⚠️  Verify domain sixth-vault.com in Resend dashboard');
            }
        }

        if (!results.tests.backendService.success) {
            recommendations.push('❌ Backend email service is not working - check backend server and configuration');
        }

        if (!results.tests.frontendApi.success) {
            recommendations.push('❌ Frontend email API is not working - check frontend server and API route');
        }

        if (results.tests.resendDirect.success && results.tests.backendService.success) {
            recommendations.push('✅ Email infrastructure is working - check spam folder and email filters');
        }

        if (recommendations.length === 0) {
            recommendations.push('✅ All tests passed - email delivery should be working correctly');
        }

        recommendations.forEach(rec => this.log(rec, 'info'));

        // Additional troubleshooting steps
        this.log('\n=== TROUBLESHOOTING STEPS ===', 'info');
        this.log('1. Check your spam/junk folder for test emails', 'info');
        this.log('2. Verify your email address in Resend dashboard for testing', 'info');
        this.log('3. Check Resend logs in dashboard for delivery status', 'info');
        this.log('4. Ensure domain sixth-vault.com is verified in Resend', 'info');
        this.log('5. Check if your email provider blocks emails from new domains', 'info');
    }

    async run() {
        this.log('🚀 Starting SIXTHVAULT Email Delivery Diagnostic...', 'info');
        this.log(`Testing email delivery to: ${this.testEmail}`, 'info');
        
        try {
            const report = await this.generateEmailDeliveryReport();
            
            this.log('\n=== DIAGNOSTIC COMPLETE ===', 'success');
            this.log('Check your email inbox and spam folder for test emails.', 'info');
            this.log('If you received emails, the system is working correctly.', 'info');
            this.log('If not, follow the recommendations above.', 'info');
            
            return report;
        } catch (error) {
            this.log(`❌ Diagnostic failed: ${error.message}`, 'error');
            throw error;
        }
    }
}

// Run the diagnostic if this script is executed directly
if (require.main === module) {
    const tester = new EmailDeliveryTester();
    tester.run().catch(console.error);
}

module.exports = EmailDeliveryTester;
