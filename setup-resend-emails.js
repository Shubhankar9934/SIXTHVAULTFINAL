#!/usr/bin/env node

/**
 * Resend Email Address Setup Script
 * 
 * This script helps you add and verify email addresses for your Resend domain
 * Run with: node setup-resend-emails.js
 */

const https = require('https');

class ResendEmailSetup {
    constructor() {
        this.resendApiKey = 're_4nS8L25m_5Yuaq2ffwvDnmR8PqjAwzqxY';
        this.domain = 'sixth-vault.com';
        this.emailsToAdd = [
            'noreply@sixth-vault.com',
            'send@sixth-vault.com',
            'support@sixth-vault.com',
            'hello@sixth-vault.com'
        ];
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
            const req = https.request(url, options, (res) => {
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

    async checkDomainStatus() {
        this.log('Checking current domain status...', 'info');
        
        try {
            const response = await this.makeRequest('https://api.resend.com/domains', {
                method: 'GET',
                headers: {
                    'Authorization': `Bearer ${this.resendApiKey}`,
                    'Content-Type': 'application/json'
                }
            });

            if (response.status === 200) {
                const domains = response.data.data || [];
                const sixthVaultDomain = domains.find(d => d.name === this.domain);
                
                if (sixthVaultDomain) {
                    this.log(`✅ Domain ${this.domain} found with status: ${sixthVaultDomain.status}`, 'success');
                    this.log(`Domain ID: ${sixthVaultDomain.id}`, 'info');
                    return { success: true, domain: sixthVaultDomain };
                } else {
                    this.log(`❌ Domain ${this.domain} not found in your Resend account`, 'error');
                    return { success: false, error: 'Domain not found' };
                }
            } else {
                this.log(`❌ Failed to check domain status: ${JSON.stringify(response.data)}`, 'error');
                return { success: false, error: response.data };
            }
        } catch (error) {
            this.log(`❌ Error checking domain status: ${error.message}`, 'error');
            return { success: false, error: error.message };
        }
    }

    async testEmailAddresses() {
        this.log('Testing email addresses...', 'info');
        
        const results = [];
        
        for (const emailAddress of this.emailsToAdd) {
            this.log(`Testing ${emailAddress}...`, 'info');
            
            try {
                const payload = JSON.stringify({
                    from: `SIXTHVAULT <${emailAddress}>`,
                    to: ['shubhankarbittu9934@gmail.com'],
                    subject: `Test from ${emailAddress}`,
                    html: `
                        <h1>Email Address Test</h1>
                        <p>This is a test email from <strong>${emailAddress}</strong></p>
                        <p>If you receive this, the email address is working correctly.</p>
                        <p>Timestamp: ${new Date().toISOString()}</p>
                    `,
                    text: `
                        Email Address Test
                        
                        This is a test email from ${emailAddress}
                        If you receive this, the email address is working correctly.
                        
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
                    this.log(`✅ ${emailAddress} - SUCCESS (ID: ${response.data.id})`, 'success');
                    results.push({ email: emailAddress, success: true, messageId: response.data.id });
                } else {
                    this.log(`❌ ${emailAddress} - FAILED: ${JSON.stringify(response.data)}`, 'error');
                    results.push({ email: emailAddress, success: false, error: response.data });
                }
            } catch (error) {
                this.log(`❌ ${emailAddress} - ERROR: ${error.message}`, 'error');
                results.push({ email: emailAddress, success: false, error: error.message });
            }

            // Wait 1 second between requests to avoid rate limiting
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        return results;
    }

    generateInstructions(testResults) {
        this.log('\n=== SETUP INSTRUCTIONS ===', 'info');
        
        const workingEmails = testResults.filter(r => r.success);
        const failingEmails = testResults.filter(r => !r.success);

        if (workingEmails.length > 0) {
            this.log('✅ Working email addresses:', 'success');
            workingEmails.forEach(result => {
                this.log(`   - ${result.email}`, 'success');
            });
        }

        if (failingEmails.length > 0) {
            this.log('❌ Email addresses that need to be added to Resend:', 'warning');
            failingEmails.forEach(result => {
                this.log(`   - ${result.email}`, 'warning');
            });

            this.log('\n📋 To fix the failing email addresses:', 'info');
            this.log('1. Go to https://resend.com/domains', 'info');
            this.log(`2. Click on your domain: ${this.domain}`, 'info');
            this.log('3. Go to the "Email addresses" or "Senders" section', 'info');
            this.log('4. Add the following email addresses:', 'info');
            
            failingEmails.forEach(result => {
                this.log(`   - ${result.email}`, 'info');
            });

            this.log('5. Wait for verification (usually instant for verified domains)', 'info');
            this.log('6. Run this script again to test', 'info');
        }

        if (workingEmails.length === testResults.length) {
            this.log('\n🎉 All email addresses are working correctly!', 'success');
            this.log('Your SIXTHVAULT email system is fully configured.', 'success');
        }
    }

    async run() {
        this.log('🚀 Starting Resend Email Address Setup...', 'info');
        
        try {
            // Check domain status
            const domainCheck = await this.checkDomainStatus();
            if (!domainCheck.success) {
                this.log('❌ Cannot proceed without a verified domain', 'error');
                return;
            }

            // Test email addresses
            this.log('\n=== TESTING EMAIL ADDRESSES ===', 'info');
            const testResults = await this.testEmailAddresses();

            // Generate instructions
            this.generateInstructions(testResults);

            this.log('\n=== SETUP COMPLETE ===', 'success');
            this.log('Check your email inbox for test messages.', 'info');
            
            return testResults;
        } catch (error) {
            this.log(`❌ Setup failed: ${error.message}`, 'error');
            throw error;
        }
    }
}

// Run the setup if this script is executed directly
if (require.main === module) {
    const setup = new ResendEmailSetup();
    setup.run().catch(console.error);
}

module.exports = ResendEmailSetup;
