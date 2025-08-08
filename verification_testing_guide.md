# Verification Fix Testing Guide

## 🎯 What Was Fixed

### Issues Identified:
1. **Input Sanitization**: Verification codes weren't properly sanitized
2. **Character Handling**: Spaces, dashes, and special characters caused mismatches
3. **Case Sensitivity**: Inconsistent uppercase/lowercase handling
4. **Debug Visibility**: No logging to identify verification failures

### Solutions Implemented:
1. **Input Sanitization**: Remove all non-alphanumeric characters
2. **Format Validation**: Ensure exactly 6 characters
3. **Consistent Casing**: Convert both input and stored codes to uppercase
4. **Debug Logging**: Detailed logs for verification attempts
5. **Better Error Messages**: More helpful user feedback

## 🚀 Deployment Steps

### 1. Run the Deployment Script
```bash
# From your local machine (Windows)
./deploy_verification_fix.sh
```

### 2. Alternative Manual Deployment
If the script doesn't work, deploy manually:

```bash
# SSH into your EC2 instance
ssh -i "C:/Users/Shubhankar/Downloads/sixthvault-key.pem" ubuntu@52.66.243.156

# Navigate to project directory
cd /home/ubuntu/SIXTHVAULTFINAL

# Pull latest changes
git pull origin main

# Navigate to backend directory
cd Rag_Backend

# Activate virtual environment
source venv/bin/activate

# Restart backend
pm2 restart sixthvault-backend

# Check status
pm2 status
```

## 🧪 Testing the Fix

### Step 1: Create a New Account
1. Go to `https://sixth-vault.com/register`
2. Fill in the registration form
3. Submit the registration

### Step 2: Monitor Backend Logs
```bash
# SSH into EC2
ssh -i "C:/Users/Shubhankar/Downloads/sixthvault-key.pem" ubuntu@52.66.243.156

# Watch backend logs in real-time
pm2 logs sixthvault-backend --lines 50
```

### Step 3: Check Email/Console for Verification Code
- Check your email for the verification code
- Or check the backend logs for the code (in development mode)

### Step 4: Test Verification
1. Go to the verification page
2. Enter the verification code exactly as received
3. Submit the verification

### Step 5: Monitor Debug Logs
Look for these log messages:
```
Verification attempt for email: user@example.com
Original input: 'ABC123'
Sanitized input: 'ABC123'
Stored code: 'ABC123'
Codes match: True
Verification successful for email: user@example.com
```

## 🔍 Debugging Common Issues

### Issue 1: Code Format Errors
**Symptoms**: "Verification code must be exactly 6 characters"
**Debug**: Check the log for "Invalid code format"
**Solution**: Ensure the code is exactly 6 alphanumeric characters

### Issue 2: Code Mismatch
**Symptoms**: "Invalid verification code"
**Debug**: Compare "Original input" vs "Sanitized input" vs "Stored code" in logs
**Solution**: Check for hidden characters or copy-paste issues

### Issue 3: Expired Code
**Symptoms**: "Verification code has expired"
**Debug**: Check the expiration time in logs
**Solution**: Request a new verification code

### Issue 4: Registration Not Found
**Symptoms**: "Registration not found"
**Debug**: Check if temp user exists in database
**Solution**: Register again

## 📊 Monitoring Commands

### Check Backend Status
```bash
pm2 status
```

### View Recent Logs
```bash
pm2 logs sixthvault-backend --lines 50
```

### Follow Logs in Real-time
```bash
pm2 logs sixthvault-backend --follow
```

### Check Backend Health
```bash
curl http://localhost:8000/health
```

### Test API Endpoints
```bash
# Test registration endpoint
curl -X POST https://sixth-vault.com/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"TestPass123","first_name":"Test","last_name":"User","company":"Test Co"}'

# Test verification endpoint
curl -X POST https://sixth-vault.com/api/auth/verify \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","verification_code":"ABC123"}'
```

## 🎯 Expected Behavior After Fix

### Successful Registration Flow:
1. **Registration**: User submits registration form
2. **Email Sent**: Verification code sent to email
3. **Code Generation**: 6-character alphanumeric code generated
4. **Storage**: Code stored in database with expiration
5. **Verification**: User enters code (with any formatting)
6. **Sanitization**: Code is cleaned and validated
7. **Comparison**: Sanitized input matches stored code
8. **Success**: User account is created and verified

### Debug Log Example:
```
Verification attempt for email: user@example.com
Original input: ' A B C 1 2 3 '
Sanitized input: 'ABC123'
Stored code: 'ABC123'
Codes match: True
Verification successful for email: user@example.com
```

## 🚨 Troubleshooting

### If Verification Still Fails:

1. **Check Email Service**:
   ```bash
   # Check if email service is working
   grep -i "email" /home/ubuntu/SIXTHVAULTFINAL/Rag_Backend/.env
   ```

2. **Check Database Connection**:
   ```bash
   # Test database connection
   cd /home/ubuntu/SIXTHVAULTFINAL/Rag_Backend
   source venv/bin/activate
   python -c "from app.database import engine; print('DB OK')"
   ```

3. **Check Code Generation**:
   ```bash
   # Look for code generation in logs
   pm2 logs sixthvault-backend | grep -i "verification"
   ```

4. **Manual Database Check**:
   ```bash
   # Connect to database and check temp users
   # (This requires database access credentials)
   ```

## 📝 Code Changes Summary

### Files Modified:
- `Rag_Backend/app/auth/routes.py`

### Key Changes:
1. **Input Sanitization Function**:
   ```python
   input_code = ''.join(c for c in input_code if c.isalnum())
   ```

2. **Format Validation**:
   ```python
   if not input_code or len(input_code) != 6:
       raise HTTPException(...)
   ```

3. **Debug Logging**:
   ```python
   print(f"Original input: '{request.verification_code}'")
   print(f"Sanitized input: '{input_code}'")
   print(f"Stored code: '{stored_code}'")
   ```

4. **Consistent Comparison**:
   ```python
   stored_code = temp_user.verification_code.strip().upper()
   if stored_code != input_code:
   ```

## 🎉 Success Indicators

✅ **Registration works**: New accounts can be created
✅ **Email delivery**: Verification codes are sent
✅ **Code validation**: Codes are properly sanitized and validated
✅ **Verification success**: Users can verify their accounts
✅ **Debug visibility**: Clear logs show verification process
✅ **Error handling**: Helpful error messages for users

## 📞 Support

If issues persist after deployment:
1. Check the debug logs for specific error messages
2. Verify all environment variables are set correctly
3. Ensure the email service is configured properly
4. Test with different email addresses
5. Check database connectivity and table structure

The fix addresses the core verification issues and provides comprehensive debugging capabilities to identify any remaining problems.
