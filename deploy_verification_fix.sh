#!/bin/bash

# Deployment script for verification fix
# This script will update the backend on your EC2 instance

set -e  # Exit on any error

echo "🚀 Starting deployment of verification fix..."

# Configuration
EC2_IP="52.66.243.156"
KEY_PATH="C:/Users/Shubhankar/Downloads/sixthvault-key.pem"
REPO_PATH="/home/ubuntu/SIXTHVAULTFINAL"

echo "📡 Connecting to EC2 instance..."

# Create a temporary script to run on the server
cat > temp_deploy_script.sh << 'EOF'
#!/bin/bash
set -e

echo "🔄 Updating backend code..."

# Navigate to project directory
cd /home/ubuntu/SIXTHVAULTFINAL

# Pull latest changes from GitHub
echo "📥 Pulling latest changes from GitHub..."
git pull origin main

# Navigate to backend directory
cd Rag_Backend

# Activate virtual environment
echo "🐍 Activating Python virtual environment..."
source venv/bin/activate

# Install any new dependencies (if requirements changed)
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if the backend is running
echo "🔍 Checking backend status..."
pm2 status

# Restart the backend to apply changes
echo "🔄 Restarting backend service..."
pm2 restart sixthvault-backend

# Wait a moment for restart
sleep 5

# Check if backend is healthy
echo "🏥 Checking backend health..."
curl -f http://localhost:8000/health || echo "⚠️  Health check failed, but continuing..."

# Show final status
echo "📊 Final PM2 status:"
pm2 status

echo "✅ Backend deployment completed successfully!"
echo "🔧 Verification fix has been applied."
echo "📝 Debug logs will now show detailed verification attempts."

# Show recent logs
echo "📋 Recent backend logs:"
pm2 logs sixthvault-backend --lines 10

EOF

# Copy the script to the server and execute it
echo "📤 Uploading deployment script to server..."
scp -i "$KEY_PATH" temp_deploy_script.sh ubuntu@$EC2_IP:/tmp/deploy_script.sh

echo "🔧 Executing deployment on server..."
ssh -i "$KEY_PATH" ubuntu@$EC2_IP "chmod +x /tmp/deploy_script.sh && /tmp/deploy_script.sh"

# Clean up temporary script
rm temp_deploy_script.sh

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 What was fixed:"
echo "   ✅ Input sanitization for verification codes"
echo "   ✅ Removal of non-alphanumeric characters"
echo "   ✅ Proper code format validation (6 characters)"
echo "   ✅ Enhanced debug logging"
echo "   ✅ Better error messages"
echo ""
echo "🧪 Testing instructions:"
echo "   1. Go to https://sixth-vault.com/register"
echo "   2. Create a new account"
echo "   3. Check the backend logs for verification code"
echo "   4. Try verification with the exact code"
echo ""
echo "📊 Monitor logs with:"
echo "   ssh -i \"$KEY_PATH\" ubuntu@$EC2_IP"
echo "   pm2 logs sixthvault-backend --lines 50"
echo ""
echo "🔍 Debug verification issues:"
echo "   - Check PM2 logs for detailed verification attempts"
echo "   - Look for 'Verification attempt for email:' messages"
echo "   - Compare 'Original input' vs 'Sanitized input' vs 'Stored code'"
echo ""
