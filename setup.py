#!/usr/bin/env python3
"""
Setup script to prepare the FastAPI application for Vercel deployment
"""

import os
import subprocess
import sys
from pathlib import Path

def check_requirements():
    """Check if required files exist"""
    required_files = ['Book1.csv', 'app.py', 'requirements.txt']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def train_model():
    """Run the training script to generate pickle files"""
    print("🤖 Training ML model and generating pickle files...")
    try:
        subprocess.check_call([sys.executable, 'train_model.py'])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False

def verify_pickle_files():
    """Verify that pickle files were created"""
    pickle_files = ['emotion_predictor.pkl', 'label_encoder.pkl']
    missing_files = []
    
    for file in pickle_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing pickle files: {', '.join(missing_files)}")
        return False
    
    print("✅ All pickle files created successfully")
    return True

def create_env_template():
    """Create environment template file"""
    env_content = """# Environment Variables for Production
OPENAI_API_KEY=your_openai_api_key_here
ENVIRONMENT=production
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env.example file")
    print("💡 Remember to set your OPENAI_API_KEY in Vercel environment variables")

def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/

# Environment Variables
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Vercel
.vercel

# Logs
*.log

# Temporary files
*.tmp
*.temp
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore file")

def display_deployment_instructions():
    """Display deployment instructions"""
    instructions = """
🚀 DEPLOYMENT INSTRUCTIONS FOR VERCEL
=====================================

1. Install Vercel CLI (if not already installed):
   npm install -g vercel

2. Login to your Vercel account:
   vercel login

3. Deploy your application:
   vercel --prod

4. Set environment variables in Vercel dashboard:
   - Go to your project settings
   - Add OPENAI_API_KEY with your actual OpenAI API key

5. Your API will be available at:
   https://your-project-name.vercel.app

📋 API ENDPOINTS:
================
- POST /api/place_order - Submit a trade
- GET /api/market_data/{symbol} - Get market data
- POST /mood/ - Submit mood data
- GET /dashboard/{user_id}/emotional_trends - Get emotional trends
- GET /users/{user_id} - Get user info
- POST /journal/ - Submit journal entry
- GET /recommendations/{user_id} - Get recommendations
- WebSocket /ws - Real-time updates

🔧 Testing locally:
==================
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload

📱 Frontend Integration:
=======================
Your React/Next.js frontend can connect to:
- Production: https://your-project-name.vercel.app
- Local: http://localhost:8000

🎯 Important Notes:
==================
- All pickle files are included in the deployment
- CORS is configured for all origins (adjust for production)
- Rate limiting is applied to prevent abuse
- WebSocket support is available for real-time features
"""
    
    print(instructions)

def main():
    """Main setup function"""
    print("🚀 Setting up FastAPI Emotion Trading AI for Vercel deployment")
    print("=" * 60)
    
    # Step 1: Check requirements
    if not check_requirements():
        return
    
    # Step 2: Install dependencies
    # if not install_dependencies():
    #     return
    
    # Step 3: Train model and create pickle files
    if not train_model():
        print("⚠️  Training failed, but continuing with deployment setup...")
    
    # Step 4: Verify pickle files
    if not verify_pickle_files():
        print("⚠️  Some pickle files are missing, the app might not work correctly")
    
    # Step 5: Create additional files
    create_env_template()
    create_gitignore()
    
    # Step 6: Display deployment instructions
    display_deployment_instructions()
    
    print("=" * 60)
    print("✅ Setup completed! Your app is ready for Vercel deployment.")

if __name__ == "__main__":
    main()
