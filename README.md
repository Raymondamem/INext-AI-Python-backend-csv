INext AI backend using python fastAPI and machine learning CSV file for models
# Run all tests
    test_market_data()
    test_mood_submission()
    test_trade_placement()
    test_emotional_trends()
    test_recommendations()
    test_journal()
    test_performance()


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
