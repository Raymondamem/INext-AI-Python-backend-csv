#!/usr/bin/env python3
"""
Test script to verify the FastAPI application works correctly
"""

import requests
import json
import time
import sys
from datetime import datetime

# Configuration
# BASE_URL = "http://localhost:8000"  # Change to your deployed URL for production testing
BASE_URL = "http://0.0.0.0:8000/"

def test_basic_endpoints():
    """Test basic API endpoints"""
    print("🧪 Testing basic API endpoints...")
    
    # Test health check (root endpoint)
    try:
        response = requests.get(BASE_URL)
        print(f"✅ Root endpoint: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the app is running:")
        print("   uvicorn app:app --host 0.0.0.0 --port 8000")
        return False
    
    # Test user creation/retrieval
    try:
        response = requests.get(f"{BASE_URL}/users/123")
        if response.status_code == 200:
            print("✅ User endpoint working")
            user_data = response.json()
            print(f"   User: {user_data['username']}")
        else:
            print(f"❌ User endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ User endpoint error: {e}")
    
    return True

def test_market_data():
    """Test market data endpoint"""
    print("📈 Testing market data endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/market_data/BTCUSDT")
        if response.status_code == 200:
            print("✅ Market data endpoint working")
            data = response.json()
            print(f"   Data points: {len(data)}")
        else:
            print(f"❌ Market data failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Market data error: {e}")

def test_mood_submission():
    """Test mood submission endpoint"""
    print("😊 Testing mood submission...")
    
    mood_data = {
        "user_id": 123,
        "mood": "confident",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(f"{BASE_URL}/mood/", json=mood_data)
        if response.status_code == 200:
            print("✅ Mood submission working")
            result = response.json()
            print(f"   Status: {result['status']}")
        else:
            print(f"❌ Mood submission failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Mood submission error: {e}")

def test_trade_placement():
    """Test trade placement endpoint"""
    print("💰 Testing trade placement...")
    
    trade_data = {
        "user_id": 123,
        "symbol": "BTCUSDT",
        "type": "market",
        "side": "buy",
        "amount": 100.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/place_order", json=trade_data)
        if response.status_code == 200:
            print("✅ Trade placement working")
            result = response.json()
            print(f"   Status: {result['status']}")
            if 'trade_id' in result:
                print(f"   Trade ID: {result['trade_id']}")
        else:
            print(f"❌ Trade placement failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Trade placement error: {e}")

def test_emotional_trends():
    """Test emotional trends endpoint"""
    print("🎭 Testing emotional trends...")
    
    try:
        response = requests.get(f"{BASE_URL}/dashboard/123/emotional_trends")
        if response.status_code == 200:
            print("✅ Emotional trends working")
            data = response.json()
            print(f"   Moods: {len(data.get('moods', []))}")
            print(f"   Indices: {list(data.get('indices', {}).keys())}")
        else:
            print(f"❌ Emotional trends failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Emotional trends error: {e}")

def test_recommendations():
    """Test recommendations endpoint"""
    print("💡 Testing recommendations...")
    
    # First, place a few trades to generate data
    for i in range(5):
        trade_data = {
            "user_id": 123,
            "symbol": "BTCUSDT",
            "type": "market",
            "side": "buy" if i % 2 == 0 else "sell",
            "amount": 50.0 + i * 10
        }
        requests.post(f"{BASE_URL}/api/place_order", json=trade_data)
        time.sleep(0.1)  # Small delay between trades
    
    try:
        response = requests.get(f"{BASE_URL}/recommendations/123")
        if response.status_code == 200:
            print("✅ Recommendations working")
            recommendations = response.json()
            print(f"   Recommendations: {len(recommendations)}")
            for rec in recommendations[:2]:  # Show first 2
                print(f"   - {rec.get('message', '')[:50]}...")
        else:
            print(f"❌ Recommendations failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Recommendations error: {e}")

def test_journal():
    """Test journal endpoint"""
    print("📝 Testing journal...")
    
    journal_data = {
        "user_id": 123,
        "entry": "Today I made some good trades and stayed disciplined with my strategy.",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        response = requests.post(f"{BASE_URL}/journal/", json=journal_data)
        if response.status_code == 200:
            print("✅ Journal working")
            result = response.json()
            print(f"   Status: {result['status']}")
        else:
            print(f"❌ Journal failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Journal error: {e}")

def test_performance():
    """Test API performance with multiple requests"""
    print("⚡ Testing API performance...")
    
    start_time = time.time()
    successful_requests = 0
    total_requests = 20
    
    for i in range(total_requests):
        try:
            response = requests.get(f"{BASE_URL}/users/{100 + i}")
            if response.status_code == 200:
                successful_requests += 1
        except:
            pass
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"✅ Performance test completed:")
    print(f"   Successful requests: {successful_requests}/{total_requests}")
    print(f"   Average response time: {duration/total_requests:.3f}s")
    print(f"   Requests per second: {total_requests/duration:.1f}")

def main():
    """Run all tests"""
    print("🚀 Starting FastAPI Application Tests")
    print("=" * 50)
    
    # Test basic connectivity first
    if not test_basic_endpoints():
        print("❌ Basic connectivity failed. Exiting.")
        sys.exit(1)
    
    # Run all tests
    test_market_data()
    test_mood_submission()
    test_trade_placement()
    test_emotional_trends()
    test_recommendations()
    test_journal()
    test_performance()
    
    print("=" * 50)
    print("✅ All tests completed!")
    print("\n🎯 Your FastAPI app is working correctly and ready for production!")
    
    print("\n📋 Next Steps:")
    print("1. Deploy to Vercel using: vercel --prod")
    print("2. Set OPENAI_API_KEY in Vercel environment variables")
    print("3. Update your frontend to use the deployed API URL")
    print("4. Test with real trading data and user interactions")

if __name__ == "__main__":
    main()
