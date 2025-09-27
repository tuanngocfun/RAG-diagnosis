#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real Integration Tests for Medical RAG Chatbot
Tests actual functionality, not just imports
"""

import os
import sys
import json
import time
import requests
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Load environment variables first
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded environment from {env_file}")
    else:
        print(f"⚠️ Environment file not found at {env_file}")
except ImportError:
    print("⚠️ python-dotenv not available")

# Add the parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_rag_pipeline_direct():
    """Test the RAG pipeline directly without API"""
    print("🧪 Testing RAG pipeline directly...")
    
    try:
        from rag.retriever.medgemma4b_qdrant_crossencoder_medcpt import (
            qdrant_ask_text, answer_with_gemini
        )
        
        # Test a simple question
        test_question = "What is leishmaniasis?"
        
        print(f"📝 Testing question: {test_question}")
        
        # Test retrieval
        print("🔍 Testing document retrieval...")
        retrieval_result = qdrant_ask_text(
            question=test_question,
            top_k=3,
            use_reranker=True
        )
        
        hits = retrieval_result.get("hits", [])
        if not hits:
            print("❌ No documents retrieved - check Qdrant connection and data")
            return False
            
        print(f"✅ Retrieved {len(hits)} documents")
        
        # Validate hit structure
        first_hit = hits[0]
        required_fields = ['doc_id', 'score', 'page_index']
        missing_fields = [field for field in required_fields if field not in first_hit]
        
        if missing_fields:
            print(f"❌ Missing required fields in hits: {missing_fields}")
            return False
            
        print(f"✅ Hit structure valid. Top hit: {first_hit['doc_id']} (score: {first_hit.get('score', 0):.3f})")
        
        # Test generation
        print("🤖 Testing answer generation...")
        generation_result = answer_with_gemini(
            question=test_question,
            hits=hits,
            take=2
        )
        
        answer = generation_result.get("answer", "")
        if not answer or len(answer.strip()) < 10:
            print("❌ Generated answer is empty or too short")
            return False
            
        print(f"✅ Generated answer ({len(answer)} chars)")
        # Show first 500 chars to see actual medical content
        preview = answer[:500].replace('\n', ' ').strip()
        print(f"   Preview: {preview}...")
        
        # Test evidence structure
        evidence = generation_result.get("evidence", [])
        if evidence:
            print(f"✅ Generated {len(evidence)} evidence citations:")
            for i, (span, cite) in enumerate(evidence[:3], 1):
                print(f"   [{i}] {span[:100]}... -> {cite}")
            if len(evidence) > 3:
                print(f"   ... and {len(evidence) - 3} more citations")
        else:
            print("⚠️ No evidence citations generated")
            
        return True
        
    except Exception as e:
        print(f"❌ RAG pipeline test failed: {e}")
        return False

def test_backend_api():
    """Test the FastAPI backend"""
    print("🧪 Testing FastAPI backend...")
    
    # Start the backend server
    print("🚀 Starting backend server...")
    backend_process = None
    
    try:
        backend_process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=Path(__file__).parent / "server",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(os.environ)
        )
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        for attempt in range(30):  # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:8000/healthz", timeout=2)
                if response.status_code == 200:
                    print("✅ Backend server started successfully")
                    break
            except requests.exceptions.RequestException:
                time.sleep(1)
        else:
            print("❌ Backend server failed to start")
            return False
        
        # Test health endpoint
        print("🏥 Testing health endpoint...")
        health_response = requests.get("http://localhost:8000/healthz")
        
        if health_response.status_code != 200:
            print(f"❌ Health check failed with status {health_response.status_code}")
            return False
            
        health_data = health_response.json()
        print(f"✅ Health check passed: {health_data}")
        
        # Test ask endpoint
        print("💬 Testing ask endpoint...")
        ask_payload = {
            "question": "What are the symptoms of leishmaniasis?",
            "top_k": 5,
            "case_type": "cutaneous",
            "images_per_answer": 2
        }
        
        ask_response = requests.post(
            "http://localhost:8000/api/ask",
            json=ask_payload,
            timeout=60  # Give it time for model loading
        )
        
        if ask_response.status_code != 200:
            print(f"❌ Ask endpoint failed with status {ask_response.status_code}")
            print(f"Response: {ask_response.text}")
            return False
            
        ask_data = ask_response.json()
        
        # Validate response structure
        required_fields = ['answer', 'hits', 'evidence', 'used_images']
        missing_fields = [field for field in required_fields if field not in ask_data]
        
        if missing_fields:
            print(f"❌ Missing fields in API response: {missing_fields}")
            return False
            
        print(f"✅ API response valid")
        answer_text = ask_data.get('answer', '')
        print(f"   Answer length: {len(answer_text)}")
        if answer_text:
            # Show actual answer preview
            preview = answer_text[:300].replace('\n', ' ').strip()
            print(f"   Answer preview: {preview}...")
        
        hits = ask_data.get('hits', [])
        evidence = ask_data.get('evidence', [])
        print(f"   Hits count: {len(hits)}")
        print(f"   Evidence count: {len(evidence)}")
        
        # Show actual evidence
        if evidence:
            print("   Evidence samples:")
            for i, (span, cite) in enumerate(evidence[:2], 1):
                print(f"     [{i}] {span[:80]}... -> {cite}")
        
        # Show hit samples
        if hits:
            print("   Top hit samples:")
            for i, hit in enumerate(hits[:2], 1):
                print(f"     Hit {i}: {hit.get('doc_id', 'unknown')} (score: {hit.get('score', 0):.3f})")
        
        print(f"   Images count: {len(ask_data.get('used_images', []))}")
        
        # Test with invalid request (should return 422 for validation error)
        print("🔍 Testing error handling...")
        error_response = requests.post(
            "http://localhost:8000/api/ask",
            json={"question": "", "top_k": -1},  # Invalid: empty question and negative top_k
            timeout=30
        )
        
        if error_response.status_code == 422:
            print("✅ Error handling works - validation errors return 422 as expected")
        elif error_response.status_code == 200:
            error_data = error_response.json()
            if error_data.get('note'):
                print("✅ Error handling works - empty question handled gracefully with note")
            else:
                print("⚠️ Empty question should return error or note")
        else:
            print(f"⚠️ Unexpected error handling status: {error_response.status_code}")
            print(f"   Response: {error_response.text[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Backend API test failed: {e}")
        return False
        
    finally:
        # Clean up - terminate backend process
        if backend_process:
            print("🧹 Cleaning up backend process...")
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()

def test_cli_tool():
    """Test the CLI tool"""
    print("🧪 Testing CLI tool...")
    
    try:
        # Test help command
        print("📖 Testing CLI help...")
        help_result = subprocess.run(
            [sys.executable, "ask.py", "--help"],
            cwd=Path(__file__).parent / "tools",
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if help_result.returncode != 0:
            print("❌ CLI help command failed")
            return False
            
        print("✅ CLI help works")
        
        # Test actual question
        print("💬 Testing CLI question...")
        question_result = subprocess.run(
            [sys.executable, "ask.py", 
             "--question", "What is cutaneous leishmaniasis?",
             "--top-k", "3"],
            cwd=Path(__file__).parent / "tools",
            capture_output=True,
            text=True,
            timeout=120  # Give it time for model loading
        )
        
        if question_result.returncode != 0:
            print(f"❌ CLI question failed")
            print(f"STDOUT: {question_result.stdout}")
            print(f"STDERR: {question_result.stderr}")
            return False
            
        # Parse the JSON output
        try:
            result = json.loads(question_result.stdout)
            
            # Validate structure
            required_fields = ['question', 'answer', 'hits']
            missing_fields = [field for field in required_fields if field not in result]
            
            if missing_fields:
                print(f"❌ CLI output missing fields: {missing_fields}")
                return False
                
            print(f"✅ CLI question succeeded")
            print(f"   Question: {result['question']}")
            
            answer = result.get('answer', '')
            print(f"   Answer length: {len(answer)}")
            if answer:
                # Show actual answer preview
                preview = answer[:300].replace('\n', ' ').strip()
                print(f"   Answer preview: {preview}...")
            
            hits = result.get('hits', [])
            evidence = result.get('evidence', [])
            print(f"   Hits count: {len(hits)}")
            print(f"   Evidence count: {len(evidence)}")
            
            # Show evidence samples
            if evidence:
                print("   Evidence samples:")
                for i, (span, cite) in enumerate(evidence[:2], 1):
                    print(f"     [{i}] {span[:80]}... -> {cite}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"❌ CLI output is not valid JSON: {e}")
            print(f"STDOUT (first 500 chars): {question_result.stdout[:500]}")
            if question_result.stderr:
                print(f"STDERR: {question_result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ CLI tool test failed: {e}")
        return False

def test_environment_setup():
    """Test environment configuration"""
    print("🧪 Testing environment setup...")
    
    # Check required environment variables
    required_vars = ['QDRANT_URL', 'QDRANT_API_KEY', 'HF_TOKEN']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        return False
        
    print("✅ Required environment variables are set")
    
    # Test Qdrant connectivity
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        
        collections = client.get_collections()
        print(f"✅ Qdrant connection successful. Collections: {len(collections.collections)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Medical RAG Chatbot - Real Integration Tests")
    print("=" * 50)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("RAG Pipeline Direct", test_rag_pipeline_direct),
        ("Backend API", test_backend_api),
        ("CLI Tool", test_cli_tool),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            results[test_name] = result
            
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n🎯 Test Results Summary")
    print("=" * 30)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The chatbot is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())