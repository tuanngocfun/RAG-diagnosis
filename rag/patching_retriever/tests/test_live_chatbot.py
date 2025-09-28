#!/usr/bin/env python3
"""
Test the actual Docker chatbot to see the real behavior
"""

import requests
import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chatbot_api():
    """Test the running chatbot API"""
    base_url = "http://localhost:8000"
    
    # Check if the API is running
    try:
        health_response = requests.get(f"{base_url}/healthz", timeout=5)
        if health_response.status_code == 200:
            logger.info("✅ Chatbot API is running")
        else:
            logger.error(f"❌ Health check failed: {health_response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Cannot connect to chatbot API: {e}")
        logger.info("💡 Make sure the Docker container is running on localhost:8000")
        return False
    
    # Test simple question
    test_questions = [
        "What is leishmaniasis?",
        "How do you treat cutaneous leishmaniasis?",
        "What are the symptoms of visceral leishmaniasis?"
    ]
    
    for question in test_questions:
        logger.info(f"\n🔍 Testing question: {question}")
        
        try:
            response = requests.post(
                f"{base_url}/api/ask",
                json={"question": question, "top_k": 3, "images_per_answer": 1},
                timeout=60  # Give it time for model inference
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "")
                hits = result.get("hits", [])
                
                logger.info(f"✅ Got response: {len(answer)} chars, {len(hits)} hits")
                logger.info(f"Answer preview: {answer[:200]}...")
                
                # Check for the same issues we saw in the logs
                if len(answer) > 2000:
                    logger.warning("⚠️ Very long answer - potential repetition issue")
                
                if answer.count("year-old") > 2:
                    logger.warning("⚠️ Multiple case descriptions in answer")
                
                if "user You are a medical expert" in answer:
                    logger.warning("⚠️ Prompt leakage detected")
                
                # Check for gibberish patterns
                if len(answer.split()) > 100:
                    word_count = len(answer.split())
                    medical_terms = sum(1 for word in answer.split() if any(term in word.lower() for term in [
                        'medical', 'clinical', 'diagnosis', 'treatment', 'therapeutic', 'pathological'
                    ]))
                    
                    if medical_terms > word_count * 0.15:  # >15% medical terms
                        logger.warning(f"⚠️ Very high medical term density: {medical_terms}/{word_count}")
                
            else:
                logger.error(f"❌ API request failed: {response.status_code}")
                logger.error(f"Response: {response.text[:500]}...")
                
        except requests.exceptions.Timeout:
            logger.error("❌ Request timed out")
        except Exception as e:
            logger.error(f"❌ Request failed: {e}")
    
    return True

def test_with_file_upload():
    """Test the chatbot with file upload to reproduce the exact issue from logs"""
    base_url = "http://localhost:8000"
    
    # Create a simple test image
    from PIL import Image
    test_img = Image.new('RGB', (300, 200), color='lightblue')
    test_path = "/tmp/test_medical.png"
    test_img.save(test_path)
    
    logger.info("🔍 Testing with file upload (reproducing the logged scenario)")
    
    try:
        # Upload file and ask question - this should reproduce the exact logged behavior
        with open(test_path, 'rb') as f:
            files = {'files': ('test_medical.png', f, 'image/png')}
            data = {
                'question': 'how can you cure this?',
                'top_k': 3,
                'images_per_answer': 2
            }
            
            response = requests.post(
                f"{base_url}/api/ask-with-files",
                files=files,
                data=data,
                timeout=120
            )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "")
            
            logger.info(f"✅ File upload response: {len(answer)} chars")
            logger.info(f"Answer preview: {answer[:300]}...")
            
            # Save the full response for analysis
            with open("/tmp/test_response.json", "w") as f:
                json.dump(result, f, indent=2)
            
            logger.info("📁 Full response saved to /tmp/test_response.json")
            
            # This should match the behavior we saw in the terminal logs
            if "user You are a medical expert" in answer:
                logger.warning("🎯 REPRODUCED: Prompt leakage issue")
            
            if len(answer) > 5000:
                logger.warning("🎯 REPRODUCED: Very long garbage answer")
            
        else:
            logger.error(f"❌ File upload failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            
    except Exception as e:
        logger.error(f"❌ File upload test failed: {e}")
    
    finally:
        # Clean up
        if Path(test_path).exists():
            Path(test_path).unlink()

def main():
    """Run chatbot tests"""
    logger.info("🚀 Testing running chatbot to identify issues...")
    
    # Test basic API functionality
    if test_chatbot_api():
        logger.info("\n" + "="*60)
        # Test file upload scenario
        test_with_file_upload()
    
    logger.info("\n" + "="*60)
    logger.info("🎯 Test Summary:")
    logger.info("- Check the logs above for specific issues identified")
    logger.info("- Compare with the terminal logs in .github/terminal-log.md")
    logger.info("- Full response saved to /tmp/test_response.json for analysis")

if __name__ == "__main__":
    main()