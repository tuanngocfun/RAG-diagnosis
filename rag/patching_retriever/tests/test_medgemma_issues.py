#!/usr/bin/env python3
"""
Test script to specifically test the MedGemma answer generation issues
"""

import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment from the Docker env file
app_env = Path("/home/students/Leishmania/app/.env.docker")
if app_env.exists():
    load_dotenv(app_env)

# Set up environment 
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("RAG_ROOT", "/home/students/Leishmania")
os.environ.setdefault("RAG_EXTRACT_ROOT", "/home/students/Leishmania/kaggle/working2/extract")

# Add project root to path
project_root = Path("/home/students/Leishmania")
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_medgemma_generation():
    """Test MedGemma generation with a simple case to identify the garbage output issue"""
    try:
        logger.info("🔧 Testing MedGemma answer generation...")
        
        from rag.retriever.medgemma4b_qdrant_crossencoder_medcpt import MedGemma4B, CFG
        from PIL import Image
        
        # Initialize MedGemma generator
        generator = MedGemma4B(model_id=CFG.GEN_MODEL_ID)
        logger.info("✅ MedGemma generator loaded successfully")
        
        # Create a simple test image (medical-looking)
        test_image = Image.new('RGB', (400, 300), color='white')
        test_image_path = "/tmp/test_medical_image.png"
        test_image.save(test_image_path)
        
        # Test questions of different types
        test_cases = [
            {
                "question": "What is the treatment for leishmaniasis?",
                "context": "Leishmaniasis is a parasitic disease caused by Leishmania protozoa transmitted by sandfly bites. Treatment includes antimonial compounds, amphotericin B, or miltefosine depending on the type and severity.",
                "expected_issues": ["long_answer", "repetitive_text"]
            },
            {
                "question": "How can you cure this lesion?",
                "context": "A cutaneous lesion showing characteristics of leishmaniasis with amastigotes present in tissue samples.",
                "expected_issues": ["gibberish", "case_description_repetition"]
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n--- Test Case {i}: {test_case['question']} ---")
            
            try:
                # Generate answer with minimal context to isolate the issue
                answer = generator.answer(
                    question=test_case["question"],
                    image_paths=[test_image_path],
                    spans=[],  # No evidence spans to avoid regurgitation
                    context_text=test_case["context"],
                    max_output_tokens=512,  # Limit tokens to prevent runaway generation
                    images_per_answer=1
                )
                
                logger.info(f"Generated answer length: {len(answer)} chars")
                logger.info(f"Answer preview (first 300 chars): {answer[:300]}...")
                
                # Check for specific issues
                issues_found = []
                
                if len(answer) > 2000:
                    issues_found.append("VERY_LONG_ANSWER")
                    logger.warning(f"⚠️ Answer is very long ({len(answer)} chars)")
                
                if answer.count("year-old") > 2:
                    issues_found.append("CASE_REPETITION")
                    logger.warning(f"⚠️ Contains {answer.count('year-old')} case descriptions")
                
                # Check for gibberish (lots of medical terms without coherent structure)
                medical_term_density = sum(1 for term in [
                    "medical", "clinical", "diagnosis", "treatment", "patient", "lesion",
                    "microscopy", "histopathology", "therapeutic", "pharmacological"
                ] if term in answer.lower())
                
                if medical_term_density > len(answer.split()) * 0.1:  # >10% medical terms
                    issues_found.append("HIGH_MEDICAL_TERM_DENSITY")
                    logger.warning(f"⚠️ High medical term density: {medical_term_density} terms")
                
                # Check for repetitive patterns
                words = answer.split()
                if len(words) > 50:
                    word_freq = {}
                    for word in words:
                        word_freq[word] = word_freq.get(word, 0) + 1
                    
                    repeated_words = [w for w, c in word_freq.items() if c > 5 and len(w) > 3]
                    if repeated_words:
                        issues_found.append("REPETITIVE_WORDS")
                        logger.warning(f"⚠️ Highly repeated words: {repeated_words}")
                
                # Check if answer actually addresses the question
                question_words = set(test_case["question"].lower().split())
                answer_words = set(answer.lower().split())
                overlap = len(question_words & answer_words)
                
                if overlap < 2:  # Very little overlap with question
                    issues_found.append("OFF_TOPIC")
                    logger.warning("⚠️ Answer seems off-topic from question")
                
                logger.info(f"Issues identified: {issues_found}")
                
                # Test the normalization function
                from rag.retriever.run_batch_answers_medgemma4b_test_medcpt import _normalize_answer
                normalized = _normalize_answer(answer)
                
                if len(normalized) != len(answer):
                    logger.info(f"Normalization changed length: {len(answer)} -> {len(normalized)}")
                    logger.info(f"Normalized preview: {normalized[:200]}...")
                
            except Exception as e:
                logger.error(f"❌ Generation failed for test case {i}: {e}")
                import traceback
                traceback.print_exc()
        
        # Clean up
        if Path(test_image_path).exists():
            Path(test_image_path).unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MedGemma generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_image_processing():
    """Test the image token processing issue specifically"""
    try:
        logger.info("🔧 Testing image token processing in MedGemma...")
        
        from rag.retriever.medgemma4b_qdrant_crossencoder_medcpt import MedGemma4B, CFG
        from PIL import Image
        
        # Initialize MedGemma generator
        generator = MedGemma4B(model_id=CFG.GEN_MODEL_ID)
        
        # Create test images
        test_images = []
        for i in range(2):
            img = Image.new('RGB', (300, 300), color=('red', 'blue')[i])
            path = f"/tmp/test_img_{i}.png"
            img.save(path)
            test_images.append(path)
        
        # Test the specific scenario from the logs
        logger.info("Testing scenario: 2 images with simple question")
        
        try:
            # This should reproduce the "Image token format issue"
            answer = generator.answer(
                question="What do you see in these images?",
                image_paths=test_images,
                spans=[],
                context_text="Medical image analysis context.",
                max_output_tokens=256,
                images_per_answer=2
            )
            
            logger.info("✅ Multi-image generation succeeded")
            logger.info(f"Answer: {answer[:150]}...")
            
        except Exception as e:
            logger.error(f"❌ Multi-image generation failed: {e}")
            logger.info("This matches the 'Image token format issue' from the logs")
        
        # Test with single image
        try:
            answer = generator.answer(
                question="What do you see in this image?",
                image_paths=[test_images[0]],
                spans=[],
                context_text="Medical image analysis context.",
                max_output_tokens=256,
                images_per_answer=1
            )
            
            logger.info("✅ Single-image generation succeeded")
            logger.info(f"Answer: {answer[:150]}...")
            
        except Exception as e:
            logger.error(f"❌ Single-image generation failed: {e}")
        
        # Clean up
        for path in test_images:
            if Path(path).exists():
                Path(path).unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Image processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run focused tests on the specific issues"""
    logger.info("🚀 Starting focused MedGemma issue testing...")
    
    tests = [
        ("Image Processing", test_image_processing),
        ("Answer Generation Quality", test_medgemma_generation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name}: {status}")

if __name__ == "__main__":
    main()