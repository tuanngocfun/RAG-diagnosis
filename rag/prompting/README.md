# RAG Prompting Module: Standalone MedGemma-4B-IT

This module provides a standalone implementation of MedGemma-4B-IT for multimodal medical image analysis **without requiring a RAG pipeline**. It directly processes medical images and text for diagnostic analysis using the same model infrastructure as the existing RAG system.

## Overview

The standalone system is designed for scenarios where you want to analyze medical images directly without the overhead of document retrieval. It's particularly useful when you have specific images or cases to analyze rather than needing to search through large document collections.

## Key Features

- **🚀 No RAG Pipeline**: Direct image and text processing
- **🩺 Multimodal Analysis**: Combines images and text naturally  
- **📋 Case-based Diagnosis**: Analyze complete case directories
- **⚡ Fast Response**: No retrieval overhead
- **🔄 Batch Processing**: Efficient processing of multiple cases
- **🐋 Docker Ready**: Works with existing container setup

## Files Overview

### Core Implementation
- **`medgemma4b_standalone.py`**: Main standalone MedGemma implementation
- **`run_batch_medgemma4b_standalone.py`**: Batch processing script
- **`__init__.py`**: Module exports and documentation

### Demo and Testing  
- **`medgemma4b_standalone_demo.ipynb`**: Interactive demonstration notebook
- **`test_standalone.py`**: Test script to verify functionality
- **`README.md`**: This documentation

## Quick Start

### 1. Basic Usage

```python
from rag.prompting.medgemma4b_standalone import StandaloneMedGemma4B

# Initialize analyzer
analyzer = StandaloneMedGemma4B()

# Analyze medical images
result = analyzer.analyze_images(
    image_paths=["/path/to/medical_image.png"], 
    question="What diagnostic features are visible?"
)
print(result)

# Diagnose a case by case ID
diagnosis = analyzer.diagnose_case(
    case_id="example_case_001",
    question="What is the most likely diagnosis?"
)
print(diagnosis)
```

### 2. Batch Processing

```bash
# Process questions from manifest file
python -m rag.prompting.run_batch_medgemma4b_standalone \
    --manifest kaggle/working2/rag_knowledge_base/questions_manifest.jsonl \
    --out kaggle/working2/rag_knowledge_base/answers/rtx4090/answers_medgemma4b_standalone.ndjson \
    --images_per_answer 3 \
    --resume
```

### 3. Docker Usage

```bash
cd /home/students/Leishmania && \
docker run --rm -it --gpus all \
    --user $(id -u):$(id -g) \
    --env-file /home/students/Leishmania/.env.docker \
    -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
    -e HF_HOME=/data4t/hf \
    -e TRANSFORMERS_CACHE=/data4t/hf/transformers \
    -v /home/students/Leishmania:/app \
    -v /data4t/hf:/data4t/hf \
    -w /app leish-gem25:latest \
    python -m rag.prompting.run_batch_medgemma4b_standalone \
        --manifest /app/kaggle/working2/rag_knowledge_base/questions_manifest.jsonl \
        --out kaggle/working2/rag_knowledge_base/answers/rtx4090/answers_medgemma4b_standalone.ndjson \
        --images_per_answer 3 \
        --resume
```

## Configuration

The system uses the same environment variables as the existing RAG system:

### Required Environment Variables
- `RAG_EXTRACT_ROOT`: Path to extracted case images (default: `/app/kaggle/working2/extract`)
- `TRANSFORMERS_CACHE`: Hugging Face model cache directory (default: `/data4t/hf/transformers`)

### Optional Environment Variables  
- `HF_HUB_OFFLINE=1`: Force offline mode for Hugging Face
- `TRANSFORMERS_OFFLINE=1`: Force offline mode for transformers

## API Reference

### StandaloneMedGemma4B Class

#### Methods

**`__init__(model_id=None)`**
- Initialize the standalone MedGemma analyzer
- `model_id`: Optional custom model ID (default: "google/medgemma-4b-it")

**`analyze_images(image_paths, question="", max_output_tokens=None)`**
- Analyze medical images with optional question
- `image_paths`: List of image file paths
- `question`: Optional medical question about the images
- `max_output_tokens`: Maximum tokens to generate
- Returns: Medical analysis text

**`answer_medical_question(question, image_paths=None, context="", max_output_tokens=None)`**
- Answer medical question with optional images and context
- `question`: Medical question to answer
- `image_paths`: Optional medical images for context
- `context`: Optional text context
- `max_output_tokens`: Maximum tokens to generate
- Returns: Medical answer

**`diagnose_case(case_id, question="", seed_images=None, max_output_tokens=None)`**
- Provide diagnostic analysis for a case
- `case_id`: Case identifier to locate in extract root
- `question`: Optional specific diagnostic question
- `seed_images`: Optional specific images to use
- `max_output_tokens`: Maximum tokens to generate
- Returns: Diagnostic analysis

### Utility Functions

**`create_medgemma_analyzer()`**
- Factory function to create analyzer instance
- Returns: StandaloneMedGemma4B instance

**`analyze_medical_images(analyzer, image_paths, question="")`**
- Helper function for image analysis
- Returns: Dictionary with analysis and metadata

**`batch_analyze_cases(analyzer, case_ids, questions=None)`**
- Batch process multiple cases
- Returns: List of analysis results

## Command Line Options

### run_batch_medgemma4b_standalone.py

```bash
python -m rag.prompting.run_batch_medgemma4b_standalone [OPTIONS]
```

**Required Arguments:**
- `--manifest`: Input questions manifest (JSONL format)
- `--out`: Output answers file (NDJSON format)

**Optional Arguments:**
- `--images_per_answer`: Maximum images per answer (default: 3)
- `--resume`: Resume from existing output file
- `--retry_errors`: Retry items that previously had errors
- `--fsync_interval`: Fsync after N processed items (default: 25)

## Input/Output Formats

### Manifest Input Format (JSONL)
```json
{
  "question_id": "q001",
  "case_id": "case_001", 
  "doc_id": "case_001",
  "question": "What is the most likely diagnosis?",
  "seed_image_paths": ["/path/to/image1.png", "/path/to/image2.png"]
}
```

### Output Format (NDJSON)
```json
{
  "question_id": "q001",
  "case_id": "case_001",
  "doc_id": "case_001", 
  "question": "What is the most likely diagnosis?",
  "answer": "Based on the clinical presentation...",
  "used_images": ["/path/to/image1.png", "/path/to/image2.png"],
  "model_type": "medgemma4b_standalone",
  "model_id": "google/medgemma-4b-it",
  "processing_mode": "standalone"
}
```

## Comparison: Standalone vs RAG

| Feature | Standalone MedGemma | RAG Pipeline |
|---------|-------------------|--------------|
| **Setup Complexity** | ✅ Simple | ❌ Complex (requires vector DB) |
| **Response Speed** | ✅ Fast (no retrieval) | ❌ Slower (retrieval + generation) |  
| **Dependencies** | ✅ Minimal | ❌ Many (ColQwen2, Qdrant, etc.) |
| **Image Analysis** | ✅ Direct multimodal | ❌ Via retrieval |
| **Document Search** | ❌ Not available | ✅ Full-text search |
| **Context Breadth** | ❌ Limited to input | ✅ Large knowledge base |
| **Deployment** | ✅ Lightweight | ❌ Heavy infrastructure |

## Use Cases

### When to Use Standalone MedGemma:
- ✅ **Direct Image Analysis**: You have specific images to analyze
- ✅ **Case-based Diagnosis**: Analyzing complete case directories  
- ✅ **Real-time Analysis**: Need fast response without retrieval
- ✅ **Lightweight Deployment**: Limited infrastructure
- ✅ **Focused Analysis**: Don't need document search capabilities

### When to Use RAG Pipeline:
- ✅ **Document Search**: Need to search across large collections
- ✅ **Evidence Retrieval**: Want supporting evidence from documents
- ✅ **Broad Context**: Need access to extensive knowledge base
- ✅ **Research Queries**: Exploratory questions across documents

## Testing

Run the test script to verify installation:

```bash
cd /home/students/Leishmania
python rag/prompting/test_standalone.py
```

This will check:
- ✅ Import functionality  
- ✅ Configuration setup
- ✅ Utility functions
- ✅ Environment compatibility
- ⚠️ Model loading (may require actual model files)

## Troubleshooting

### Common Issues

**1. Model Loading Errors**
```
FileNotFoundError: Model not found
```
**Solution**: Ensure MedGemma-4B-IT model is downloaded to the cache directory:
```bash
# Check model cache location
echo $TRANSFORMERS_CACHE
ls -la /data4t/hf/transformers/models--google--medgemma-4b-it/
```

**2. Case Directory Not Found**  
```
Case directory not found for case_id: xxx
```
**Solution**: Verify the extract root path and case structure:
```bash
# Check extract directory
ls -la $RAG_EXTRACT_ROOT/
# Should contain case directories with pages/ subdirectories
```

**3. Image Loading Errors**
```
Failed to load image: xxx
```
**Solution**: Check image file permissions and format:
```bash
# Verify image files exist and are readable
ls -la /path/to/image.png
file /path/to/image.png  # Should show image format
```

**4. GPU Memory Issues**
```
CUDA out of memory
```
**Solution**: The system automatically handles memory optimization, but you can:
- Reduce `images_per_answer` parameter
- Use CPU mode by setting `CUDA_VISIBLE_DEVICES=""`
- Restart to clear GPU memory

## Performance Tips

### Memory Optimization
- **Image Resizing**: Images are automatically resized to 1024px max dimension
- **Batch Size**: Process images in small batches to manage memory
- **GPU Cleanup**: Automatic memory cleanup between operations

### Speed Optimization  
- **Model Caching**: Models are loaded once and reused
- **Image Preprocessing**: Efficient PIL image handling
- **Text Generation**: Optimized generation parameters for medical content

## Integration with Existing System

The standalone system is designed to complement, not replace, the existing RAG system:

### Shared Components
- ✅ Same model cache directory (`/data4t/hf/`)
- ✅ Same case directory structure (`RAG_EXTRACT_ROOT`)
- ✅ Compatible with existing Docker setup
- ✅ Same MedGemma-4B-IT model

### Independent Components
- ✅ No vector database dependency
- ✅ No ColQwen2 retrieval requirement
- ✅ Separate processing pipeline
- ✅ Direct image analysis workflow

## Future Enhancements

Potential improvements for the standalone system:

1. **📊 Evaluation Metrics**: Integration with medical evaluation frameworks
2. **🔧 Fine-tuning Support**: Custom model fine-tuning capabilities  
3. **📱 Web Interface**: Simple web UI for interactive analysis
4. **🔄 Streaming Output**: Real-time response streaming
5. **📋 Report Generation**: Structured medical report output
6. **🔍 OCR Integration**: Automatic text extraction from images

## Contributing

When contributing to the standalone system:

1. **Follow Existing Patterns**: Use the same code style as RAG components
2. **Maintain Compatibility**: Ensure compatibility with Docker setup
3. **Test Thoroughly**: Use the test script to verify changes
4. **Document Changes**: Update README and docstrings
5. **Performance**: Consider memory and GPU usage impacts

## Support

For issues with the standalone system:

1. **Check Test Results**: Run `test_standalone.py` first
2. **Verify Environment**: Ensure all environment variables are set
3. **Model Files**: Confirm MedGemma-4B-IT model is downloaded
4. **Case Directory**: Verify extract directory structure
5. **GPU Setup**: Check CUDA availability and memory

The standalone system provides a streamlined path for direct medical image analysis while maintaining compatibility with the existing Leishmania RAG infrastructure.