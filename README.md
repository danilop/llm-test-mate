# LLM Test Mate 🤝

A friendly testing framework for LLM-generated content. Makes it easy to validate outputs from large language models using semantic similarity and LLM-based evaluation.

## 🚀 Features

- 📊 Semantic similarity testing using sentence transformers
- 🤖 LLM-based evaluation of content quality and correctness
- 🔧 Easy integration with pytest
- 📝 Comprehensive test reports
- 🎯 Sensible defaults with flexible overrides

## 🏃‍♂️ Quick Start

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# To run the examples
python examples.py

# To run the tests
pytest                      # Run all tests
pytest test_examples.py     # Run all tests in file
pytest test_examples.py -v  # Run with verbose output
pytest test_examples.py::test_semantic_similarity  # Run a specific test
```

The test examples (`test_examples.py`) include:
- Semantic similarity testing
- LLM-based evaluation with Claude
- Custom evaluation criteria with Llama
- Model comparison tests
- Parameterized threshold testing

Here's how to get you started using this tool:

```python
from llm_test_mate import LLMTestMate

# Initialize the test mate with your preferences
tester = LLMTestMate(
    llm_model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    similarity_threshold=0.8,
    temperature=0.7
)

# Example: Basic semantic similarity test
reference_text = "The quick brown fox jumps over the lazy dog."
generated_text = "A swift brown fox leaps above a sleepy canine."

# Simple similarity check using default settings
result = tester.semantic_similarity(
    generated_text, 
    reference_text
)
print(f"Similarity score: {result['similarity']:.2f}")
print(f"Passed threshold: {result['passed']}")

# LLM-based evaluation
eval_result = tester.llm_evaluate(
    generated_text,
    reference_text
)
print(f"Evaluation result: {eval_result}")
```

Sample output:
```
Similarity score: 0.89
Passed threshold: True

Evaluation result: {
    "passed": true,
    "similarity_score": 0.92,
    "analysis": {
        "semantic_match": "Both texts describe the same action of a fox jumping over a dog, using synonymous terms",
        "content_match": "All key elements are present: fox (brown), jumping action, and dog (lazy/sleepy)",
        "key_differences": [
            "Word choice: 'quick' vs 'swift'",
            "Phrase structure: 'jumps over' vs 'leaps above'"
        ]
    },
    "model_used": "anthropic.claude-3-5-sonnet-20240620-v1:0"
}
```

## 📖 Usage Examples

### 1. Basic Text Generation and Evaluation

```python
from litellm import completion
from llm_test_mate import LLMTestMate

def generate_text(prompt: str) -> str:
    """Helper function to generate text using LiteLLM"""
    response = completion(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# Initialize with custom settings
tester = LLMTestMate(
    similarity_threshold=0.8,
    temperature=0.7,
    max_tokens=500
)

# Generate and evaluate a Python description
reference_summary = "Python is a high-level programming language known for its simplicity and readability."
generated_summary = generate_text("Write a one-sentence description of Python programming language")

eval_result = tester.llm_evaluate(
    generated_summary,
    reference_summary
)
```

Sample output:
```
Generated summary: "Python is an intuitive, high-level programming language that emphasizes code readability and clean syntax."

Evaluation result: {
    "passed": true,
    "similarity_score": 0.88,
    "analysis": {
        "semantic_match": "Both descriptions emphasize Python's key characteristics of being high-level and readable",
        "content_match": "Core concepts of high-level language, simplicity/readability are present in both",
        "key_differences": [
            "Generated text adds mention of 'clean syntax'",
            "Generated text uses 'intuitive' instead of 'simple'"
        ]
    },
    "model_used": "anthropic.claude-3-5-sonnet-20240620-v1:0"
}
```

### 2. Custom Evaluation Criteria

```python
# Initialize with custom criteria
tester = LLMTestMate(
    evaluation_criteria="""
    Evaluate the marketing effectiveness of the generated text compared to the reference.
    Consider:
    1. Feature Coverage: Are all key features mentioned?
    2. Tone: Is it engaging and professional?
    3. Clarity: Is the message clear and concise?

    Return JSON with:
    {
        "passed": boolean,
        "effectiveness_score": float (0-1),
        "analysis": {
            "feature_coverage": string,
            "tone_analysis": string,
            "suggestions": list[string]
        }
    }
    """
)

product_description = "Our new smartphone features a 6.1-inch OLED display, 12MP camera, and all-day battery life."
generated_description = generate_text("Write a short description of a smartphone's key features")

eval_result = tester.llm_evaluate(
    generated_description,
    product_description
)
```

Sample output:
```
Generated description: "Experience stunning visuals on the brilliant 6.1-inch OLED screen, capture perfect moments with the advanced 12MP camera, and enjoy worry-free usage with long-lasting battery performance."

Evaluation result: {
    "passed": true,
    "effectiveness_score": 0.92,
    "analysis": {
        "feature_coverage": "All three key features (display, camera, battery) are mentioned with accurate specifications",
        "tone_analysis": "Engaging and professional tone with active language and benefit-focused messaging",
        "suggestions": [
            "Could add specific battery duration metrics",
            "Consider mentioning additional camera capabilities",
            "May add screen resolution details for more technical appeal"
        ]
    },
    "model_used": "meta.llama3-2-90b-instruct-v1:0"
}
```

### 3. Using with Pytest

```python
import pytest
from llm_test_mate import LLMTestMate

@pytest.fixture
def tester():
    return LLMTestMate(
        similarity_threshold=0.8,
        temperature=0.7
    )

def test_generated_content(tester):
    generated = generate_text("Explain what is Python")
    expected = "Python is a high-level programming language..."
    
    # Check semantic similarity
    sem_result = tester.semantic_similarity(
        generated,
        expected
    )
    
    # Evaluate with LLM
    llm_result = tester.llm_evaluate(
        generated,
        expected
    )
    
    assert sem_result["passed"], f"Similarity: {sem_result['similarity']}"
    assert llm_result["passed"], f"LLM evaluation: {llm_result['analysis']}"
```

## 🛠️ Advanced Usage

### Combined Testing Approach

```python
def test_comprehensive_check(embedding_model):
    generated = generate_text("Write a recipe")
    expected = """
    Recipe must include:
    - Ingredients list
    - Instructions
    - Cooking time
    """
    
    # Check similarity
    sem_result = tester.semantic_similarity(
        generated,
        expected
    )
    
    # Detailed evaluation
    llm_result = tester.llm_evaluate(
        generated,
        expected
    )
    
    assert sem_result["passed"], "Failed similarity check"
    assert llm_result["passed"], f"Failed requirements: {llm_result['reasoning']}"
```

## 📊 Comprehensive Test Results

When running tests with LLM Test Mate, you get comprehensive results from two types of evaluations:

### Semantic Similarity Results
```python
{
    "similarity": 0.85,        # Similarity score between 0-1
    "embedding_model": "all-MiniLM-L6-v2",  # Model used for embeddings
    "passed": True,           # Whether it passed the threshold
    "threshold": 0.8          # The threshold used for this test
}
```

### LLM Evaluation Results
```python
{
    "passed": True,           # Overall pass/fail assessment
    "similarity_score": 0.9,  # Semantic similarity assessment by LLM
    "analysis": {
        "semantic_match": "The texts convey very similar meanings...",
        "content_match": "Both texts cover the same key points...",
        "key_differences": [
            "Minor variation in word choice",
            "Slightly different emphasis on..."
        ]
    },
    "model_used": "anthropic.claude-3-5-sonnet-20240620-v1:0"  # Model used for evaluation
}
```

For custom evaluation criteria, the results will match your specified JSON structure. For example, with marketing evaluation:
```python
{
    "passed": True,
    "effectiveness_score": 0.85,
    "analysis": {
        "feature_coverage": "All key features mentioned...",
        "tone_analysis": "Professional and engaging...",
        "suggestions": [
            "Consider emphasizing battery life more",
            "Add specific camera capabilities"
        ]
    },
    "model_used": "meta.llama3-2-90b-instruct-v1:0"
}
```

### Benefits of Combined Testing
When using both approaches together, you get:
- Quantitative similarity metrics from embedding comparison
- Qualitative content evaluation from LLM analysis
- Model-specific insights (can compare different LLM evaluations)
- Clear pass/fail indicators for automated testing
- Detailed feedback for manual review

This comprehensive approach helps ensure both semantic closeness to reference content and qualitative correctness of the generated output.

## 🔧 Adding to Your Project

The simplest way to add LLM Test Mate to your project is to copy the `llm_test_mate.py` file:

1. Copy `llm_test_mate.py` to your project's test directory
2. Add the required dependencies to your dev requirements:

```txt
# dev-requirements.txt or requirements-dev.txt
litellm
sentence-transformer
pytest  # if using with pytest
boto3   # if using Amazon Bedrock
```

3. Install the dev dependencies:
```bash
pip install -r dev-requirements.txt
```

### Project Structure

Typical integration into an existing project:

```
your_project/
├── src/
│   └── your_code.py
├── tests/
│   ├── llm_test_mate.py    # Copy the file here
│   ├── your_test_file.py   # Your LLM tests
│   └── conftest.py         # Pytest fixtures
├── dev-requirements.txt    # Add dependencies here
└── pytest.ini              # Optional pytest configuration
```

Example `conftest.py`:
```python
import pytest
from llm_test_mate import LLMTestMate

@pytest.fixture
def llm_tester():
    return LLMTestMate(
        similarity_threshold=0.8,
        temperature=0.7
    )

@pytest.fixture
def strict_llm_tester():
    return LLMTestMate(
        similarity_threshold=0.9,
        temperature=0.5
    )
```

Example test file:
```python
def test_product_description(llm_tester):
    expected = "Our product helps you test LLM outputs effectively."
    generated = your_llm_function("Describe our product")
    
    result = llm_tester.semantic_similarity(generated, expected)
    assert result['passed'], f"Generated text not similar enough: {result['similarity']}"
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Support

- Documentation: [Link to docs]
- Issues: [GitHub Issues]
- Questions: [GitHub Discussions]

## 🙏 Acknowledgments

- Built with [LiteLLM](https://github.com/BerriAI/litellm)
- Uses [sentence-transformers](https://github.com/UKPLab/sentence-transformers)