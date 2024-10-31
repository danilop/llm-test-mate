import pytest
from litellm import completion
from llm_test_mate import LLMTestMate

ALTERNATIVE_LLM_MODEL = "bedrock/meta.llama3-1-70b-instruct-v1:0"

def generate_text(prompt: str, model: str = ALTERNATIVE_LLM_MODEL) -> str:
    """Helper function to generate text using LiteLLM"""
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

@pytest.fixture
def default_tester():
    return LLMTestMate(
        similarity_threshold=0.8,
        temperature=0.7
    )

@pytest.fixture
def llama_tester():
    return LLMTestMate(
        llm_model=ALTERNATIVE_LLM_MODEL,
        similarity_threshold=0.8,
        temperature=0.7
    )

def test_semantic_similarity(default_tester):
    """Test basic semantic similarity comparison"""
    reference_text = "The quick brown fox jumps over the lazy dog."
    test_text = "A swift brown fox leaps above a sleepy canine."
    
    result = default_tester.semantic_similarity(
        test_text, 
        reference_text
    )
    
    assert result['similarity'] > 0.8
    assert result['passed'] is True
    assert 'embedding_model' in result

def test_llm_evaluation_claude(default_tester):
    """Test LLM-based evaluation with Claude"""
    reference_summary = "Python is a high-level programming language known for its simplicity and readability."
    generated_summary = generate_text("Write a one-sentence description of Python programming language")
    
    eval_result = default_tester.llm_evaluate(
        generated_summary,
        reference_summary
    )
    
    assert eval_result['passed'] is True
    assert 'similarity_score' in eval_result
    assert 'analysis' in eval_result
    assert eval_result['model_used'].endswith('claude-3-5-sonnet-20240620-v1:0')

def test_custom_evaluation_llama(llama_tester):
    """Test custom evaluation criteria with Llama"""
    product_description = "Our new smartphone features a 6.1-inch OLED display, 12MP camera, and all-day battery life."
    generated_description = generate_text("Write a short description of a smartphone's key features")
    
    custom_criteria = """
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
    
    eval_result = llama_tester.llm_evaluate(
        generated_description,
        product_description,
        criteria=custom_criteria
    )
    
    assert eval_result['passed'] is True
    assert 'effectiveness_score' in eval_result
    assert 'analysis' in eval_result
    assert isinstance(eval_result['analysis']['suggestions'], list)
    assert eval_result['model_used'].endswith('llama3-1-70b-instruct-v1:0')

def test_model_comparison(default_tester, llama_tester):
    """Test comparing evaluations from different models"""
    test_text = "The weather is sunny and warm today."
    reference = "It's a bright and hot day outside."
    
    claude_result = default_tester.llm_evaluate(test_text, reference)
    llama_result = llama_tester.llm_evaluate(test_text, reference)
    
    # Both evaluations should pass
    assert claude_result['passed'] is True
    assert llama_result['passed'] is True
    
    # Verify different models were used
    assert claude_result['model_used'] != llama_result['model_used']
    assert 'claude' in claude_result['model_used'].lower()
    assert 'llama' in llama_result['model_used'].lower()

@pytest.mark.parametrize("threshold,expected_pass", [
    (0.7, True),
    (0.9, False),
])
def test_different_thresholds(threshold, expected_pass):
    """Test semantic similarity with different thresholds"""
    tester = LLMTestMate(similarity_threshold=threshold)
    
    text1 = "The cat sat on the mat."
    text2 = "A feline rested on the rug."
    
    result = tester.semantic_similarity(text1, text2)
    assert result['passed'] is expected_pass
    assert result['threshold'] == threshold 