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
def alternative_tester():
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

def test_llm_evaluation_default(default_tester):
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

def test_custom_evaluation_alternative(alternative_tester):
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
    
    eval_result = alternative_tester.llm_evaluate(
        generated_description,
        product_description,
        criteria=custom_criteria
    )
    
    assert eval_result['passed'] is True
    assert 'effectiveness_score' in eval_result
    assert 'analysis' in eval_result
    assert isinstance(eval_result['analysis']['suggestions'], list)
    assert eval_result['model_used'].endswith('llama3-1-70b-instruct-v1:0')

def test_model_comparison(default_tester, alternative_tester):
    """Test comparing evaluations from different models"""
    test_text = "The weather is sunny and warm today."
    reference = "It's a bright and hot day outside."
    
    default_result = default_tester.llm_evaluate(test_text, reference)
    alternative_result = alternative_tester.llm_evaluate(test_text, reference)
    
    # Both evaluations should pass
    assert default_result['passed'] is True
    assert alternative_result['passed'] is True
    
    # Verify different models were used
    assert default_result['model_used'] != alternative_result['model_used']
    assert 'claude' in default_result['model_used'].lower()
    assert 'llama' in alternative_result['model_used'].lower()

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

def test_string_similarity(default_tester):
    """Test string similarity comparison using Damerau-Levenshtein distance"""
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "The quikc brown fox jumps over the lasy dog."  # Intentional typos
    
    result = default_tester.string_similarity(
        text1,
        text2,
        threshold=0.9
    )
    
    assert 0 <= result['similarity'] <= 1
    assert 0 <= result['distance'] <= 1
    assert result['similarity'] == pytest.approx(1.0 - result['distance'])
    assert 'passed' in result
    assert result['method'] == 'damerau-levenshtein'
    assert 'normalized' in result

def test_string_similarity_perfect_match(default_tester):
    """Test string similarity with identical strings"""
    text = "The quick brown fox."
    
    result = default_tester.string_similarity(text, text)
    
    assert result['similarity'] == 1.0
    assert result['distance'] == 0.0
    assert result['passed'] is True

def test_string_similarity_normalization(default_tester):
    """Test string similarity with different normalization settings"""
    text1 = "The CAT sat on the mat."
    text2 = "the  cat   sat  on  the    mat."  # Extra spaces
    
    # Test with both normalizations
    result1 = default_tester.string_similarity(
        text1,
        text2,
        normalize_case=True,
        normalize_whitespace=True
    )
    
    # Test without normalizations
    result2 = default_tester.string_similarity(
        text1,
        text2,
        normalize_case=False,
        normalize_whitespace=False
    )
    
    assert result1['similarity'] == 1.0  # Should be perfect match with normalization
    assert result1['distance'] == 0.0
    assert result2['similarity'] < 1.0  # Should not be perfect match without normalization
    assert result2['distance'] > 0.0

def test_string_similarity_threshold(default_tester):
    """Test string similarity threshold behavior"""
    text1 = "Hello world"
    text2 = "Helo world"  # One character deletion
    
    # Test with different thresholds
    result1 = default_tester.string_similarity(text1, text2, threshold=0.8)
    result2 = default_tester.string_similarity(text1, text2, threshold=0.99)
    
    assert result1['passed'] is True  # Should pass with lower threshold
    assert result2['passed'] is False  # Should fail with very high threshold

def test_string_similarity_methods(default_tester):
    """Test different string similarity methods"""
    text1 = "hello world"
    text2 = "helo world"  # One character deletion
    
    # Compare different methods
    methods = ["damerau-levenshtein", "levenshtein", "hamming", "jaro", "jaro-winkler", "indel"]
    results = {}
    
    for method in methods:
        results[method] = default_tester.string_similarity(
            text1, text2, method=method
        )
    
    # DamerauLevenshtein should handle transpositions better
    assert results["damerau-levenshtein"]["similarity"] >= results["levenshtein"]["similarity"]

def test_string_similarity_processor(default_tester):
    """Test custom string processor"""
    def remove_punctuation(s: str) -> str:
        return ''.join(c for c in s if c.isalnum() or c.isspace())
    
    text1 = "Hello, world!"
    text2 = "Hello world"
    
    # Without processor
    result1 = default_tester.string_similarity(text1, text2)
    
    # With processor
    result2 = default_tester.string_similarity(
        text1, text2,
        processor=remove_punctuation
    )
    
    assert result2["similarity"] > result1["similarity"]

def test_string_similarity_punctuation(default_tester):
    """Test string similarity with punctuation handling"""
    text1 = "Hello, world!"
    text2 = "Hello world"
    
    # With punctuation removal
    result1 = default_tester.string_similarity(
        text1,
        text2,
        remove_punctuation=True
    )
    
    # Without punctuation removal
    result2 = default_tester.string_similarity(
        text1,
        text2,
        remove_punctuation=False
    )
    
    assert result1["similarity"] == 1.0  # Should be perfect match with punctuation removal
    assert result2["similarity"] < 1.0  # Should not be perfect match with punctuation

def test_string_similarity_processor(default_tester):
    """Test custom string processor with punctuation removal"""
    def custom_processor(s: str) -> str:
        # Custom processing beyond punctuation removal
        return s.replace('world', 'earth')
    
    text1 = "Hello, world!"
    text2 = "Hello earth"
    
    # With processor
    result = default_tester.string_similarity(
        text1,
        text2,
        processor=custom_processor,
        remove_punctuation=True  # Will remove punctuation before custom processing
    )
    
    assert result["similarity"] == 1.0  # Should be perfect match after both transformations