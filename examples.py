import json

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

def main():
    # First use the default model (Claude)
    print("\n=== Testing with Default Model (Claude) ===")
    default_tester = LLMTestMate(
        similarity_threshold=0.8,
        temperature=0.7
    )

    # Example 1: Basic semantic similarity comparison with Claude
    print("\n=== Example 1: Semantic Similarity Test (Claude) ===")
    reference_text = "The quick brown fox jumps over the lazy dog."
    test_text = "A swift brown fox leaps above a sleepy canine."
    print(f"Test text: {test_text}")
    print(f"Reference text: {reference_text}")

    result = default_tester.semantic_similarity(
        test_text, 
        reference_text
    )
    print(f"Similarity score: {result['similarity']:.2f}")
    print(f"Passed threshold: {result.get('passed', 'No threshold set')}")

    # Example 2: LLM-based evaluation with Claude
    print("\n=== Example 2: LLM Evaluation with Claude ===")
    reference_summary = "Python is a high-level programming language known for its simplicity and readability."
    prompt = "Write a one-sentence description of Python programming language"
    generated_summary = generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated summary: {generated_summary}")
    print(f"Reference summary: {reference_summary}")

    eval_result = default_tester.llm_evaluate(
        generated_summary,
        reference_summary
    )
    print("Generated summary:", generated_summary)
    print("Evaluation result (Claude):", json.dumps(eval_result, indent=2))

    # New Example: String similarity comparison
    print("\n=== Example: String Similarity Test ===")
    text1 = "The quick brown fox jumps over the lazy dog."
    text2 = "The quikc brown fox jumps over the lasy dog."  # Intentional typos
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")

    string_result = default_tester.string_similarity(
        text1,
        text2,
        threshold=0.9  # High threshold for string similarity
    )
    print(f"String similarity score: {string_result['similarity']:.2f}")
    print(f"Edit distance: {string_result['distance']:.2f}")
    print(f"Passed threshold: {string_result.get('passed', 'No threshold set')}")

    # Compare all similarity methods
    semantic_result = default_tester.semantic_similarity(text1, text2)
    llm_result = default_tester.llm_evaluate(text1, text2)
    
    print(f"\nComparison of different similarity measures:")
    print(f"String similarity: {string_result['similarity']:.2f}")
    print(f"Semantic similarity: {semantic_result['similarity']:.2f}")
    print(f"LLM similarity score: {llm_result['similarity_score']:.2f}")

    # Now switch to Llama model
    print("\n=== Testing with Llama Model ===")
    llama_tester = LLMTestMate(
        llm_model=ALTERNATIVE_LLM_MODEL,
        similarity_threshold=0.8,
        temperature=0.7
    )

    # Example 3: Custom evaluation criteria with Llama
    print("\n=== Example 3: LLM Evaluation with Llama ===")
    product_description = "Our new smartphone features a 6.1-inch OLED display, 12MP camera, and all-day battery life."
    prompt = "Write a short description of a smartphone's key features" 
    generated_description = generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated description: {generated_description}")
    print(f"Product description: {product_description}")

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
    
    print(f"Custom criteria: {custom_criteria}")

    eval_result = llama_tester.llm_evaluate(
        generated_description,
        product_description,
        criteria=custom_criteria
    )
    print("Evaluation result (Llama):", json.dumps(eval_result, indent=2))

    # Compare both models on the same task
    print("\n=== Model Comparison on Same Task ===")
    test_text = "The weather is sunny and warm today."
    reference = "It's a bright and hot day outside."
    print(f"Test text: {test_text}")
    print(f"Reference: {reference}")
    
    claude_result = default_tester.llm_evaluate(test_text, reference)
    llama_result = llama_tester.llm_evaluate(test_text, reference)
    
    print("\nClaude evaluation:", json.dumps(claude_result, indent=2))
    print("\nLlama evaluation:", json.dumps(llama_result, indent=2))

    # Comprehensive String Similarity Examples
    print("\n=== String Similarity Examples ===")

    # 1. Compare different methods with various text types
    print("\n1. Comparing different similarity methods:")
    test_pairs = [
        ("John Smith", "Jon Smyth"),           # Name variation
        ("Hello World", "Helo World"),         # Simple typo
        ("The Quick Fox", "The Quick Fox"),    # Perfect match
        ("Quick Fox", "The Quick Fox"),        # Different length
        ("The quick brown fox", "The quikc brown fox")  # Transposed letters
    ]

    methods = ["damerau-levenshtein", "levenshtein", "hamming", "jaro", "jaro-winkler", "indel"]

    for text1, text2 in test_pairs:
        print(f"\nComparing: '{text1}' vs '{text2}'")
        for method in methods:
            result = default_tester.string_similarity(
                text1,
                text2,
                method=method
            )
            print(f"{method:20}: {result['similarity']:.3f}")

    # 2. Normalization options
    print("\n2. Text normalization options:")
    text1 = "Hello,  WORLD!"
    text2 = "hello world"

    normalization_tests = [
        {"normalize_case": False, "normalize_whitespace": False, "remove_punctuation": False},
        {"normalize_case": True, "normalize_whitespace": False, "remove_punctuation": False},
        {"normalize_case": True, "normalize_whitespace": True, "remove_punctuation": False},
        {"normalize_case": True, "normalize_whitespace": True, "remove_punctuation": True},
    ]

    for options in normalization_tests:
        result = default_tester.string_similarity(
            text1,
            text2,
            **options
        )
        print(f"\nOptions: {options}")
        print(f"Similarity: {result['similarity']:.3f}")

    # 3. Custom text processor
    print("\n3. Custom text processor:")
    def remove_special_chars(text: str) -> str:
        return ''.join(c for c in text if c.isalnum() or c.isspace())

    text1 = "Hello! @#$ World"
    text2 = "Hello World"

    result = default_tester.string_similarity(
        text1,
        text2,
        processor=remove_special_chars
    )
    print(f"Original texts: '{text1}' vs '{text2}'")
    print(f"Similarity with processor: {result['similarity']:.3f}")

    # 4. Combined options example
    print("\n4. Combined options:")
    text1 = "Hello,  WORLD! @#$"
    text2 = "hello world"
    
    result = default_tester.string_similarity(
        text1,
        text2,
        method="damerau-levenshtein",
        normalize_case=True,
        normalize_whitespace=True,
        remove_punctuation=True,
        processor=remove_special_chars,
        threshold=0.9
    )
    print(f"Original texts: '{text1}' vs '{text2}'")
    print(f"Similarity with all options: {result['similarity']:.3f}")
    print(f"Passed threshold: {result['passed']}")

if __name__ == "__main__":
    main()