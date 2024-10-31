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

if __name__ == "__main__":
    main()