import json

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

# Example: String similarity test using Damerau-Levenshtein distance
print("1. String similarity test")
result = tester.string_similarity(
    reference_text,
    generated_text,
    threshold=0.4
)

print(f"Similarity score: {result['similarity']:.2f}")
print(f"Passed threshold: {result['passed']}")

# Simple similarity check using default settings
print("2. Semantic similarity test")
result = tester.semantic_similarity(
    generated_text, 
    reference_text
)
print(f"Similarity score: {result['similarity']:.2f}")
print(f"Passed threshold: {result['passed']}")

# LLM-based evaluation
print("3. LLM-based evaluation")
result = tester.llm_evaluate(
    generated_text,
    reference_text
)
print("Evaluation result:")
print(json.dumps(result, indent=4))
