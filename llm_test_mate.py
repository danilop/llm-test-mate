from typing import Any, Dict, Optional, Union
from litellm import completion
from sentence_transformers import SentenceTransformer
import json

class LLMTestMate:
    """
    A class for evaluating LLM-generated content using semantic similarity 
    and LLM-based evaluation.
    """
    
    DEFAULT_LLM_MODEL = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
    DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    
    DEFAULT_CRITERIA = """
    Evaluate the similarity between the provided text and the reference.
    Consider:
    1. Semantic Similarity: Do they convey the same meaning?
    2. Content Overlap: Are the key points and information similar?
    3. Intent Match: Do they serve the same purpose?
    4. Contextual Alignment: Are they appropriate for the same context?
    
    Return only JSON with:
    {
        "passed": boolean,
        "similarity_score": float (0-1),
        "analysis": {
            "semantic_match": string (explanation of meaning similarity),
            "content_match": string (explanation of content overlap),
            "key_differences": list[string] (if any)
        }
    }
    """
    
    def __init__(
        self,
        llm_model: str = DEFAULT_LLM_MODEL,
        embedding_model: Union[str, SentenceTransformer, None] = None,
        similarity_threshold: Optional[float] = 0.8,
        evaluation_criteria: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **llm_kwargs
    ):
        """
        Initialize LLMTestMate.
        
        Args:
            llm_model: Model identifier for LLM evaluation
            embedding_model: Model for semantic similarity (string name or instance)
            similarity_threshold: Default threshold for semantic similarity tests
            evaluation_criteria: Default criteria for LLM evaluation
            temperature: Temperature for LLM responses
            max_tokens: Maximum tokens for LLM responses
            **llm_kwargs: Additional kwargs for LiteLLM
        """
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_kwargs = llm_kwargs
        self.similarity_threshold = similarity_threshold
        self.evaluation_criteria = evaluation_criteria or self.DEFAULT_CRITERIA
        
        # Initialize embedding model
        if embedding_model is None:
            self._embedding_model = SentenceTransformer(self.DEFAULT_EMBEDDING_MODEL)
        elif isinstance(embedding_model, str):
            self._embedding_model = SentenceTransformer(embedding_model)
        else:
            self._embedding_model = embedding_model

    def _cosine_similarity(self, vec1: list, vec2: list) -> float:
        """
        Calculate cosine similarity between two vectors.
        """
        if not any(vec1) or not any(vec2):
            return 0.0
            
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = (sum(x * x for x in vec1)) ** 0.5
        norm2 = (sum(x * x for x in vec2)) ** 0.5
        return dot_product / (norm1 * norm2)

    def semantic_similarity(
        self,
        text1: str, 
        text2: str, 
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: Generated text to evaluate
            text2: Reference text to compare against
            threshold: Optional threshold to override default
            
        Returns:
            Dict with similarity score and pass/fail status
        """
        emb1 = self._embedding_model.encode(text1)
        emb2 = self._embedding_model.encode(text2)
        
        similarity = self._cosine_similarity(emb1, emb2)
        
        result = {
            "similarity": float(similarity),
            "embedding_model": (
                self._embedding_model.get_model_card() 
                if hasattr(self._embedding_model, 'get_model_card') 
                else str(self._embedding_model)
            )
        }
        
        # Use instance threshold if none provided
        if threshold is None:
            threshold = self.similarity_threshold
            
        if threshold is not None:
            result["passed"] = similarity >= threshold
            result["threshold"] = threshold
        
        return result

    def llm_evaluate(
        self,
        text: str,
        reference: str,
        criteria: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate text using an LLM.
        
        Args:
            text: Generated text to evaluate
            reference: Reference text
            criteria: Optional criteria to override default
            model: Override default LLM model
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            **kwargs: Additional LiteLLM parameters
            
        Returns:
            Dict with evaluation results
        """
        # Use instance criteria if none provided
        evaluation_criteria = criteria or self.evaluation_criteria
        
        prompt = f"""
        Evaluate the following text against the reference:
        
        Generated text:
        {text}
        
        Reference text:
        {reference}
        
        Evaluation criteria:
        {evaluation_criteria}
        
        Provide evaluation as valid JSON.
        """
        
        # Merge kwargs with instance defaults
        eval_kwargs = self.llm_kwargs.copy()
        eval_kwargs.update(kwargs)
        
        response = completion(
            model=model or self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **eval_kwargs
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            result["model_used"] = response.model
            return result
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse LLM response JSON: {str(e)}",
                "raw_response": response.choices[0].message.content,
                "model_used": response.model
            }
