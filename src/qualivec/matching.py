"""Semantic matching utilities for QualiVec."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


class SemanticMatcher:
    """Handles semantic matching for QualiVec."""
    
    def __init__(self, 
                 threshold: float = 0.7,
                 verbose: bool = True):
        """Initialize the semantic matcher.
        
        Args:
            threshold: Cosine similarity threshold for matching.
            verbose: Whether to print status messages.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")
            
        self.threshold = threshold
        self.verbose = verbose
    
    def match(self, 
              query_embeddings: np.ndarray, 
              reference_data: Dict[str, Any],
              return_similarities: bool = False) -> pd.DataFrame:
        """Match query embeddings against reference vectors.
        
        Args:
            query_embeddings: Embeddings of the query texts.
            reference_data: Dictionary with reference vector information.
            return_similarities: Whether to return all similarity scores.
            
        Returns:
            DataFrame with matching results.
        """
        if self.verbose:
            print(f"Matching {len(query_embeddings)} queries against {len(reference_data['embeddings'])} reference vectors")
            print(f"Using cosine similarity threshold: {self.threshold}")
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embeddings, reference_data['embeddings'])
        
        # Find best matches
        best_match_indices = np.argmax(similarities, axis=1)
        best_match_scores = np.max(similarities, axis=1)
        
        # Apply threshold
        matches_mask = best_match_scores >= self.threshold
        
        # Create results
        classes = np.array(reference_data['classes'])[best_match_indices]
        nodes = np.array(reference_data['nodes'])[best_match_indices]
        
        # Apply threshold (set to "Other" if below threshold)
        classes = np.where(matches_mask, classes, "Other")
        nodes = np.where(matches_mask, nodes, "")
        
        # Create result DataFrame
        results = pd.DataFrame({
            "predicted_class": classes,
            "matched_node": nodes,
            "similarity_score": best_match_scores
        })
        
        if return_similarities:
            results["all_similarities"] = list(similarities)
        
        if self.verbose:
            print(f"Matching complete: {matches_mask.sum()} matches above threshold ({matches_mask.mean():.1%})")
            print(f"Class distribution:\n{results['predicted_class'].value_counts().head(10)}")
        
        return results
    
    def classify_corpus(self,
                        corpus_embeddings: np.ndarray,
                        reference_data: Dict[str, Any],
                        corpus_df: pd.DataFrame) -> pd.DataFrame:
        """Classify an entire corpus using semantic matching.
        
        Args:
            corpus_embeddings: Embeddings of the corpus texts.
            reference_data: Dictionary with reference vector information.
            corpus_df: DataFrame containing the original corpus.
            
        Returns:
            DataFrame with classification results.
        """
        # Perform matching
        match_results = self.match(corpus_embeddings, reference_data)
        
        # Combine with original corpus
        result_df = pd.concat([corpus_df.reset_index(drop=True), 
                              match_results.reset_index(drop=True)], axis=1)
        
        if self.verbose:
            print(f"Classified {len(result_df)} documents")
            print(f"Class distribution:\n{result_df['predicted_class'].value_counts().head(10)}")
        
        return result_df
