"""Classification utilities for QualiVec."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from qualivec.data import DataLoader
from qualivec.embedding import EmbeddingModel
from qualivec.matching import SemanticMatcher


class Classifier:
    """Handles classification for QualiVec."""
    
    def __init__(self, 
                 embedding_model: Optional[EmbeddingModel] = None,
                 matcher: Optional[SemanticMatcher] = None,
                 verbose: bool = True):
        """Initialize the classifier.
        
        Args:
            embedding_model: Model for generating embeddings.
            matcher: Model for semantic matching.
            verbose: Whether to print status messages.
        """
        self.embedding_model = embedding_model
        self.matcher = matcher
        self.verbose = verbose
        self.data_loader = DataLoader(verbose=verbose)
    
    def load_models(self, 
                   model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                   threshold: float = 0.7):
        """Load embedding model and matcher.
        
        Args:
            model_name: Name of the HuggingFace model to use.
            threshold: Cosine similarity threshold for matching.
        """
        if self.verbose:
            print(f"Loading embedding model: {model_name}")
        
        self.embedding_model = EmbeddingModel(model_name=model_name, verbose=self.verbose)
        self.matcher = SemanticMatcher(threshold=threshold, verbose=self.verbose)
        
        if self.verbose:
            print("Models loaded successfully")
    
    def prepare_reference_vectors(self, 
                                 reference_path: str,
                                 class_column: str = "class",
                                 node_column: str = "matching_node") -> Dict[str, Any]:
        """Prepare reference vectors from a CSV file.
        
        Args:
            reference_path: Path to the CSV file with reference vectors.
            class_column: Name of the column containing class labels.
            node_column: Name of the column containing matching nodes.
            
        Returns:
            Dictionary with reference vector information.
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded. Call load_models first.")
        
        # Load reference vectors
        reference_df = self.data_loader.load_reference_vectors(
            reference_path, class_column=class_column, node_column=node_column
        )
        
        # Generate embeddings
        reference_data = self.embedding_model.embed_reference_vectors(
            reference_df, class_column=class_column, node_column=node_column
        )
        
        if self.verbose:
            print(f"Prepared {len(reference_data['embeddings'])} reference vectors")
            print(f"Unique classes: {len(reference_data['class_to_idx'])}")
        
        return reference_data
    
    def classify(self, 
                corpus_path: str,
                reference_data: Dict[str, Any],
                sentence_column: str = "sentence",
                output_path: Optional[str] = None) -> pd.DataFrame:
        """Classify texts in a corpus using reference vectors.
        
        Args:
            corpus_path: Path to the CSV file with corpus.
            reference_data: Dictionary with reference vector information.
            sentence_column: Name of the column containing sentences.
            output_path: Path to save the classification results.
            
        Returns:
            DataFrame with classification results.
        """
        if self.embedding_model is None or self.matcher is None:
            raise ValueError("Models not loaded. Call load_models first.")
        
        # Load corpus
        corpus_df = self.data_loader.load_corpus(corpus_path, sentence_column=sentence_column)
        
        # Generate embeddings
        corpus_embeddings = self.embedding_model.embed_dataframe(
            corpus_df, text_column=sentence_column
        )
        
        # Classify
        results_df = self.matcher.classify_corpus(
            corpus_embeddings, reference_data, corpus_df
        )
        
        # Save results if output path provided
        if output_path is not None:
            self.data_loader.save_dataframe(results_df, output_path)
            if self.verbose:
                print(f"Saved classification results to {output_path}")
        
        return results_df
    
    def evaluate_classification(self,
                              labeled_path: str,
                              reference_data: Dict[str, Any],
                              sentence_column: str = "sentence",
                              label_column: str = "label",
                              optimize_threshold: bool = False,
                              start: float = 0.5,
                              end: float = 0.9,
                              step: float = 0.01) -> Dict[str, Any]:
        """Evaluate classification performance on labeled data.
        
        Args:
            labeled_path: Path to the CSV file with labeled data.
            reference_data: Dictionary with reference vector information.
            sentence_column: Name of the column containing sentences.
            label_column: Name of the column containing true labels.
            optimize_threshold: Whether to optimize the threshold.
            start: Start threshold value for optimization.
            end: End threshold value for optimization.
            step: Threshold step size for optimization.
            
        Returns:
            Dictionary with evaluation results.
        """
        from qualivec.evaluation import Evaluator
        from qualivec.optimization import ThresholdOptimizer
        
        if self.embedding_model is None:
            raise ValueError("Embedding model not loaded. Call load_models first.")
        
        # Load labeled data
        labeled_df = self.data_loader.load_labeled_data(labeled_path, label_column=label_column)
        
        # Validate labels
        valid = self.data_loader.validate_labels(
            labeled_df, 
            pd.DataFrame({
                "class": reference_data["classes"]
            }).drop_duplicates(),
            label_column=label_column,
            class_column="class"
        )
        
        if not valid and self.verbose:
            print("Warning: Some labels in the labeled data are not in reference vectors")
        
        # Generate embeddings
        labeled_embeddings = self.embedding_model.embed_dataframe(
            labeled_df, text_column=sentence_column
        )
        
        # True labels
        true_labels = labeled_df[label_column].tolist()
        
        if optimize_threshold:
            # Optimize threshold
            if self.verbose:
                print("Optimizing threshold...")
            
            optimizer = ThresholdOptimizer(verbose=self.verbose)
            optimization_results = optimizer.optimize(
                labeled_embeddings,
                reference_data,
                true_labels,
                start=start,
                end=end,
                step=step,
                metric="f1_macro"
            )
            
            # Update matcher with optimal threshold
            self.matcher = SemanticMatcher(threshold=optimization_results["optimal_threshold"], 
                                          verbose=self.verbose)
            
            return optimization_results
        else:
            # Evaluate with current threshold
            if self.matcher is None:
                raise ValueError("Matcher not loaded. Call load_models first.")
            
            # Get predictions
            match_results = self.matcher.match(labeled_embeddings, reference_data)
            predicted_labels = match_results["predicted_class"].tolist()
            
            # Evaluate
            evaluator = Evaluator(verbose=self.verbose)
            eval_results = evaluator.bootstrap_evaluate(true_labels, predicted_labels)
            
            return eval_results
