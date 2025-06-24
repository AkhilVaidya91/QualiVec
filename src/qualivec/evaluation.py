"""Evaluation utilities for QualiVec."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class Evaluator:
    """Handles evaluation for QualiVec."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the evaluator.
        
        Args:
            verbose: Whether to print status messages.
        """
        self.verbose = verbose
    
    def evaluate(self, 
                true_labels: List[str], 
                predicted_labels: List[str],
                class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Evaluate predictions against true labels.
        
        Args:
            true_labels: List of true class labels.
            predicted_labels: List of predicted class labels.
            class_names: List of class names for detailed metrics.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        if len(true_labels) != len(predicted_labels):
            raise ValueError(f"Length mismatch: {len(true_labels)} true labels vs {len(predicted_labels)} predictions")
        
        if self.verbose:
            print(f"Evaluating {len(true_labels)} predictions")
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # If class_names not provided, use unique values from true and predicted
        if class_names is None:
            class_names = sorted(set(true_labels) | set(predicted_labels))
        
        # Calculate precision, recall, F1 (macro average)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, predicted_labels, average='macro'
        )
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels, labels=class_names, average=None
        )
        
        # Create class-wise metrics
        class_metrics = {
            "precision": {cls: p for cls, p in zip(class_names, precision)},
            "recall": {cls: r for cls, r in zip(class_names, recall)},
            "f1": {cls: f for cls, f in zip(class_names, f1)},
            "support": {cls: s for cls, s in zip(class_names, support)}
        }
        
        # Create confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=class_names)
        
        # Compile results
        results = {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "class_metrics": class_metrics,
            "confusion_matrix": cm,
            "confusion_matrix_labels": class_names,
            "n_samples": len(true_labels)
        }
        
        if self.verbose:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision (macro): {precision_macro:.4f}")
            print(f"Recall (macro): {recall_macro:.4f}")
            print(f"F1 (macro): {f1_macro:.4f}")
        
        return results
    
    def bootstrap_evaluate(self,
                          true_labels: List[str],
                          predicted_labels: List[str],
                          n_iterations: int = 1000,
                          confidence_levels: List[float] = [0.9, 0.95, 0.99],
                          random_seed: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate with bootstrap confidence intervals.
        
        Args:
            true_labels: List of true class labels.
            predicted_labels: List of predicted class labels.
            n_iterations: Number of bootstrap iterations.
            confidence_levels: Confidence levels to compute.
            random_seed: Random seed for reproducibility.
            
        Returns:
            Dictionary with evaluation metrics and confidence intervals.
        """
        if len(true_labels) != len(predicted_labels):
            raise ValueError(f"Length mismatch: {len(true_labels)} true labels vs {len(predicted_labels)} predictions")
        
        if self.verbose:
            print(f"Running bootstrap evaluation with {n_iterations} iterations")
        
        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize storage for bootstrap results
        bootstrap_metrics = {
            "accuracy": [],
            "precision_macro": [],
            "recall_macro": [],
            "f1_macro": []
        }
        
        # Original evaluation
        original_results = self.evaluate(true_labels, predicted_labels)
        
        # Run bootstrap iterations
        n_samples = len(true_labels)
        
        for _ in tqdm(range(n_iterations), disable=not self.verbose):
            # Sample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Get bootstrap sample
            bootstrap_true = [true_labels[i] for i in indices]
            bootstrap_pred = [predicted_labels[i] for i in indices]
            
            # Evaluate
            results = self.evaluate(bootstrap_true, bootstrap_pred)
            
            # Store results
            bootstrap_metrics["accuracy"].append(results["accuracy"])
            bootstrap_metrics["precision_macro"].append(results["precision_macro"])
            bootstrap_metrics["recall_macro"].append(results["recall_macro"])
            bootstrap_metrics["f1_macro"].append(results["f1_macro"])
        
        # Calculate confidence intervals
        confidence_intervals = {}
        
        for metric, values in bootstrap_metrics.items():
            confidence_intervals[metric] = {}
            for level in confidence_levels:
                lower_percentile = (1 - level) / 2 * 100
                upper_percentile = (1 + level) / 2 * 100
                
                lower = np.percentile(values, lower_percentile)
                upper = np.percentile(values, upper_percentile)
                
                confidence_intervals[metric][level] = (lower, upper)
        
        # Combine results
        results = {
            "point_estimates": {
                "accuracy": original_results["accuracy"],
                "precision_macro": original_results["precision_macro"],
                "recall_macro": original_results["recall_macro"],
                "f1_macro": original_results["f1_macro"]
            },
            "confidence_intervals": confidence_intervals,
            "bootstrap_distribution": bootstrap_metrics,
            "n_iterations": n_iterations,
            "n_samples": n_samples
        }
        
        if self.verbose:
            print(f"Bootstrap evaluation complete")
            print(f"Accuracy: {results['point_estimates']['accuracy']:.4f}")
            for level in confidence_levels:
                lower, upper = results['confidence_intervals']['accuracy'][level]
                print(f"  {level*100:.0f}% CI: ({lower:.4f}, {upper:.4f})")
        
        return results
    
    def plot_confusion_matrix(self, 
                             confusion_matrix: np.ndarray, 
                             class_names: List[str],
                             figsize: Tuple[int, int] = (10, 8),
                             title: str = "Confusion Matrix"):
        """Plot a confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix as numpy array.
            class_names: List of class names.
            figsize: Figure size as (width, height).
            title: Plot title.
        """
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt="d", 
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_bootstrap_distributions(self, bootstrap_results: Dict[str, Any], figsize: Tuple[int, int] = (12, 8)):
        """Plot bootstrap distributions for key metrics.
        
        Args:
            bootstrap_results: Results from bootstrap_evaluate.
            figsize: Figure size as (width, height).
        """
        metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        
        plt.figure(figsize=figsize)
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i+1)
            
            # Get distribution data
            values = bootstrap_results["bootstrap_distribution"][metric]
            
            # Plot histogram
            sns.histplot(values, kde=True)
            
            # Add point estimate
            point_est = bootstrap_results["point_estimates"][metric]
            plt.axvline(point_est, color='red', linestyle='--', label=f'Point est: {point_est:.4f}')
            
            # Add confidence intervals
            for level, (lower, upper) in bootstrap_results["confidence_intervals"][metric].items():
                plt.axvline(lower, color='green', linestyle=':', 
                          label=f'{level*100:.0f}% CI: ({lower:.4f}, {upper:.4f})')
                plt.axvline(upper, color='green', linestyle=':')
            
            plt.title(f"{metric.replace('_', ' ').title()}")
            
            if i == 0:  # Only add legend to first plot
                plt.legend(loc='best')
        
        plt.tight_layout()
        plt.show()
