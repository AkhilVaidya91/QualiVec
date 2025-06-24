"""Threshold optimization utilities for QualiVec."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from qualivec.matching import SemanticMatcher
from qualivec.evaluation import Evaluator


class ThresholdOptimizer:
    """Handles threshold optimization for QualiVec."""
    
    def __init__(self, 
                 verbose: bool = True):
        """Initialize the threshold optimizer.
        
        Args:
            verbose: Whether to print status messages.
        """
        self.verbose = verbose
        self.evaluator = Evaluator(verbose=False)
    
    def optimize(self,
                query_embeddings: np.ndarray,
                reference_data: Dict[str, Any],
                true_labels: List[str],
                start: float = 0.0,
                end: float = 1.0,
                step: float = 0.01,
                metric: str = "f1_macro",
                bootstrap: bool = True,
                n_bootstrap: int = 100,
                confidence_level: float = 0.95,
                random_seed: Optional[int] = None) -> Dict[str, Any]:
        """Find the optimal similarity threshold.
        
        Args:
            query_embeddings: Embeddings of the query texts.
            reference_data: Dictionary with reference vector information.
            true_labels: True class labels for evaluation.
            start: Start threshold value.
            end: End threshold value.
            step: Threshold step size.
            metric: Metric to optimize ("accuracy", "precision_macro", "recall_macro", "f1_macro").
            bootstrap: Whether to use bootstrap evaluation.
            n_bootstrap: Number of bootstrap iterations.
            confidence_level: Confidence level for bootstrap.
            random_seed: Random seed for reproducibility.
            
        Returns:
            Dictionary with optimization results.
        """
        if not 0 <= start < end <= 1:
            raise ValueError("Threshold range must be between 0 and 1")
        
        if metric not in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            raise ValueError(f"Unsupported metric: {metric}")
        
        if self.verbose:
            print(f"Optimizing threshold for {metric}")
            print(f"Threshold range: {start} to {end} (step: {step})")
        
        # Generate threshold values
        thresholds = np.arange(start, end + step/2, step)
        
        # Initialize results storage
        results = {
            "thresholds": [],
            "accuracy": [],
            "precision_macro": [],
            "recall_macro": [],
            "f1_macro": [],
            "class_distribution": []
        }
        
        if bootstrap:
            results["confidence_intervals"] = []
        
        # Evaluate each threshold
        for threshold in tqdm(thresholds, disable=not self.verbose):
            # Create matcher with current threshold
            matcher = SemanticMatcher(threshold=threshold, verbose=False)
            
            # Get predictions
            match_results = matcher.match(query_embeddings, reference_data)
            predicted_labels = match_results["predicted_class"].tolist()
            
            # Calculate class distribution
            class_distribution = pd.Series(predicted_labels).value_counts().to_dict()
            
            # Evaluate
            if bootstrap:
                eval_results = self.evaluator.bootstrap_evaluate(
                    true_labels, 
                    predicted_labels,
                    n_iterations=n_bootstrap,
                    confidence_levels=[confidence_level],
                    random_seed=random_seed
                )
                
                # Extract point estimates
                point_estimates = eval_results["point_estimates"]
                
                # Extract confidence intervals
                ci = {m: eval_results["confidence_intervals"][m][confidence_level] 
                     for m in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]}
                
                results["confidence_intervals"].append(ci)
            else:
                eval_results = self.evaluator.evaluate(true_labels, predicted_labels)
                point_estimates = {
                    "accuracy": eval_results["accuracy"],
                    "precision_macro": eval_results["precision_macro"],
                    "recall_macro": eval_results["recall_macro"],
                    "f1_macro": eval_results["f1_macro"]
                }
            
            # Store results
            results["thresholds"].append(threshold)
            results["accuracy"].append(point_estimates["accuracy"])
            results["precision_macro"].append(point_estimates["precision_macro"])
            results["recall_macro"].append(point_estimates["recall_macro"])
            results["f1_macro"].append(point_estimates["f1_macro"])
            results["class_distribution"].append(class_distribution)
        
        # Find optimal threshold
        optimal_idx = np.argmax(results[metric])
        optimal_threshold = results["thresholds"][optimal_idx]
        optimal_metrics = {
            "accuracy": results["accuracy"][optimal_idx],
            "precision_macro": results["precision_macro"][optimal_idx],
            "recall_macro": results["recall_macro"][optimal_idx],
            "f1_macro": results["f1_macro"][optimal_idx]
        }
        
        if bootstrap:
            optimal_ci = results["confidence_intervals"][optimal_idx]
        else:
            optimal_ci = None
        
        # Compile results
        optimization_results = {
            "optimal_threshold": optimal_threshold,
            "optimal_metrics": optimal_metrics,
            "optimal_confidence_intervals": optimal_ci,
            "results_by_threshold": results,
            "optimized_metric": metric,
            "n_thresholds": len(thresholds)
        }
        
        if self.verbose:
            print(f"Optimal threshold: {optimal_threshold:.4f}")
            print(f"Optimal {metric}: {optimal_metrics[metric]:.4f}")
            if bootstrap:
                lower, upper = optimal_ci[metric]
                print(f"  {confidence_level*100:.0f}% CI: ({lower:.4f}, {upper:.4f})")
        
        return optimization_results
    
    def plot_optimization_results(self, 
                                 results: Dict[str, Any], 
                                 metrics: Optional[List[str]] = None,
                                 figsize: Tuple[int, int] = (12, 6)):
        """Plot optimization results.
        
        Args:
            results: Results from optimize method.
            metrics: List of metrics to plot.
            figsize: Figure size as (width, height).
        """
        if metrics is None:
            metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
        
        plt.figure(figsize=figsize)
        
        # Get data
        thresholds = results["results_by_threshold"]["thresholds"]
        
        # Plot metrics
        for metric in metrics:
            values = results["results_by_threshold"][metric]
            plt.plot(thresholds, values, label=metric.replace("_", " ").title())
            
            # Highlight optimal threshold
            if metric == results["optimized_metric"]:
                optimal_threshold = results["optimal_threshold"]
                optimal_value = results["optimal_metrics"][metric]
                plt.scatter([optimal_threshold], [optimal_value], color='red', s=100, zorder=5)
                plt.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.5,
                          label=f"Optimal Threshold: {optimal_threshold:.4f}")
        
        plt.xlabel("Threshold")
        plt.ylabel("Metric Value")
        plt.title("Threshold Optimization Results")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_class_distribution(self, 
                               results: Dict[str, Any],
                               top_n: int = 10,
                               figsize: Tuple[int, int] = (12, 8)):
        """Plot class distribution at different thresholds.
        
        Args:
            results: Results from optimize method.
            top_n: Number of top classes to show.
            figsize: Figure size as (width, height).
        """
        # Get data
        thresholds = results["results_by_threshold"]["thresholds"]
        distributions = results["results_by_threshold"]["class_distribution"]
        
        # Find all classes
        all_classes = set()
        for dist in distributions:
            all_classes.update(dist.keys())
        
        # Count total occurrences to find top classes
        total_counts = {}
        for cls in all_classes:
            total_counts[cls] = sum(dist.get(cls, 0) for dist in distributions)
        
        # Get top N classes
        top_classes = sorted(all_classes, key=lambda x: total_counts[x], reverse=True)[:top_n]
        
        # Create data for plot
        data = []
        for i, threshold in enumerate(thresholds):
            dist = distributions[i]
            for cls in top_classes:
                data.append({
                    "Threshold": threshold,
                    "Class": cls,
                    "Count": dist.get(cls, 0)
                })
        
        # Create dataframe
        df = pd.DataFrame(data)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Use seaborn for line plot
        sns.lineplot(data=df, x="Threshold", y="Count", hue="Class")
        
        # Add vertical line for optimal threshold
        optimal_threshold = results["optimal_threshold"]
        plt.axvline(optimal_threshold, color='red', linestyle='--', alpha=0.5,
                  label=f"Optimal Threshold: {optimal_threshold:.4f}")
        
        plt.title("Class Distribution by Threshold")
        plt.xlabel("Threshold")
        plt.ylabel("Count")
        plt.legend(title="Class")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
