"""Test file for QualiVec evaluation module."""

import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.qualivec.evaluation import Evaluator

def generate_sample_data(n_samples=1000, n_classes=5, error_rate=0.2, random_seed=42):
    """Generate sample classification data with controlled error rate.
    
    Args:
        n_samples: Number of samples to generate
        n_classes: Number of classes
        error_rate: Rate of misclassification
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (true_labels, predicted_labels, class_names)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Create class names
    class_names = [f"Class_{i}" for i in range(n_classes)]
    
    # Generate true labels with balanced classes
    true_labels = []
    for i in range(n_samples):
        true_labels.append(class_names[i % n_classes])
    
    # Shuffle to make it more realistic
    random.shuffle(true_labels)
    
    # Generate predictions with controlled error rate
    predicted_labels = []
    for true_label in true_labels:
        if random.random() < error_rate:
            # Make a mistake - choose a different class
            other_classes = [c for c in class_names if c != true_label]
            predicted_labels.append(random.choice(other_classes))
        else:
            # Correct prediction
            predicted_labels.append(true_label)
    
    return true_labels, predicted_labels, class_names

def test_basic_evaluation():
    """Test basic evaluation functionality."""
    print("\n===== Testing Basic Evaluation =====")
    
    # Generate sample data
    true_labels, predicted_labels, class_names = generate_sample_data(
        n_samples=500, 
        n_classes=4,
        error_rate=0.3
    )
    
    # Create evaluator
    evaluator = Evaluator(verbose=True)
    
    # Run evaluation
    results = evaluator.evaluate(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=class_names
    )
    
    # Print class-wise metrics
    print("\nClass-wise metrics:")
    for cls in class_names:
        print(f"  {cls}: F1={results['class_metrics']['f1'][cls]:.4f}, "
              f"Precision={results['class_metrics']['precision'][cls]:.4f}, "
              f"Recall={results['class_metrics']['recall'][cls]:.4f}, "
              f"Support={results['class_metrics']['support'][cls]}")
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        confusion_matrix=results['confusion_matrix'],
        class_names=results['confusion_matrix_labels']
    )
    
    return results

def test_bootstrap_evaluation():
    """Test bootstrap evaluation functionality."""
    print("\n===== Testing Bootstrap Evaluation =====")
    
    # Generate sample data
    true_labels, predicted_labels, _ = generate_sample_data(
        n_samples=200,  # Smaller dataset for faster bootstrap
        n_classes=3,
        error_rate=0.25
    )
    
    # Create evaluator
    evaluator = Evaluator(verbose=True)
    
    # Run bootstrap evaluation with fewer iterations for test
    bootstrap_results = evaluator.bootstrap_evaluate(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        n_iterations=100,  # Reduced for testing
        confidence_levels=[0.9, 0.95],
        random_seed=42
    )
    
    # Plot bootstrap distributions
    evaluator.plot_bootstrap_distributions(bootstrap_results)
    
    return bootstrap_results

def main():
    """Run all tests."""
    print("Testing QualiVec Evaluation Module")
    
    # Run basic evaluation test
    basic_results = test_basic_evaluation()
    
    # Run bootstrap evaluation test
    bootstrap_results = test_bootstrap_evaluation()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()
