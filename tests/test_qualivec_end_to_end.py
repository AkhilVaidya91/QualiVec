import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qualivec.data import DataLoader
from src.qualivec.embedding import EmbeddingModel
from src.qualivec.matching import SemanticMatcher
from src.qualivec.classification import Classifier
from src.qualivec.evaluation import Evaluator
from src.qualivec.optimization import ThresholdOptimizer

def prepare_reference_data(input_path, output_path):
    """Prepare reference data by renaming columns."""
    print(f"Preparing reference data from {input_path}")
    
    # Load the reference data
    reference_df = pd.read_csv(input_path)
    
    # Rename columns to match what QualiVec expects
    reference_df = reference_df.rename(columns={
        'tag': 'class',
        'sentence': 'matching_node'
    })
    
    # Save the prepared reference data
    reference_df.to_csv(output_path, index=False)
    print(f"Reference data prepared and saved to {output_path}")
    
    return reference_df

def prepare_labeled_data(input_path, output_path):
    """Prepare labeled data by renaming columns and converting labels."""
    print(f"Preparing labeled data from {input_path}")
    
    # Load the labeled data
    labeled_df = pd.read_csv(input_path)
    
    # Rename columns to match what QualiVec expects
    labeled_df = labeled_df.rename(columns={
        'Label': 'label'  # Rename to lowercase to match QualiVec expectations
    })
    
    # Convert '0' labels to 'Other'
    labeled_df['label'] = labeled_df['label'].replace('0', 'Other')
    
    # Report the label conversion if any '0' values were found
    if (labeled_df['label'] == 'Other').any():
        other_count = (labeled_df['label'] == 'Other').sum()
        print(f"Converted {other_count} labels from '0' to 'Other'")
    
    # Save the prepared labeled data
    labeled_df.to_csv(output_path, index=False)
    print(f"Labeled data prepared and saved to {output_path}")
    
    return labeled_df

def main(labeled_data_path, reference_data_path):
    """Run end-to-end test of QualiVec with the given data."""
    print("Starting end-to-end test of QualiVec package")
    
    # Create directories for prepared data if they don't exist
    os.makedirs('prepared_data', exist_ok=True)
    
    # Prepare data
    prepared_reference_path = 'prepared_data/reference.csv'
    prepared_labeled_path = 'prepared_data/labeled.csv'
    
    reference_df = prepare_reference_data(reference_data_path, prepared_reference_path)
    labeled_df = prepare_labeled_data(labeled_data_path, prepared_labeled_path)
    
    print(f"Using full dataset with {len(labeled_df)} samples for optimization and evaluation")
    
    # Initialize classifier
    classifier = Classifier(verbose=True)
    classifier.load_models(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        threshold=0.7  # Initial threshold (will be optimized)
    )
    
    # Prepare reference vectors
    reference_data = classifier.prepare_reference_vectors(
        reference_path=prepared_reference_path,
        class_column='class',
        node_column='matching_node'
    )
    
    # Optimize threshold using the full dataset
    print("\nOptimizing threshold using full dataset...")
    optimization_results = classifier.evaluate_classification(
        labeled_path=prepared_labeled_path,
        reference_data=reference_data,
        sentence_column='sentence',
        label_column='label',
        optimize_threshold=True,
        start=0.5,
        end=0.9,
        step=0.01
    )
    
    # Plot optimization results
    optimizer = ThresholdOptimizer(verbose=True)
    optimizer.plot_optimization_results(optimization_results)
    optimizer.plot_class_distribution(optimization_results)
    
    optimal_threshold = optimization_results["optimal_threshold"]
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    
    # Evaluate on the same full dataset
    print("\nEvaluating on full dataset...")
    classifier.matcher = SemanticMatcher(threshold=optimal_threshold, verbose=True)
    
    # Load labeled data and generate embeddings
    embedding_model = classifier.embedding_model
    data_loader = DataLoader(verbose=True)
    full_df = data_loader.load_labeled_data(prepared_labeled_path, label_column='label')
    
    full_embeddings = embedding_model.embed_dataframe(full_df, text_column='sentence') # type: ignore
    
    # Classify full dataset
    match_results = classifier.matcher.match(full_embeddings, reference_data)
    predicted_labels = match_results["predicted_class"].tolist()
    true_labels = full_df['label'].tolist()
    
    # Evaluate results
    evaluator = Evaluator(verbose=True)
    eval_results = evaluator.evaluate(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=list(set(true_labels) | set(predicted_labels))
    )
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(
        confusion_matrix=eval_results['confusion_matrix'],
        class_names=eval_results['confusion_matrix_labels']
    )
    
    # Bootstrap evaluation for confidence intervals
    print("\nRunning bootstrap evaluation...")
    bootstrap_results = evaluator.bootstrap_evaluate(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        n_iterations=100  # Use fewer iterations for faster results
    )
    
    # Plot bootstrap distributions
    evaluator.plot_bootstrap_distributions(bootstrap_results)
    
    print("\nEnd-to-end test completed successfully!")
    
    return {
        'optimization_results': optimization_results,
        'evaluation_results': eval_results,
        'bootstrap_results': bootstrap_results
    }

if __name__ == "__main__":
    
    main(r"C:\Users\Akhil PC\Documents\projects\personal\Data Analytics Library\manual_label_sample.csv" , r"C:\Users\Akhil PC\Documents\projects\personal\Data Analytics Library\reference_vector_data_csr.csv")