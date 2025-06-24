"""Data loading and validation utilities for QualiVec."""

import os
import pandas as pd
from typing import List, Optional, Dict, Any, Union, Tuple


class DataLoader:
    """Handles data loading and validation for QualiVec."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the DataLoader.
        
        Args:
            verbose: Whether to print status messages.
        """
        self.verbose = verbose
    
    def load_corpus(self, filepath: str, sentence_column: str = "sentence") -> pd.DataFrame:
        """Load a corpus from a CSV file.
        
        Args:
            filepath: Path to the CSV file.
            sentence_column: Name of the column containing sentences.
            
        Returns:
            DataFrame containing the corpus.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the sentence column is missing.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Load the data
        if self.verbose:
            print(f"Loading corpus from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        # Validate schema
        if sentence_column not in df.columns:
            raise ValueError(f"Required column '{sentence_column}' not found in the CSV file.")
        
        # Basic validation
        if df[sentence_column].isna().any():
            if self.verbose:
                print(f"Warning: {df[sentence_column].isna().sum()} null values found in '{sentence_column}' column.")
        
        if self.verbose:
            print(f"Loaded {len(df)} rows from {filepath}")
        
        return df
    
    def load_reference_vectors(self, filepath: str, class_column: str = "class", 
                               node_column: str = "matching_node") -> pd.DataFrame:
        """Load reference vectors from a CSV file.
        
        Args:
            filepath: Path to the CSV file.
            class_column: Name of the column containing class labels.
            node_column: Name of the column containing matching nodes.
            
        Returns:
            DataFrame containing the reference vectors.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If required columns are missing.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if self.verbose:
            print(f"Loading reference vectors from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        # Validate schema
        required_columns = [class_column, node_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Required columns {missing_columns} not found in the CSV file.")
        
        # Basic validation
        if df[class_column].isna().any() or df[node_column].isna().any():
            if self.verbose:
                print(f"Warning: Null values found in reference vectors.")
        
        if self.verbose:
            print(f"Loaded {len(df)} reference vectors from {filepath}")
            print(f"Unique classes: {df[class_column].nunique()}")
        
        return df
    
    def load_labeled_data(self, filepath: str, label_column: str = "label") -> pd.DataFrame:
        """Load manually labeled data from a CSV file.
        
        Args:
            filepath: Path to the CSV file.
            label_column: Name of the column containing labels.
            
        Returns:
            DataFrame containing the labeled data.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the label column is missing.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if self.verbose:
            print(f"Loading labeled data from {filepath}...")
        
        df = pd.read_csv(filepath)
        
        # Validate schema
        if label_column not in df.columns:
            raise ValueError(f"Required column '{label_column}' not found in the CSV file.")
        
        # Basic validation
        if df[label_column].isna().any():
            if self.verbose:
                print(f"Warning: {df[label_column].isna().sum()} null values found in '{label_column}' column.")
        
        if self.verbose:
            print(f"Loaded {len(df)} labeled samples from {filepath}")
            print(f"Label distribution:\n{df[label_column].value_counts()}")
        
        return df
    
    def save_dataframe(self, df: pd.DataFrame, filepath: str) -> None:
        """Save a DataFrame to a CSV file.
        
        Args:
            df: DataFrame to save.
            filepath: Path to save the CSV file.
        """
        df.to_csv(filepath, index=False)
        
        if self.verbose:
            print(f"Saved {len(df)} rows to {filepath}")
    
    def validate_labels(self, labeled_df: pd.DataFrame, reference_df: pd.DataFrame, 
                        label_column: str = "label", class_column: str = "class") -> bool:
        """Validate that labels in the labeled data are a subset of those in the reference data.
        
        Args:
            labeled_df: DataFrame containing labeled data.
            reference_df: DataFrame containing reference vectors.
            label_column: Name of the column containing labels in labeled_df.
            class_column: Name of the column containing classes in reference_df.
            
        Returns:
            True if validation passes, False otherwise.
        """
        labeled_classes = set(labeled_df[label_column].unique())
        reference_classes = set(reference_df[class_column].unique())
        
        unknown_classes = labeled_classes - reference_classes
        
        if unknown_classes:
            if self.verbose:
                print(f"Warning: Found {len(unknown_classes)} labels in labeled data that are not in reference vectors:")
                print(unknown_classes)
            return False
        
        if self.verbose:
            print("Label validation passed: All labels in labeled data are in reference vectors.")
        
        return True
