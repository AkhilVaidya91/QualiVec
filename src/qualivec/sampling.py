"""Sampling utilities for QualiVec."""

import pandas as pd
import numpy as np
from typing import Optional, Union, Literal


class Sampler:
    """Handles sampling mechanisms for QualiVec."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the Sampler.
        
        Args:
            verbose: Whether to print status messages.
        """
        self.verbose = verbose
    
    def sample(self, 
               df: pd.DataFrame, 
               sampling_type: Literal["random", "stratified"] = "random", 
               sample_size: Union[int, float] = 0.1, 
               stratify_column: Optional[str] = None, 
               seed: Optional[int] = None,
               label_column: str = "Label") -> pd.DataFrame:
        """Sample data from a DataFrame.
        
        Args:
            df: DataFrame to sample from.
            sampling_type: Type of sampling ("random" or "stratified").
            sample_size: Size of the sample. If float, interpreted as a fraction.
            stratify_column: Column to stratify by (required for stratified sampling).
            seed: Random seed for reproducibility.
            label_column: Name of the label column to add to the output.
            
        Returns:
            DataFrame containing the sampled data.
            
        Raises:
            ValueError: If parameters are invalid.
        """
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Calculate sample size if given as a fraction
        if isinstance(sample_size, float):
            if not 0 < sample_size <= 1:
                raise ValueError("Sample size as fraction must be between 0 and 1.")
            n_samples = int(len(df) * sample_size)
        else:
            if not 0 < sample_size <= len(df):
                raise ValueError(f"Sample size must be between 1 and {len(df)}.")
            n_samples = sample_size
        
        if self.verbose:
            print(f"Sampling {n_samples} rows ({n_samples/len(df):.1%} of data)...")
        
        # Perform sampling
        if sampling_type == "random":
            sample = df.sample(n=n_samples, random_state=seed)
            
        elif sampling_type == "stratified":
            if stratify_column is None:
                raise ValueError("stratify_column must be provided for stratified sampling.")
                
            if stratify_column not in df.columns:
                raise ValueError(f"Stratification column '{stratify_column}' not found in DataFrame.")
            
            # Check for NaN values in stratification column
            if df[stratify_column].isna().any():
                raise ValueError(f"NaN values found in stratification column '{stratify_column}'.")
            
            # Calculate the proportion for each stratum
            strata = df[stratify_column].value_counts(normalize=True)
            
            # Create empty sample DataFrame
            sample = pd.DataFrame(columns=df.columns)
            
            # Sample from each stratum
            for stratum, proportion in strata.items():
                stratum_df = df[df[stratify_column] == stratum]
                stratum_samples = max(1, int(n_samples * proportion))
                stratum_sample = stratum_df.sample(n=min(stratum_samples, len(stratum_df)), 
                                                 random_state=seed)
                sample = pd.concat([sample, stratum_sample])
            
            if self.verbose:
                print(f"Stratified sampling based on '{stratify_column}':")
                for stratum, count in sample[stratify_column].value_counts().items():
                    print(f"  - {stratum}: {count} samples ({count/n_samples:.1%})")
        else:
            raise ValueError(f"Unknown sampling type: {sampling_type}")
        
        # Add empty label column for manual annotation
        if label_column not in sample.columns:
            sample[label_column] = None
        
        if self.verbose:
            print(f"Created sample with {len(sample)} rows.")
        
        return sample
