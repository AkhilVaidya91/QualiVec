"""Embedding utilities for QualiVec."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import os


class EmbeddingModel:
    """Handles text embedding for QualiVec."""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 verbose: bool = True):
        """Initialize the embedding model.
        
        Args:
            model_name: Name of the HuggingFace model to use.
            device: Device to use for computation ('cpu' or 'cuda').
            cache_dir: Directory to cache models.
            verbose: Whether to print status messages.
        """
        self.model_name = model_name
        self.verbose = verbose
        self.cache_dir = cache_dir
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if self.verbose:
            print(f"Using device: {self.device}")
            print(f"Loading model: {model_name}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        
        if self.verbose:
            print(f"Model loaded successfully")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling operation to get sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed.
            batch_size: Batch size for processing.
            
        Returns:
            Numpy array of embeddings.
        """
        if self.verbose:
            print(f"Generating embeddings for {len(texts)} texts")
        
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), disable=not self.verbose):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                          max_length=512, return_tensors='pt').to(self.device)
            
            # Get model output
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Mean pooling
            batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            # Add to list
            embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)
        
        if self.verbose:
            print(f"Generated embeddings with shape: {all_embeddings.shape}")
        
        return all_embeddings
    
    def embed_dataframe(self, 
                        df: pd.DataFrame, 
                        text_column: str,
                        batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for texts in a DataFrame column.
        
        Args:
            df: DataFrame containing texts.
            text_column: Name of the column containing texts.
            batch_size: Batch size for processing.
            
        Returns:
            Numpy array of embeddings.
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")
        
        texts = df[text_column].fillna("").tolist()
        return self.embed_texts(texts, batch_size)
    
    def embed_reference_vectors(self, 
                               df: pd.DataFrame, 
                               class_column: str = "class",
                               node_column: str = "matching_node",
                               batch_size: int = 32) -> Dict[str, Any]:
        """Generate embeddings for reference vectors.
        
        Args:
            df: DataFrame containing reference vectors.
            class_column: Name of the column containing class labels.
            node_column: Name of the column containing matching nodes.
            batch_size: Batch size for processing.
            
        Returns:
            Dictionary with class info and embeddings.
        """
        required_columns = [class_column, node_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Required columns {missing_columns} not found in DataFrame.")
        
        # Get texts and generate embeddings
        texts = df[node_column].fillna("").tolist()
        embeddings = self.embed_texts(texts, batch_size)
        
        # Create result dictionary
        result = {
            "classes": df[class_column].tolist(),
            "nodes": df[node_column].tolist(),
            "embeddings": embeddings,
            "class_to_idx": {cls: i for i, cls in enumerate(df[class_column].unique())}
        }
        
        if self.verbose:
            print(f"Generated embeddings for {len(result['classes'])} reference vectors")
            print(f"Unique classes: {len(result['class_to_idx'])}")
        
        return result
