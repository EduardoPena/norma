"""
SLM (Small Language Model) Classifier for rule detection.

Supports two types of models:
1. Classification models (fine-tuned transformers with classification head)
2. Sentence transformers (e.g., mpnet) with optional classifier head

Usage Examples:
---------------

# Example 1: Using a pre-trained sentence transformer (mpnet)
classifier = SLMRuleClassifier(
    model_path="sentence-transformers/all-mpnet-base-v2",
    model_type="sentence-transformer"
)

# Example 2: Using a fine-tuned classification model
classifier = SLMRuleClassifier(
    model_path="models/my_trained_classifier",
    model_type="classification"
)

# Example 3: Using mpnet with a trained classifier head
# First, train and save a classifier head:
#   from sklearn.linear_model import LogisticRegression
#   from sentence_transformers import SentenceTransformer
#   import pickle, os
#
#   # Train
#   model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
#   embeddings = model.encode(train_texts)
#   clf = LogisticRegression().fit(embeddings, train_labels)
#
#   # Save
#   save_dir = "models/mpnet_with_classifier"
#   os.makedirs(save_dir, exist_ok=True)
#   model.save(save_dir)
#   with open(os.path.join(save_dir, "classifier.pkl"), "wb") as f:
#       pickle.dump(clf, f)
#   with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
#       pickle.dump({
#           "best_threshold": 0.5,
#           "dq_rule_idx": 1,
#           "label_encoder": label_encoder
#       }, f)
#
# Then use it:
classifier = SLMRuleClassifier(
    model_path="models/mpnet_with_classifier",
    model_type="sentence-transformer"
)
"""

import os
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

from src.classifiers.base import BaseRuleClassifier




class RuleDataset(Dataset):
    """Dataset for tokenized text sequences."""
    
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }


class SLMRuleClassifier(BaseRuleClassifier):
    """Small Language Model classifier using fine-tuned transformers or sentence transformers."""
    
    def __init__(
        self,
        model_path: str,
        batch_size: int = 8,
        device: str = None,
        chunk_size: Optional[int] = None,
        model_type: str = 'auto'
    ):
        """
        Initialize SLM classifier.
        
        Args:
            model_path: Path to the trained model directory or HuggingFace model name
            batch_size: Batch size for inference
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
            chunk_size: Max sentences per API call (None = process all at once)
            model_type: Type of model - 'classification', 'sentence-transformer', or 'auto'
        """
        super().__init__(chunk_size=chunk_size)
        self.model_path = model_path
        self.batch_size = batch_size
        self.model_type = model_type
        self.use_sentence_transformer = False
        self.classifier_head = None
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model and config
        self._load_model()
    
    def _load_model(self):
        """Load trained model, tokenizer, and configuration."""
        
        # Check if it's a local path
        is_local_path = os.path.exists(self.model_path)
        
        if is_local_path:
            print(f"Loading SLM model from {self.model_path}...")
        else:
            print(f"Loading SLM model '{self.model_path}' from HuggingFace...")
        
        # Auto-detect model type if needed
        if self.model_type == 'auto':
            self.model_type = self._detect_model_type()
        
        # Load model based on type
        if self.model_type == 'sentence-transformer':
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Install it with: pip install sentence-transformers"
                )
            
            print("Loading as sentence transformer model...")
            self.model = SentenceTransformer(self.model_path, device=self.device)
            self.tokenizer = None  # SentenceTransformer handles tokenization internally
            self.use_sentence_transformer = True
            
            # Load classifier head if exists
            if is_local_path:
                classifier_path = os.path.join(self.model_path, 'classifier.pkl')
                if os.path.exists(classifier_path):
                    with open(classifier_path, 'rb') as f:
                        self.classifier_head = pickle.load(f)
                    print("Loaded classifier head")
        else:
            print("Loading as classification model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            self.use_sentence_transformer = False
        
        # Load config (label encoder, threshold, etc.)
        config_path = os.path.join(self.model_path, 'config.pkl') if is_local_path else None
        if config_path and os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                model_config = pickle.load(f)
            
            self.label_encoder = model_config.get('label_encoder')
            self.best_threshold = model_config.get('best_threshold', 0.5)
            self.dq_rule_idx = model_config.get('dq_rule_idx', 0)
            self.non_dq_rule_idx = 1 - self.dq_rule_idx
            
            print(f"Model loaded successfully!")
            print(f"Device: {self.device}")
            print(f"Best threshold: {self.best_threshold:.3f}")
            if self.label_encoder:
                print(f"Classes: {self.label_encoder.classes_}")
        else:
            print("Warning: config.pkl not found, using default threshold=0.5")
            self.label_encoder = None
            self.best_threshold = 0.5
            self.dq_rule_idx = 0
            self.non_dq_rule_idx = 1
    
    def _detect_model_type(self) -> str:
        """Auto-detect if model is a sentence transformer or classification model."""
        is_local_path = os.path.exists(self.model_path)
        
        # Check for sentence-transformer indicators
        if is_local_path:
            # Check for sentence-transformers config files
            if os.path.exists(os.path.join(self.model_path, 'modules.json')):
                return 'sentence-transformer'
            # Check for classifier head
            if os.path.exists(os.path.join(self.model_path, 'classifier.pkl')):
                return 'sentence-transformer'
        else:
            # Check model name patterns
            if any(name in self.model_path.lower() for name in 
                   ['sentence-transformers', 'mpnet', 'all-minilm', 'all-mpnet']):
                return 'sentence-transformer'
        
        return 'classification'
    
    def _classify_sentences(
        self,
        sentences: List[str],
        context: Dict[str, Any]
    ) -> List[bool]:
        """
        Implementation-specific classification logic.
        
        Args:
            sentences: List of sentences to classify
            context: Context dictionary (not used by SLM)
        
        Returns:
            List of boolean predictions (True = is a dq-rule)
        """
        if not sentences:
            return []
        
        # Get probabilities
        probs = self._predict_proba(sentences)
        dq_probs = probs[:, self.dq_rule_idx]
        
        # Apply threshold
        predictions = (dq_probs >= self.best_threshold).tolist()
        
        return predictions
    
    def classify_with_probabilities(
        self,
        sentences: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[List[bool], List[float]]:
        """
        Classify sentences and return probabilities.
        
        Args:
            sentences: List of sentences to classify
            context: Optional context (not used by SLM)
        
        Returns:
            Tuple of (predictions, probabilities) where probabilities are for dq-rule class
        """
        if not sentences:
            return [], []
        
        # Get probabilities
        probs = self._predict_proba(sentences)
        dq_probs = probs[:, self.dq_rule_idx]
        
        # Apply threshold
        predictions = (dq_probs >= self.best_threshold).tolist()
        
        return predictions, dq_probs.tolist()
    
    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Get probability predictions for texts.
        
        Args:
            texts: List of texts to classify
        
        Returns:
            NumPy array of shape (n_samples, n_classes) with probabilities
        """
        if self.use_sentence_transformer:
            return self._predict_proba_sentence_transformer(texts)
        else:
            return self._predict_proba_classification(texts)
    
    def _predict_proba_classification(self, texts: List[str]) -> np.ndarray:
        """Get predictions using classification model."""
        dataset = RuleDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_logits = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(outputs.logits.cpu())
        
        logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(logits, dim=-1).numpy()
        
        return probs
    
    def _predict_proba_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Get predictions using sentence transformer + classifier head."""
        # Get embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        if self.classifier_head is not None:
            # Use trained classifier head
            try:
                probs = self.classifier_head.predict_proba(embeddings)
                return probs
            except AttributeError:
                # If classifier doesn't have predict_proba, use decision function
                scores = self.classifier_head.decision_function(embeddings)
                # Convert to probabilities using sigmoid
                if len(scores.shape) == 1:
                    # Binary classification
                    pos_probs = 1 / (1 + np.exp(-scores))
                    probs = np.column_stack([1 - pos_probs, pos_probs])
                else:
                    # Multi-class (use softmax)
                    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                return probs
        else:
            # No classifier head - use simple cosine similarity or return dummy probs
            print("Warning: No classifier head found, returning neutral probabilities")
            return np.full((len(texts), 2), 0.5)

    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Feature importance not applicable for transformer models."""
        return {
            "method": "SLM (transformer)",
            "note": "Feature importance not available for neural models"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            "classifier_type": "SLM (Small Language Model)",
            "model_path": self.model_path,
            "device": self.device,
            "threshold": self.best_threshold,
            "batch_size": self.batch_size,
            "use_sentence_transformer": self.use_sentence_transformer,
        }
        
        if self.use_sentence_transformer:
            info["model_type"] = "SentenceTransformer"
            info["has_classifier_head"] = self.classifier_head is not None
            if self.classifier_head is not None:
                info["classifier_head_type"] = type(self.classifier_head).__name__
        else:
            info["model_type"] = self.model.__class__.__name__
            info["num_parameters"] = sum(p.numel() for p in self.model.parameters())
        
        return info


def create_slm_classifier(model_path: str, **kwargs) -> SLMRuleClassifier:
    """
    Factory function to create an SLM classifier.
    
    Args:
        model_path: Path to trained model directory
        **kwargs: Additional arguments for SLMRuleClassifier
    
    Returns:
        Initialized SLMRuleClassifier instance
    """
    return SLMRuleClassifier(model_path=model_path, **kwargs)
