"""
Trauma-Former: Time-Series Transformer for TIC Prediction
Implementation as described in Section 3.3 of the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, List

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time-series data"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                           (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return x

class MultiHeadAttentionWithExplanation(nn.Module):
    """Multi-head attention with attention weight extraction for interpretability"""
    
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query, key, value: Input tensors [batch_size, seq_len, d_model]
            mask: Optional mask tensor
        Returns:
            Output tensor and attention weights
        """
        batch_size = query.size(0)
        
        # Linear projections
        query = self.q_linear(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        key = self.k_linear(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        value = self.v_linear(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Store attention weights for later analysis
        self.attention_weights = attention_weights.detach().cpu()
        
        output = torch.matmul(attention_weights, value)
        
        # Concatenate heads and apply output linear layer
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.out_linear(output)
        
        return output, attention_weights
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get the latest attention weights for interpretability"""
        return self.attention_weights

class TransformerEncoderLayerWithAttention(nn.Module):
    """Transformer encoder layer with attention weight tracking"""
    
    def __init__(self, d_model: int, nhead: int = 4, 
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttentionWithExplanation(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
    
    def forward(self, src: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: Input tensor [batch_size, seq_len, d_model]
            src_mask: Optional mask tensor
        Returns:
            Encoded tensor
        """
        # Self-attention
        src2, attention_weights = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src, attention_weights

class TraumaFormer(nn.Module):
    """
    Trauma-Former: Time-Series Transformer for Trauma-Induced Coagulopathy Prediction
    
    Architecture:
    1. Input embedding layer
    2. Sinusoidal positional encoding
    3. Two Transformer encoder layers with multi-head attention
    4. Global average pooling
    5. Classification head with sigmoid activation
    
    As described in Section 3.3 of the paper.
    """
    
    def __init__(self, input_dim: int = 4, d_model: int = 128, 
                 nhead: int = 4, num_layers: int = 2, 
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        
        # Store configuration
        self.config = {
            'input_dim': input_dim,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout
        }
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(d_model)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithAttention(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
        # Store attention weights for interpretability
        self.attention_weights_history = []
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, 
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Forward pass of Trauma-Former
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            return_attention: Whether to return attention weights
        Returns:
            predictions: Tensor [batch_size, 1] with risk probabilities
            attention_weights: Optional list of attention weights from each layer
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Input embedding
        x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Store attention weights if requested
        attention_weights_list = []
        
        # Transformer encoder layers
        for i, layer in enumerate(self.encoder_layers):
            x, attention_weights = layer(x)
            attention_weights_list.append(attention_weights)
        
        # Layer normalization
        x = self.norm(x)
        
        # Global average pooling over time dimension
        # This aggregates temporal information into a single vector per sample
        x = torch.mean(x, dim=1)  # [batch_size, d_model]
        
        # Classification
        predictions = self.classifier(x).squeeze(-1)  # [batch_size]
        
        if return_attention:
            # Stack attention weights
            # attention_weights_list: list of [batch_size, nhead, seq_len, seq_len]
            stacked_attention = torch.stack(attention_weights_list, dim=1)  # [batch_size, num_layers, nhead, seq_len, seq_len]
            
            # Store for later analysis
            self.attention_weights_history.append(stacked_attention.detach().cpu())
            
            return predictions, attention_weights_list
        else:
            return predictions
    
    def get_attention_maps(self, layer_idx: int = -1) -> Optional[torch.Tensor]:
        """
        Get attention maps from a specific layer
        
        Args:
            layer_idx: Layer index (default: -1 for last layer)
        Returns:
            Attention weights tensor or None
        """
        if not self.attention_weights_history:
            return None
        
        latest_attention = self.attention_weights_history[-1]
        if layer_idx < 0:
            layer_idx = len(self.encoder_layers) + layer_idx
        
        if 0 <= layer_idx < len(self.encoder_layers):
            return latest_attention[:, layer_idx, ...]
        else:
            return None
    
    def compute_flops(self, seq_len: int = 30) -> Dict:
        """
        Compute FLOPs for the model (as referenced in Section 3.3.2)
        
        Returns:
            Dictionary with FLOPs breakdown
        """
        # Self-attention FLOPs: O(T^2 * d)
        attention_flops = seq_len * seq_len * self.config['d_model']
        
        # Feed-forward FLOPs: O(T * d * d_ff)
        ff_flops = seq_len * self.config['d_model'] * self.config['dim_feedforward']
        
        # Per layer FLOPs
        layer_flops = 2 * attention_flops + 2 * ff_flops  # QKV projections + attention + FFN
        
        # Total FLOPs
        total_flops = self.config['num_layers'] * layer_flops
        
        # Classification head FLOPs
        classifier_flops = (
            self.config['d_model'] * 64 +  # First linear
            64 * 32 +                      # Second linear
            32 * 1                         # Third linear
        )
        
        total_flops += classifier_flops
        
        # Convert to MFLOPs
        total_mflops = total_flops / 1e6
        
        return {
            'attention_flops_per_layer': attention_flops,
            'ff_flops_per_layer': ff_flops,
            'total_layer_flops': layer_flops,
            'classifier_flops': classifier_flops,
            'total_flops': total_flops,
            'total_mflops': total_mflops,
            'expected_inference_latency_ms': 15.2  # From paper Table 2
        }
    
    def get_model_size(self) -> Dict:
        """Get model size information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory usage
        param_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_size_mb': param_size_mb,
            'model_size_mb': param_size_mb * 1.2  # Add buffer for optimizer states
        }
    
    @classmethod
    def from_config(cls, config: Dict):
        """Create model from configuration dictionary"""
        return cls(
            input_dim=config.get('input_dim', 4),
            d_model=config.get('d_model', 128),
            nhead=config.get('nhead', 4),
            num_layers=config.get('num_layers', 2),
            dim_feedforward=config.get('dim_feedforward', 512),
            dropout=config.get('dropout', 0.1)
        )

class TraumaFormerWithUncertainty(TraumaFormer):
    """
    Extended Trauma-Former with uncertainty estimation
    Uses Monte Carlo dropout for uncertainty quantification
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_p = kwargs.get('dropout', 0.1)
        
        # Enable dropout during inference for uncertainty estimation
        self.set_dropout_train()
    
    def set_dropout_train(self):
        """Set dropout layers to training mode for MC dropout"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                                n_samples: int = 100) -> Dict:
        """
        Make predictions with uncertainty estimation using MC dropout
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
        Returns:
            Dictionary with predictions and uncertainty metrics
        """
        # Enable dropout
        self.set_dropout_train()
        
        # Collect predictions
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, return_attention=False)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)  # [n_samples, batch_size]
        
        # Compute statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Compute confidence intervals
        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'all_samples': predictions
        }

# Example usage and testing
if __name__ == "__main__":
    # Create a sample model
    model = TraumaFormer(
        input_dim=4,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1
    )
    
    # Print model summary
    print("Trauma-Former Model Summary")
    print("="*50)
    print(f"Input dimension: {model.config['input_dim']}")
    print(f"Model dimension (d_model): {model.config['d_model']}")
    print(f"Number of attention heads: {model.config['nhead']}")
    print(f"Number of encoder layers: {model.config['num_layers']}")
    print(f"Feed-forward dimension: {model.config['dim_feedforward']}")
    print(f"Dropout rate: {model.config['dropout']}")
    print()
    
    # Compute FLOPs
    flops_info = model.compute_flops(seq_len=30)
    print("Computational Requirements")
    print("="*50)
    print(f"Total FLOPs per inference: {flops_info['total_flops']:,.0f}")
    print(f"Total MFLOPs: {flops_info['total_mflops']:.2f}")
    print(f"Expected inference latency: {flops_info['expected_inference_latency_ms']} ms")
    print()
    
    # Model size
    size_info = model.get_model_size()
    print("Model Size")
    print("="*50)
    print(f"Total parameters: {size_info['total_parameters']:,}")
    print(f"Parameter size: {size_info['parameter_size_mb']:.2f} MB")
    print(f"Estimated memory usage: {size_info['model_size_mb']:.2f} MB")
    print()
    
    # Test forward pass
    batch_size = 2
    seq_len = 30
    input_dim = 4
    
    test_input = torch.randn(batch_size, seq_len, input_dim)
    
    print("Testing forward pass...")
    predictions, attention_weights = model(test_input, return_attention=True)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Number of attention weight tensors: {len(attention_weights)}")
    print(f"Attention weight shape (first layer): {attention_weights[0].shape}")
    
    print("\nModel test completed successfully!")