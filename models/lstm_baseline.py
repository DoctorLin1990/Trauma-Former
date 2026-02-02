"""
LSTM Baseline Model for TIC Prediction
Implementation as described in Section 3.4 (baseline comparison)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class LSTMBaseline(nn.Module):
    """
    LSTM baseline model for comparison with Trauma-Former
    
    Architecture:
    1. Two-layer LSTM with 64 hidden units
    2. Dropout for regularization
    3. Fully connected classification head
    """
    
    def __init__(self, input_dim: int = 4, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2, 
                 bidirectional: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate output dimension
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (improves learning)
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of LSTM baseline
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
        Returns:
            predictions: Tensor [batch_size, 1] with risk probabilities
        """
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                        batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), 
                        batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Use the last hidden state for classification
        if self.bidirectional:
            # Concatenate forward and backward final states
            last_forward = hn[-2, :, :]  # Last forward layer
            last_backward = hn[-1, :, :]  # Last backward layer
            last_hidden = torch.cat((last_forward, last_backward), dim=1)
        else:
            last_hidden = hn[-1, :, :]
        
        # Classification
        predictions = self.classifier(last_hidden).squeeze(-1)
        
        return predictions
    
    def get_model_size(self) -> Dict:
        """Get model size information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory usage
        param_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_size_mb': param_size_mb
        }

class BidirectionalLSTMWithAttention(LSTMBaseline):
    """
    Extended LSTM with attention mechanism for interpretability
    """
    
    def __init__(self, input_dim: int = 4, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__(input_dim, hidden_size, num_layers, dropout, bidirectional=True)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Update classifier for attention output
        lstm_output_dim = hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with attention
        
        Args:
            x: Input tensor
            return_attention: Whether to return attention weights
        Returns:
            predictions and optionally attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size*2]
        
        # Compute attention weights
        attention_weights = self.attention(lstm_out)  # [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # [batch_size, hidden_size*2]
        
        # Classification
        predictions = self.classifier(context_vector).squeeze(-1)
        
        if return_attention:
            return predictions, attention_weights.squeeze(-1)
        else:
            return predictions