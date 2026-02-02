#!/usr/bin/env python3
"""
Trauma-Former Training Script
Main script for training the Trauma-Former model as described in Section 3.4
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import json
from datetime import datetime
from tqdm import tqdm

# Local imports
from data.synthetic_data_generator import SyntheticTraumaDataset
from models.trauma_former import TraumaFormer
from models.lstm_baseline import LSTMBaseline
from utils.metrics import compute_metrics, plot_roc_curve
from utils.data_loader import DataLoaderWrapper

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (data, labels) in enumerate(tqdm(dataloader, desc="Training")):
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs.squeeze(), labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc="Validation"):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs.squeeze(), labels)
            
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_preds), np.array(all_labels)

def main(config_path):
    """Main training function"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Starting Trauma-Former training with config: {config_path}")
    print(f"Project: {config['project']['name']} v{config['project']['version']}")
    
    # Set random seeds
    set_seed(config['reproducibility']['seed'])
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and config['system']['cloud']['gpu_acceleration'] else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(config['paths']['results_dir'], f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(experiment_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    
    # Generate or load synthetic data
    print("\n" + "="*50)
    print("Data Preparation")
    print("="*50)
    
    if os.path.exists(os.path.join(config['paths']['data_dir'], "synthetic", "dataset.npz")):
        print("Loading existing synthetic dataset...")
        data = np.load(os.path.join(config['paths']['data_dir'], "synthetic", "dataset.npz"))
        X = data['X']
        y = data['y']
    else:
        print("Generating synthetic dataset...")
        dataset = SyntheticTraumaDataset(
            num_samples=config['data_generation']['num_samples'],
            seq_length=config['data_generation']['sequence_length'],
            num_features=config['data_generation']['num_features']
        )
        X, y = dataset.generate()
        
        # Save dataset
        os.makedirs(os.path.join(config['paths']['data_dir'], "synthetic"), exist_ok=True)
        np.savez(os.path.join(config['paths']['data_dir'], "synthetic", "dataset.npz"), X=X, y=y)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: TIC Positive = {y.sum()}, TIC Negative = {len(y) - y.sum()}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=config['training']['validation_split'] + config['training']['test_split'],
        random_state=config['reproducibility']['seed'],
        stratify=y
    )
    
    val_size = config['training']['validation_split'] / (config['training']['validation_split'] + config['training']['test_split'])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_size,
        random_state=config['reproducibility']['seed'],
        stratify=y_temp
    )
    
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Create dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    # Initialize model
    print("\n" + "="*50)
    print("Model Initialization")
    print("="*50)
    
    if config['model']['name'] == "Trauma-Former":
        model = TraumaFormer(
            input_dim=config['data_generation']['num_features'],
            d_model=config['model']['transformer']['d_model'],
            nhead=config['model']['transformer']['nhead'],
            num_layers=config['model']['transformer']['num_encoder_layers'],
            dim_feedforward=config['model']['transformer']['dim_feedforward'],
            dropout=config['model']['transformer']['dropout']
        )
    elif config['model']['name'] == "LSTM":
        model = LSTMBaseline(
            input_dim=config['data_generation']['num_features'],
            hidden_size=config['model']['lstm_baseline']['hidden_size'],
            num_layers=config['model']['lstm_baseline']['num_layers'],
            dropout=config['model']['lstm_baseline']['dropout']
        )
    else:
        raise ValueError(f"Unknown model: {config['model']['name']}")
    
    model = model.to(device)
    print(f"Model: {config['model']['name']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        betas=tuple(config['training']['optimizer']['betas'])
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['scheduler']['factor'],
        patience=config['training']['scheduler']['patience'],
        verbose=True
    )
    
    # Training loop
    print("\n" + "="*50)
    print("Training Started")
    print("="*50)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_auroc': [],
        'val_auroc': [],
        'learning_rate': []
    }
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_preds, train_labels = train_epoch(model, train_loader, criterion, optimizer, device)
        train_metrics = compute_metrics(train_labels, train_preds)
        
        # Validate
        val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device)
        val_metrics = compute_metrics(val_labels, val_preds)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auroc'].append(train_metrics['auroc'])
        history['val_auroc'].append(val_metrics['auroc'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        print(f"Train Loss: {train_loss:.4f}, Train AUROC: {train_metrics['auroc']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val AUROC: {val_metrics['auroc']:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping and model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_auroc': val_metrics['auroc'],
                'config': config
            }, os.path.join(experiment_dir, "best_model.pth"))
            
            print(f"Best model saved with Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model for final evaluation
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    checkpoint = torch.load(os.path.join(experiment_dir, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_loss, test_preds, test_labels = validate(model, test_loader, criterion, device)
    test_metrics = compute_metrics(test_labels, test_preds)
    
    print(f"Test Loss: {test_loss:.4f}")
    for metric, value in test_metrics.items():
        print(f"Test {metric.upper()}: {value:.4f}")
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'history': history,
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': checkpoint['val_loss'],
        'best_val_auroc': checkpoint['val_auroc']
    }
    
    with open(os.path.join(experiment_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate ROC curve
    plot_roc_curve(
        test_labels, test_preds,
        save_path=os.path.join(experiment_dir, "roc_curve.png"),
        title=f"ROC Curve - {config['model']['name']} (AUROC: {test_metrics['auroc']:.3f})"
    )
    
    print(f"\nExperiment completed. Results saved to: {experiment_dir}")
    print(f"Final Test AUROC: {test_metrics['auroc']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Trauma-Former model")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, 
                       help="Override data directory path")
    
    args = parser.parse_args()
    
    # Override config if needed
    if args.data_dir:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        config['paths']['data_dir'] = args.data_dir
        import tempfile
        import os
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config, temp_config)
        temp_config.close()
        args.config = temp_config.name
    
    try:
        main(args.config)
    finally:
        if args.data_dir and os.path.exists(args.config):
            os.unlink(args.config)