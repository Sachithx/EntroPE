import numpy as np
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
import os
import argparse
# from TLLM_data_provider.data_factory import data_provider
from data_provider.data_factory import data_provider
from chronos import MeanScaleUniformBins, ChronosConfig
from pathlib import Path

device = 'cuda'
PAD_ID = -1  

# WandB Configuration
WANDB_PROJECT = "bytelatent-transformer"
WANDB_ENTITY = None  # Set to your wandb username/team if needed
ENABLE_WANDB = False  # Set to False to disable wandb logging

import argparse  # Make sure this is imported

# def build_dataloader(dataset_name, features, seq_len, label_len, pred_len, flag, batch_size, pretrain):
def build_dataloader(dataset_name, data, features, seq_len, label_len, pred_len, flag, batch_size, embed='timeF'):
    args = argparse.Namespace(
        data="custom",
        root_path='dataset/',
        data_path=f'{dataset_name}.csv',
        features=features,
        target='OT',  # target column, typically "OT" in ETT datasets
        freq='h' if 'h' in dataset_name else 't',
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        embed=embed,
        batch_size=batch_size,
        num_workers=2
    )

    dataset, loader = data_provider(args, flag=flag)

    print(f"[INFO] {flag} set: {len(dataset)} samples, {len(loader)} batches")
    return dataset, loader


def build_tokenizer(quant_range, vocab_size, context_length, prediction_length):
    """
    Build a tokenizer that maps byte values to tokens.
    """
    # Create a new config with prediction_length=1
    low_limit = -1 * quant_range
    high_limit = quant_range

    tokenizer_config = ChronosConfig(
        tokenizer_class='MeanScaleUniformBins',
        tokenizer_kwargs={'low_limit': low_limit, 'high_limit': high_limit},
        context_length=context_length,
        prediction_length=prediction_length,   
        n_tokens=vocab_size,
        n_special_tokens=4,
        pad_token_id=-1,
        eos_token_id=0,
        use_eos_token=False,
        model_type='causal',
        num_samples=1,
        temperature=1.0,
        top_k=50,
        top_p=1.0
    )

    # Create a new tokenizer with the updated config
    tokenizer = MeanScaleUniformBins(low_limit, high_limit, tokenizer_config)
    return tokenizer

def get_lr(iteration, max_iters, warmup_iters, learning_rate, min_lr, decay_lr=True):
    """
    Learning rate scheduler function
    Args:
        iteration: Current iteration
        max_iters: Total number of iterations
        warmup_iters: Number of warmup iterations
        learning_rate: Peak learning rate
        min_lr: Minimum learning rate
        decay_lr: Whether to decay learning rate after warmup
    Returns:
        lr: Learning rate for current iteration
    """
    # Linear warmup
    if iteration < warmup_iters:
        return learning_rate * (iteration / warmup_iters)
    
    # Decay phase (if enabled)
    if decay_lr:
        # Cosine decay from learning_rate to min_lr
        decay_ratio = (iteration - warmup_iters) / (max_iters - warmup_iters)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # Cosine decay
        return min_lr + coeff * (learning_rate - min_lr)
    else:
        # Constant learning rate after warmup
        return learning_rate
    

def create_static_patch_lengths(batch_size, seq_len, patch_length=8):
    if seq_len == 336:
        n_patches = 43
    elif seq_len == 96:
        n_patches = 13
    elif seq_len == 512:
        n_patches = 65
    elif seq_len == 256:
        n_patches = 33
    elif seq_len == 192:
        n_patches = 13
    elif seq_len == 720:
        n_patches = 91
    else:
        raise ValueError(f"Unsupported seq_len: {seq_len}")

    l = torch.full((batch_size,n_patches), patch_length).to('cuda')
    l[:,0] = 1
    l[:,1] = patch_length - 1
    patch_lengths = l
    return patch_lengths



def validate(model, data_loader, tokenizer, patch_lengths, device, desc="Validation"):
    """Run validation or test on a data loader"""
    model.eval()
    best_val_loss = float('inf')
    total_loss = 0
    all_preds = []
    all_targets = []
    running_loss = 0.0
    
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=desc, position=0, leave=True)
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in progress_bar:
            x = batch_x.float().squeeze(-1)
            y = batch_y.float().squeeze(-1)
            
            # Tokenize input sequence
            token_ids, attention_mask, tokenizer_state = tokenizer.context_input_transform(x)
            
            # Tokenize target sequence
            target_token_ids, target_attention_mask = tokenizer.label_input_transform(y, tokenizer_state)
            
            # Forward pass
            logits = model(token_ids.to(device), patch_lengths)
            
            # Calculate loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_token_ids.reshape(-1).to(device),
                ignore_index=PAD_ID
            )
            
            # Update loss statistics
            current_loss = loss.item()
            total_loss += current_loss
            running_loss = total_loss / (i + 1)
            
            # For RMSE calculation
            # Get predicted tokens
            pred_tokens = torch.argmax(logits, dim=-1)
            
            # Add sample dimension for output_transform
            pred_tokens = pred_tokens.unsqueeze(1)  # [batch_size, 1, prediction_length]
            
            # Convert tokens to values
            pred_values = tokenizer.output_transform(pred_tokens.to('cpu'), tokenizer_state)
            
            # Remove sample dimension for comparison
            pred_values = pred_values.squeeze(1)  # [batch_size, prediction_length]
            
            # Store for RMSE calculation
            all_preds.append(pred_values.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'avg_loss': f"{running_loss:.4f}"
            })
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    
    # Calculate RMSE
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse

def improved_loss_with_gradients(logits, target_token_ids, valid_mask, vocab_size=4096):
    # CE loss (same as before)
    ce_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_token_ids.reshape(-1),
        ignore_index=PAD_ID
    )
    
    # MSE loss WITH GRADIENTS (the key improvement!)
    token_values = torch.linspace(-15.0, 15.0, vocab_size, device=logits.device)
    
    # Soft predictions (preserves gradients!)
    probs = F.softmax(logits, dim=-1)
    predicted_values = torch.sum(probs * token_values.view(1, 1, -1), dim=-1)
    
    # Target values  
    target_values = -15.0 + (target_token_ids.float() / (vocab_size - 1)) * 30.0
    
    # MSE loss (now with gradients!)
    mse_loss = F.mse_loss(predicted_values[valid_mask], target_values[valid_mask])
    
    # Combined loss (both contribute gradients!)
    total_loss = ce_loss + mse_loss  # Or weight them: ce_loss + 0.5 * mse_loss
    
    return total_loss, ce_loss, mse_loss

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()

# Create output directory
Path("output").mkdir(parents=True, exist_ok=True)

class TrainingLogger:
    """Logger for training metrics with WandB integration"""
    def __init__(self, output_dir, dataset_name, enable_wandb=False):
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        self.enable_wandb = enable_wandb
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'eval_results': []
        }
        
    def log_epoch(self, epoch, train_loss, val_loss, lr, eval_results=None, train_time=None, val_time=None):
        self.history['epoch'].append(int(epoch))
        self.history['train_loss'].append(float(train_loss))
        self.history['val_loss'].append(float(val_loss))
        self.history['learning_rate'].append(float(lr))
        
        # Prepare WandB logging data
        wandb_log = {
            'epoch': epoch,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'train/learning_rate': lr,
        }
        
        # Add timing information if available
        if train_time is not None:
            wandb_log['train/time_per_epoch'] = train_time
        if val_time is not None:
            wandb_log['val/time_per_epoch'] = val_time
        
        if eval_results:
            # Convert eval_results to JSON-serializable format
            serializable_results = self._make_json_serializable(eval_results)
            self.history['eval_results'].append({
                'epoch': int(epoch),
                'results': serializable_results
            })
            
            # Log evaluation results to wandb
            if self.enable_wandb and eval_results:
                eval_wandb_log = self._flatten_eval_results(eval_results, prefix='eval/')
                wandb_log.update(eval_wandb_log)
        
        # Log to WandB
        if self.enable_wandb:
            wandb.log(wandb_log)
        
        # Save to file
        with open(self.output_dir / f'training_history_{self.dataset_name}_{features}.json', 'w') as f:
            json.dump(self.history, f, indent=4)
    
    def _flatten_eval_results(self, results, prefix=''):
        """Flatten nested evaluation results for WandB logging"""
        flattened = {}
        
        def _flatten_dict(d, parent_key=''):
            for k, v in d.items():
                new_key = f"{parent_key}{k}" if parent_key else k
                if isinstance(v, dict):
                    _flatten_dict(v, f"{new_key}/")
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    flattened[f"{prefix}{new_key}"] = float(v)
                elif hasattr(v, 'item'):  # PyTorch tensors
                    flattened[f"{prefix}{new_key}"] = float(v.item())
        
        if isinstance(results, dict):
            _flatten_dict(results)
        
        return flattened
    
    def _make_json_serializable(self, obj):
        """Convert numpy/torch types to JSON serializable Python types"""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # PyTorch tensors
            return float(obj.item())
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)  # Fallback to string representation
    
    def print_summary(self):
        """Print training summary"""
        print(f"\n{'='*60}")
        print(f"📊 TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Total epochs: {len(self.history['epoch'])}")
        print(f"Best validation loss: {min(self.history['val_loss']):.6f}")
        print(f"Final validation loss: {self.history['val_loss'][-1]:.6f}")
        if self.history['eval_results']:
            print(f"Evaluations performed: {len(self.history['eval_results'])}")

def init_wandb(config_dict, project_name=WANDB_PROJECT, entity=WANDB_ENTITY, run_name=None):
    """Initialize WandB run with configuration"""
    if not ENABLE_WANDB:
        return None
    
    if run_name is None:
        run_name = f"{config_dict['dataset_name']}_{config_dict['features']}_{config_dict['seq_len']}"
    
    wandb.init(
        project=project_name,
        entity=entity,
        name=run_name,
        config=config_dict,
        tags=[config_dict['dataset_name'], config_dict['features'], 'bytelatent'],
        save_code=True
    )
    
    # Log model architecture as text
    if wandb.run:
        wandb.run.summary['model_params_millions'] = config_dict['model_params'] / 1e6
        # You can also log the model architecture as an artifact
        # model_artifact = wandb.Artifact('model_architecture', type='model')
        # wandb.log_artifact(model_artifact)
    
    return wandb.run