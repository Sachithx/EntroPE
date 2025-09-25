import os
import time
from contextlib import nullcontext
from pathlib import Path
import itertools
import json

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb

from model import GPTConfig, GPT
from utils.train_utils import build_dataloader, build_tokenizer, get_lr


# ============================================================================
# HYPERPARAMETER SWEEP CONFIGURATION
# ============================================================================

class SweepConfig:
    """Hyperparameter sweep configuration"""
    
    # Fixed Hardware Configuration
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    
    # Fixed Data Configuration
    dataset_name = 'ETTm2'
    features = 'M'
    batch_size = 128
    seq_len = 672
    block_size = seq_len
    
    # Fixed Training Hyperparameters
    learning_rate = 1e-3
    weight_decay = 0.05
    beta1 = 0.9
    beta2 = 0.95
    epochs = 50  # Reduced for sweep
    grad_accumulation_steps = 1
    clip_grad = 1.0
    dropout = 0.1
    bias = False
    
    # Fixed Learning Rate Schedule
    warmup_steps = 0
    min_lr_factor = 0.1
    decay_lr = True
    
    # Fixed Training Control
    patience = 15  # Early stopping patience
    save_every = 10
    seed = 42
    compile = True
    
    # Output
    output_dir = "sweep_results"
    
    # W&B Configuration
    wandb_project = "Entropy Model Sweep"
    wandb_entity = None
    wandb_tags = ["hyperparameter-sweep", "time-series", "transformer"]
    wandb_notes = "Systematic hyperparameter sweep for optimal configuration"
    wandb_log_freq = 10
    wandb_save_model = False  # Disabled for sweep to save space


# Define hyperparameter search space
HYPERPARAMETER_GRID = {
    # Model Architecture Parameters
    'n_layer': [2, 3, 4, 6],                    # Number of transformer layers
    'n_head': [2, 4, 6, 8],                     # Number of attention heads
    'n_embd': [32, 64, 96, 128],                # Embedding dimension
    'quant_range': [4, 8, 16],                  # Quantization range
    'vocab_size': [256, 512, 1024, 2048],       # Vocabulary size
}

# Alternative: Predefined "smart" combinations to reduce search space
SMART_COMBINATIONS = [
    # # Small models
    {'n_layer': 2, 'n_head': 2, 'n_embd': 16, 'quant_range': 4, 'vocab_size': 256},
    {'n_layer': 2, 'n_head': 2, 'n_embd': 32, 'quant_range': 8, 'vocab_size': 256},
    {'n_layer': 2, 'n_head': 4, 'n_embd': 64, 'quant_range': 8, 'vocab_size': 512},
    {'n_layer': 3, 'n_head': 4, 'n_embd': 64, 'quant_range': 12, 'vocab_size': 512},
    
    # # Medium models
    # {'n_layer': 3, 'n_head': 6, 'n_embd': 96, 'quant_range': 12, 'vocab_size': 1024},
    # {'n_layer': 4, 'n_head': 8, 'n_embd': 128, 'quant_range': 12, 'vocab_size': 1024},
    # {'n_layer': 4, 'n_head': 4, 'n_embd': 64, 'quant_range': 16, 'vocab_size': 1024},
    
    # # Large models
    # {'n_layer': 6, 'n_head': 8, 'n_embd': 128, 'quant_range': 16, 'vocab_size': 2048},
    # {'n_layer': 6, 'n_head': 6, 'n_embd': 96, 'quant_range': 16, 'vocab_size': 1024},
    
    # # High quantization experiments
    # {'n_layer': 3, 'n_head': 4, 'n_embd': 64, 'quant_range': 8, 'vocab_size': 2048},
    # {'n_layer': 4, 'n_head': 4, 'n_embd': 64, 'quant_range': 16, 'vocab_size': 2048},
    
    # Your current baseline for comparison
    {'n_layer': 2, 'n_head': 1, 'n_embd': 16, 'quant_range': 8, 'vocab_size': 256},
]


def generate_full_grid():
    """Generate all combinations from hyperparameter grid"""
    keys = HYPERPARAMETER_GRID.keys()
    values = HYPERPARAMETER_GRID.values()
    combinations = []
    
    for combination in itertools.product(*values):
        param_dict = dict(zip(keys, combination))
        
        # Filter out invalid combinations
        if is_valid_combination(param_dict):
            combinations.append(param_dict)
    
    return combinations


def is_valid_combination(params):
    """Filter out invalid hyperparameter combinations"""
    # Ensure embedding dimension is divisible by number of heads
    if params['n_embd'] % params['n_head'] != 0:
        return False
    
    # Ensure head dimension is reasonable (at least 4, at most 64)
    head_dim = params['n_embd'] // params['n_head']
    if head_dim < 4 or head_dim > 64:
        return False
    
    # Ensure vocab size is reasonable for quantization range
    if params['vocab_size'] < params['quant_range']:
        return False
    
    # Skip extremely large models (memory constraints)
    total_params_estimate = params['n_layer'] * params['n_embd'] * params['n_embd'] * 4
    if total_params_estimate > 10_000_000:  # 10M parameter limit
        return False
    
    return True


def create_training_config(base_config, hyperparams):
    """Create a training config with specific hyperparameters"""
    
    class DynamicTrainingConfig:
        def __init__(self, base, hyperparams):
            # Copy all attributes from base config
            for attr in dir(base):
                if not attr.startswith('_'):
                    setattr(self, attr, getattr(base, attr))
            
            # Override with hyperparameters
            for key, value in hyperparams.items():
                setattr(self, key, value)
            
            # Update block_size to match seq_len
            self.block_size = self.seq_len
            
            # Generate unique run name
            self.wandb_run_name = f"L{self.n_layer}_H{self.n_head}_E{self.n_embd}_Q{self.quant_range}_V{self.vocab_size}"
    
    return DynamicTrainingConfig(base_config, hyperparams)


# ============================================================================
# MODIFIED TRAINING FUNCTIONS FOR SWEEP
# ============================================================================

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, patience=12, verbose=False, delta=0, save_path='best_model.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'⏳ EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'💾 Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        self.val_loss_min = val_loss


@torch.no_grad()
def evaluate(model, val_loader, tokenizer, device):
    """Evaluate model on validation set"""
    model.eval()
    total_val_loss = 0
    num_batches = len(val_loader)
    
    for batch_x, batch_y, _, _ in val_loader:
        x = batch_x.float().squeeze(-1).to(device)
        y = batch_y.float().squeeze(-1).to(device)
        
        # Tokenize inputs
        token_ids, attention_mask, tokenizer_state = tokenizer.context_input_transform(x.cpu())
        target_token_ids, target_attention_mask = tokenizer.label_input_transform(y.cpu(), tokenizer_state)
        
        # Move to device
        token_ids = token_ids.to(device)
        target_token_ids = target_token_ids.to(device)
        
        # Forward pass (no grad)
        logits, loss = model(token_ids, target_token_ids)
        total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / num_batches
    return avg_val_loss


def setup_environment(config):
    """Setup training environment"""
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)


def setup_data_loaders(config):
    """Setup training and validation data loaders"""
    train_dataset, train_loader = build_dataloader(
        dataset_name=config.dataset_name,
        features=config.features, 
        seq_len=config.seq_len, 
        label_len=config.seq_len - 1, 
        pred_len=1, 
        flag='train', 
        batch_size=config.batch_size,
        pretrain=False
    )

    validate_dataset, validate_loader = build_dataloader(
        dataset_name=config.dataset_name,
        features=config.features, 
        seq_len=config.seq_len, 
        label_len=config.seq_len - 1, 
        pred_len=1, 
        flag='val', 
        batch_size=config.batch_size,
        pretrain=False
    )
    
    return train_loader, validate_loader


def setup_model(config):
    """Setup model, optimizer, and tokenizer"""
    # Model initialization
    model_args = dict(
        n_layer=config.n_layer, 
        n_head=config.n_head, 
        n_embd=config.n_embd, 
        block_size=config.block_size,
        bias=config.bias, 
        vocab_size=config.vocab_size, 
        dropout=config.dropout
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(config.device)
    
    # Optimizer
    optimizer = model.configure_optimizers(
        config.weight_decay, 
        config.learning_rate, 
        (config.beta1, config.beta2), 
        config.device_type
    )
    
    # Tokenizer
    tokenizer = build_tokenizer(
        quant_range=config.quant_range,
        vocab_size=config.vocab_size,
        context_length=config.seq_len,
        prediction_length=config.seq_len
    )
    
    # Gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))
    
    # Compile model if requested
    if config.compile:
        model = torch.compile(model)
    
    return model, optimizer, tokenizer, scaler


def train_single_configuration(config, hyperparams, run_id):
    """Train a single hyperparameter configuration"""
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting run {run_id}: {hyperparams}")
    print(f"{'='*60}")
    
    # Create config for this run
    run_config = create_training_config(config, hyperparams)
    
    # Initialize W&B for this run
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=run_config.wandb_run_name,
        config=hyperparams,
        tags=config.wandb_tags + [f"run_{run_id}"],
        notes=f"Sweep run {run_id}: {config.wandb_notes}",
        save_code=False,
        reinit=True
    )
    
    try:
        # Setup
        setup_environment(run_config)
        train_loader, validate_loader = setup_data_loaders(run_config)
        model, optimizer, tokenizer, scaler = setup_model(run_config)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        wandb.log({
            "model/total_parameters": total_params,
            "model/parameters_per_layer": total_params / run_config.n_layer,
            "model/head_dimension": run_config.n_embd // run_config.n_head,
        })
        
        print(f"📊 Model: {total_params:,} parameters | Head dim: {run_config.n_embd // run_config.n_head}")
        
        # Training setup
        num_batches = len(train_loader)
        total_steps = run_config.epochs * num_batches
        best_val_loss = float('inf')
        
        early_stopping = EarlyStopping(
            patience=run_config.patience, 
            verbose=False
        )
        
        # Training loop
        training_start_time = time.time()
        
        for epoch in range(run_config.epochs):
            # Training phase
            model.train()
            epoch_loss = 0
            
            for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
                iteration = epoch * num_batches + i
                x = batch_x.float().squeeze(-1)
                y = batch_y.float().squeeze(-1)
                
                # Get learning rate
                min_lr = run_config.learning_rate * run_config.min_lr_factor
                lr = get_lr(iteration, total_steps, run_config.warmup_steps, 
                           run_config.learning_rate, min_lr, run_config.decay_lr)
                
                # Update learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass
                token_ids, _, tokenizer_state = tokenizer.context_input_transform(x)
                target_token_ids, _ = tokenizer.label_input_transform(y, tokenizer_state)
                
                logits, loss = model(token_ids.to(run_config.device), target_token_ids.to(run_config.device))
                
                # Backward pass
                scaler.scale(loss).backward()
                
                if run_config.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), run_config.clip_grad)
                
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += loss.item()
            
            # Validation phase
            val_avg_loss = evaluate(model, validate_loader, tokenizer, run_config.device)
            train_avg_loss = epoch_loss / len(train_loader)
            
            # Track best validation loss
            if val_avg_loss < best_val_loss:
                best_val_loss = val_avg_loss
                best_epoch = epoch + 1
            
            # Log metrics
            wandb.log({
                "epoch/train_loss": train_avg_loss,
                "epoch/val_loss": val_avg_loss,
                "epoch/best_val_loss": best_val_loss,
                "epoch/learning_rate": lr,
                "epoch/epoch": epoch + 1,
            })
            
            # Early stopping check
            early_stopping(val_avg_loss, model, epoch + 1)
            
            if early_stopping.early_stop:
                print(f"⏹️  Early stopping at epoch {epoch + 1}")
                break
            
            # Progress update
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{run_config.epochs} | Train: {train_avg_loss:.4f} | Val: {val_avg_loss:.4f} | Best: {best_val_loss:.4f}")
        
        # Calculate training time
        total_training_time = time.time() - training_start_time
        
        # Final metrics
        final_metrics = {
            "final/best_val_loss": best_val_loss,
            "final/best_epoch": best_epoch,
            "final/total_epochs": epoch + 1,
            "final/training_time_minutes": total_training_time / 60,
            "final/convergence_rate": best_val_loss / (epoch + 1),  # Lower is better
            "final/efficiency_score": best_val_loss * total_params / 1000000,  # Loss * params in millions
        }
        
        wandb.log(final_metrics)
        
        print(f"✅ Completed: Best Val Loss = {best_val_loss:.4f} at epoch {best_epoch}")
        
        # Return results
        result = {
            'hyperparams': hyperparams,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'total_epochs': epoch + 1,
            'total_params': total_params,
            'training_time': total_training_time,
            'early_stopped': early_stopping.early_stop,
        }
        
        return result
    
    except Exception as e:
        print(f"❌ Error in run {run_id}: {str(e)}")
        wandb.log({"error": str(e)})
        return {
            'hyperparams': hyperparams,
            'error': str(e),
            'best_val_loss': float('inf'),
        }
    
    finally:
        wandb.finish()
        # Clear GPU cache
        torch.cuda.empty_cache()


def run_hyperparameter_sweep(use_smart_combinations=True, max_runs=None):
    """Run the complete hyperparameter sweep"""
    
    base_config = SweepConfig()
    
    # Generate combinations
    if use_smart_combinations:
        combinations = SMART_COMBINATIONS
        print(f"🔍 Using {len(combinations)} smart combinations")
    else:
        combinations = generate_full_grid()
        print(f"🔍 Generated {len(combinations)} combinations from full grid")
    
    # Limit runs if specified
    if max_runs and len(combinations) > max_runs:
        combinations = combinations[:max_runs]
        print(f"🔄 Limited to first {max_runs} combinations")
    
    print(f"🚀 Starting hyperparameter sweep with {len(combinations)} configurations")
    
    # Store results
    all_results = []
    
    # Run sweep
    for run_id, hyperparams in enumerate(combinations, 1):
        result = train_single_configuration(base_config, hyperparams, run_id)
        all_results.append(result)
        
        # Save intermediate results
        with open(f'{base_config.output_dir}/sweep_results_{run_id}.json', 'w') as f:
            json.dump(all_results, f, indent=2)
    
    # Analyze results
    analyze_sweep_results(all_results, base_config.output_dir)
    
    return all_results


def analyze_sweep_results(results, output_dir):
    """Analyze and summarize sweep results"""
    
    print(f"\n{'='*60}")
    print("📊 HYPERPARAMETER SWEEP RESULTS ANALYSIS")
    print(f"{'='*60}")
    
    # Filter out failed runs
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    print(f"✅ Successful runs: {len(successful_results)}")
    print(f"❌ Failed runs: {len(failed_results)}")
    
    if not successful_results:
        print("No successful runs to analyze!")
        return
    
    # Sort by validation loss
    successful_results.sort(key=lambda x: x['best_val_loss'])
    
    print(f"\n🏆 TOP 5 CONFIGURATIONS:")
    print("-" * 60)
    
    for i, result in enumerate(successful_results[:5], 1):
        hp = result['hyperparams']
        print(f"{i}. Val Loss: {result['best_val_loss']:.4f} | "
              f"Params: {result['total_params']:,} | "
              f"L{hp['n_layer']}_H{hp['n_head']}_E{hp['n_embd']}_Q{hp['quant_range']}_V{hp['vocab_size']}")
    
    # Save detailed results
    with open(f'{output_dir}/final_sweep_results.json', 'w') as f:
        json.dump({
            'successful_results': successful_results,
            'failed_results': failed_results,
            'summary': {
                'total_runs': len(results),
                'successful_runs': len(successful_results),
                'failed_runs': len(failed_results),
                'best_val_loss': successful_results[0]['best_val_loss'] if successful_results else None,
                'best_config': successful_results[0]['hyperparams'] if successful_results else None,
            }
        }, f, indent=2)
    
    print(f"\n📁 Detailed results saved to: {output_dir}/final_sweep_results.json")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    USE_SMART_COMBINATIONS = True  # Set to False for full grid search
    MAX_RUNS = None  # Set to limit number of runs (e.g., 20)
    
    # Run sweep
    results = run_hyperparameter_sweep(
        use_smart_combinations=USE_SMART_COMBINATIONS,
        max_runs=MAX_RUNS
    )
    
    print("🎉 Hyperparameter sweep completed!")