import time
from pathlib import Path
import torch
from tqdm import tqdm
import wandb
from models.GPT2EntropyModel import GPTConfig, GPT
from utils.train_utils import get_lr
from layers.Tokenizer import build_tokenizer
from data_provider.data_factory import data_provider

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

class TrainingConfig:
    """Training configuration parameters"""
    
    # Hardware Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    
    # Model Architecture
    n_layer = 3
    n_head = 2
    n_embd = 16
    dropout = 0.1
    bias = False
    vocab_size = 256
    
    # Data Configuration
    dataset_name = 'ETTh2'
    features = 'M'
    data="ETTh2"
    embed="timeF"
    freq = 'h'
    root_path = './dataset/'
    data_path = 'ETTh2.csv'
    batch_size = 128
    seq_len = 96
    pred_len = 1
    label_len = 95
    block_size = seq_len
    target = 'OT'
    num_workers = 4
    
    # Training Hyperparameters
    learning_rate = 1e-3
    weight_decay = 0.05
    beta1 = 0.9
    beta2 = 0.95
    epochs = 50
    grad_accumulation_steps = 1
    clip_grad = 1.0
    
    # Learning Rate Schedule
    warmup_steps = 0
    min_lr_factor = 0.05
    decay_lr = True
    
    # Training Control
    patience = 5  # Early stopping patience
    save_every = 10
    seed = 42
    compile = False
    
    # Output
    output_dir = "output"
    
    # W&B Configuration
    wandb_project = f"Entropy Model - {dataset_name}"
    wandb_entity = None  # Set to your wandb username/team if needed
    wandb_run_name = None  # Will be auto-generated if None
    wandb_tags = ["time-series", "transformer", "dynamic-patching"]
    wandb_notes = "Training context-aware dynamic patch encoder for time series"
    wandb_log_freq = 1  # Log every N batches (1 = every batch)
    wandb_save_model = True  # Save model artifacts to wandb


# ============================================================================
# EARLY STOPPING CLASS
# ============================================================================

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    
    def __init__(self, patience=12, verbose=False, delta=0, save_path='best_model.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement. 
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path for the checkpoint to be saved to.
        """
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
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'⏳ EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'💾 Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        # Get the original model state dict (handle compiled models)
        if hasattr(model, '_orig_mod'):
            # Model is compiled, save the original module
            state_dict = model._orig_mod.state_dict()
        else:
            # Model is not compiled
            state_dict = model.state_dict()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'val_loss': val_loss,
        }, self.save_path)
        self.val_loss_min = val_loss


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

@torch.no_grad()
def evaluate(model, val_loader, tokenizer, device):
    """Evaluate model on validation set"""
    model.eval()
    total_val_loss = 0
    num_batches = len(val_loader)
    
    for batch_x, batch_y, _, _ in val_loader:
        x = batch_x.float().squeeze(-1).to(device)
        y = batch_y.float().squeeze(-1).to(device)
        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
        y = y.permute(0,2,1)    # y: [Batch, Channel, Target length]
        bs, nvars, seq_len = x.shape
        x = x.reshape(bs * nvars, seq_len)
        y = y.reshape(bs * nvars, seq_len)
        
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


# ============================================================================
# WANDB SETUP FUNCTIONS
# ============================================================================

def setup_wandb(config):
    """Initialize Weights & Biases tracking"""
    
    # Convert config to dict for wandb
    config_dict = {
        # Model Architecture
        'model': {
            'n_layer': config.n_layer,
            'n_head': config.n_head,
            'n_embd': config.n_embd,
            'dropout': config.dropout,
            'bias': config.bias,
            'vocab_size': config.vocab_size,
            'block_size': config.block_size,
        },
        # Data Configuration
        'data': {
            'dataset_name': config.dataset_name,
            'features': config.features,
            'batch_size': config.batch_size,
            'seq_len': config.seq_len,
        },
        # Training Hyperparameters
        'training': {
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'beta1': config.beta1,
            'beta2': config.beta2,
            'epochs': config.epochs,
            'grad_accumulation_steps': config.grad_accumulation_steps,
            'clip_grad': config.clip_grad,
            'patience': config.patience,
        },
        # Learning Rate Schedule
        'lr_schedule': {
            'warmup_steps': config.warmup_steps,
            'min_lr_factor': config.min_lr_factor,
            'decay_lr': config.decay_lr,
        },
        # Hardware
        'hardware': {
            'device': str(config.device),
            'device_type': config.device_type,
            'dtype': config.dtype,
            'compile': config.compile,
        }
    }
    
    # Initialize wandb
    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        config=config_dict,
        tags=config.wandb_tags,
        notes=config.wandb_notes,
        save_code=True,  # Save the code that's running
    )
    
    print(f"🔄 W&B initialized - Project: {config.wandb_project}")
    print(f"📊 Run URL: {wandb.run.url}")
    
    return wandb.run


# ============================================================================
# TRAINING SETUP FUNCTIONS
# ============================================================================

def setup_environment(config):
    """Setup training environment"""
    torch.cuda.set_device(0)  # 0 here means "the first visible GPU"
    torch.set_float32_matmul_precision('high')
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Using device: {config.device}")
    print(f"Using precision: {config.dtype}")


def setup_data_loaders(config):
    """Setup training and validation data loaders"""
    train_dataset, train_loader = data_provider(config, flag='train')
    validate_dataset, validate_loader = data_provider(config, flag='val')
    
    print(f"Dataset: {config.dataset_name}, Features: {config.features}, "
          f"Batch Size: {config.batch_size}, Seq Len: {config.seq_len}")
    
    # Log dataset info to wandb
    wandb.log({
        "dataset/train_size": len(train_dataset),
        "dataset/val_size": len(validate_dataset),
        "dataset/train_batches": len(train_loader),
        "dataset/val_batches": len(validate_loader),
    })
    
    return train_loader, validate_loader


def setup_model(config, train_loader):
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
    
    # Log model info to wandb
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    wandb.log({
        "model/total_parameters": total_params,
        "model/trainable_parameters": trainable_params,
        "model/memory_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
    })
    
    print(f"📊 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optimizer
    optimizer = model.configure_optimizers(
        config.weight_decay, 
        config.learning_rate, 
        (config.beta1, config.beta2), 
        config.device_type
    )
    
    # Tokenizer
    # _, quant_info = find_quant_range(train_loader, epsilon=0.005)
    tokenizer = build_tokenizer(config)
    
    # Gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))
    
    # Compile model if requested
    if config.compile:
        model = torch.compile(model)
    
    return model, optimizer, tokenizer, scaler


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, train_loader, optimizer, tokenizer, scaler, config, epoch, 
                num_batches, total_steps, early_stopping):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    current_lr = 0
    t1 = time.time()
    
    progress_bar = tqdm(
        enumerate(train_loader), 
        total=len(train_loader), 
        desc=f"🏃 Epoch {epoch+1}/{config.epochs}", 
        position=0, 
        leave=True
    )
    
    for i, (batch_x, batch_y, _, _) in progress_bar:
        iteration = epoch * num_batches + i
        x = batch_x.float().squeeze(-1)
        y = batch_y.float().squeeze(-1)
        assert x.shape == y.shape, f"Input and target shapes do not match: {x.shape} vs {y.shape}"
        
        x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
        y = y.permute(0,2,1)    # y: [Batch, Channel, Target length]
        bs, nvars, seq_len = x.shape
        x = x.reshape(bs * nvars, seq_len)
        y = y.reshape(bs * nvars, seq_len)

        # Get learning rate
        min_lr = config.learning_rate * config.min_lr_factor
        lr = get_lr(iteration, total_steps, config.warmup_steps, 
                   config.learning_rate, min_lr, config.decay_lr)
        current_lr = lr
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        # Gradient accumulation loop
        for micro_step in range(config.grad_accumulation_steps):
            token_ids, attention_mask, tokenizer_state = tokenizer.context_input_transform(x)
            target_token_ids, target_attention_mask = tokenizer.label_input_transform(y, tokenizer_state)
            
            # Forward pass
            logits, loss = model(token_ids.to(config.device), target_token_ids.to(config.device))
            
            # Calculate loss
            loss = loss / config.grad_accumulation_steps
            
            # Backward pass
            scaler.scale(loss).backward()
            total_loss += loss.item() * config.grad_accumulation_steps
        
        # Gradient clipping
        grad_norm = 0
        if config.clip_grad > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        
        # Update weights
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        epoch_loss += total_loss
        avg_epoch_loss = epoch_loss / (i + 1)
        
        # Log to W&B every N batches
        if (i + 1) % config.wandb_log_freq == 0:
            step = epoch * num_batches + i + 1
            wandb.log({
                "train/batch_loss": total_loss,
                "train/epoch_avg_loss": avg_epoch_loss,
                "train/learning_rate": lr,
                "train/grad_norm": grad_norm,
                "train/memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
                "progress/epoch": epoch + 1,
                "progress/batch": i + 1,
                "progress/global_step": step,
                "early_stopping/counter": early_stopping.counter,
                "early_stopping/patience": early_stopping.patience,
            }, step=step)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{total_loss:.4f}",
            'avg_loss': f"{avg_epoch_loss:.4f}",
            'lr': f"{lr:.6f}",
            'patience': f"{early_stopping.counter}/{early_stopping.patience}"
        })
    
    train_time = time.time() - t1
    train_avg_loss = epoch_loss / len(train_loader)
    
    return train_avg_loss, train_time, current_lr


def main():
    """Main training function"""
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    config = TrainingConfig()
    setup_environment(config)
    
    # Initialize W&B
    wandb_run = setup_wandb(config)
    
    # Setup data loaders
    train_loader, validate_loader = setup_data_loaders(config)
    
    # Setup model, optimizer, tokenizer
    model, optimizer, tokenizer, scaler = setup_model(config, train_loader)
    
    # Training setup
    num_batches = len(train_loader)
    total_steps = config.epochs * num_batches
    best_val_loss = float('inf')
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config.patience, 
        verbose=True, 
        save_path=f'{config.output_dir}/best_model_checkpoint.pt'
    )
    
    # Training statistics tracking
    training_stats = {
        'train_losses': [],
        'val_losses': [],
        'learning_rates': [],
        'epochs_completed': 0,
        'best_epoch': 0,
        'total_training_time': 0
    }
    
    print(f"🚀 Starting training for {config.epochs} epochs with early stopping (patience={early_stopping.patience})")
    print(f"📊 Total steps: {total_steps}, Batches per epoch: {num_batches}")
    
    # Log initial metrics
    wandb.log({
        "setup/total_steps": total_steps,
        "setup/batches_per_epoch": num_batches,
        "setup/max_epochs": config.epochs,
    })
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    
    training_start_time = time.time()
    
    for epoch in range(config.epochs):
        # Training phase
        train_avg_loss, train_time, current_lr = train_epoch(
            model, train_loader, optimizer, tokenizer, scaler, config, 
            epoch, num_batches, total_steps, early_stopping
        )
        
        # Validation phase
        t2 = time.time()
        val_avg_loss = evaluate(model, validate_loader, tokenizer, config.device)
        val_time = time.time() - t2
        
        # Update training statistics
        training_stats['train_losses'].append(train_avg_loss)
        training_stats['val_losses'].append(val_avg_loss)
        training_stats['learning_rates'].append(current_lr)
        training_stats['epochs_completed'] = epoch + 1
        
        # Check if this is the best validation loss
        is_best_epoch = val_avg_loss < best_val_loss
        if is_best_epoch:
            best_val_loss = val_avg_loss
            training_stats['best_epoch'] = epoch + 1
        
        # Log epoch metrics to W&B
        epoch_metrics = {
            "epoch/train_loss": train_avg_loss,
            "epoch/val_loss": val_avg_loss,
            "epoch/best_val_loss": best_val_loss,
            "epoch/learning_rate": current_lr,
            "epoch/train_time_seconds": train_time,
            "epoch/val_time_seconds": val_time,
            "epoch/total_time_seconds": train_time + val_time,
            "epoch/is_best": is_best_epoch,
            "early_stopping/best_score": -early_stopping.best_score if early_stopping.best_score else 0,
            "early_stopping/counter": early_stopping.counter,
        }
        
        wandb.log(epoch_metrics, step=(epoch + 1) * num_batches)
        
        print(f"✅ Epoch {epoch+1}/{config.epochs} | Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f} | "
              f"Best Val: {best_val_loss:.4f} | Time: {train_time:.2f}s (Train) / {val_time:.2f}s (Val) | "
              f"LR: {current_lr:.6f}")
        
        # Early stopping check
        early_stopping(val_avg_loss, model, epoch + 1)
        
        if early_stopping.early_stop:
            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs!")
            print(f"📈 Best validation loss: {early_stopping.val_loss_min:.6f}")
            print(f"⏱️  No improvement for {early_stopping.patience} consecutive epochs")
            
            # Log early stopping event
            wandb.log({
                "early_stopping/triggered": True,
                "early_stopping/stopped_at_epoch": epoch + 1,
                "early_stopping/final_val_loss": early_stopping.val_loss_min,
            })
            break
    
    # ========================================================================
    # TRAINING COMPLETION
    # ========================================================================
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    training_stats['total_training_time'] = total_training_time
    
    # Training completed
    print(f"\n🎉 Training completed!")
    print(f"📊 Training Summary:")
    print(f"   • Total epochs: {training_stats['epochs_completed']}")
    print(f"   • Best epoch: {training_stats['best_epoch']}")
    print(f"   • Best validation loss: {best_val_loss:.6f}")
    print(f"   • Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f}m)")
    print(f"   • Average time per epoch: {total_training_time/training_stats['epochs_completed']:.2f}s")
    
    # Log final summary to W&B
    final_summary = {
        "summary/total_epochs": training_stats['epochs_completed'],
        "summary/best_epoch": training_stats['best_epoch'],
        "summary/best_val_loss": best_val_loss,
        "summary/total_training_time_seconds": total_training_time,
        "summary/total_training_time_minutes": total_training_time / 60,
        "summary/avg_time_per_epoch_seconds": total_training_time / training_stats['epochs_completed'],
        "summary/early_stopped": early_stopping.early_stop,
    }
    
    wandb.log(final_summary)
    
    # Load best model for final evaluation
    if early_stopping.save_path:
        print(f"🔄 Loading best model from {early_stopping.save_path}")
        checkpoint = torch.load(early_stopping.save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Best model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.6f})")
    
    # Save training statistics
    torch.save(training_stats, f'{config.output_dir}/training_statistics.pt')
    print(f"📁 Training statistics saved to {config.output_dir}/training_statistics.pt")
    
    # Save model artifact to W&B
    if config.wandb_save_model and early_stopping.save_path:
        model_artifact = wandb.Artifact(
            name=f"model_best_epoch_{training_stats['best_epoch']}", 
            type="model",
            description=f"Best model checkpoint from epoch {training_stats['best_epoch']} with val_loss {best_val_loss:.6f}"
        )
        model_artifact.add_file(early_stopping.save_path)
        wandb.log_artifact(model_artifact)
        print(f"📤 Model artifact saved to W&B")
    
    # Save training statistics as artifact
    stats_artifact = wandb.Artifact(
        name="training_statistics", 
        type="statistics",
        description="Complete training statistics and metrics"
    )
    stats_artifact.add_file(f'{config.output_dir}/training_statistics.pt')
    wandb.log_artifact(stats_artifact)
    
    # Finish W&B run
    wandb.finish()
    print(f"🏁 W&B run completed - View results at: {wandb_run.url}")
    
    return model, training_stats


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    model, stats = main()