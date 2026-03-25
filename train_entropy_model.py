import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from models.GPT2EntropyModel import GPTConfig, GPT
from utils.train_utils import get_lr
from layers.Tokenizer import build_tokenizer
from data_provider.data_factory import data_provider


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Train GPT-2 Entropy Model for EntroPE')

    # Dataset
    parser.add_argument('--dataset', type=str, default='ETTh2',
                        help='Dataset name (ETTh1, ETTh2, ETTm1, ETTm2, Electricity, weather)')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='Data file name')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='Root path of data files')
    parser.add_argument('--freq', type=str, default='h',
                        help='Time feature encoding frequency (h=hourly, t=minute)')
    parser.add_argument('--features', type=str, default='M',
                        help='Forecasting task type (M, S, MS)')

    # Model architecture
    parser.add_argument('--n_layer', type=int, default=2, help='Number of GPT layers')
    parser.add_argument('--n_head', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_embd', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--vocab_size', type=int, default=256, help='Vocabulary size')

    # Sequence lengths
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')

    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1)
    parser.add_argument('--clip_grad', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--min_lr_factor', type=float, default=0.05)

    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='entropy_model_checkpoints',
                        help='Directory to save entropy model checkpoints')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')

    # Wandb (optional)
    parser.add_argument('--use_wandb', action='store_true', default=False,
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='W&B project name (auto-set if not given)')

    return parser


def _get_data_key(dataset_name):
    """Map dataset name to data_factory key."""
    ett_datasets = {'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2'}
    return dataset_name if dataset_name in ett_datasets else 'custom'


def build_config(args):
    """Augment args namespace with derived fields expected by data_provider."""
    args.data = _get_data_key(args.dataset)
    args.label_len = args.seq_len - 1
    args.pred_len = 1
    args.block_size = args.seq_len
    args.target = 'OT'
    args.embed = 'timeF'
    args.dataset_name = args.dataset
    args.bias = False
    args.decay_lr = True
    args.beta1 = 0.9
    args.beta2 = 0.95
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dtype = ('bfloat16' if torch.cuda.is_available()
                  and torch.cuda.is_bf16_supported() else 'float16')
    if args.wandb_project is None:
        args.wandb_project = f"Entropy Model - {args.dataset}"
    return args


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, save_path='best_model.pt'):
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
            self._save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model, epoch):
        if self.verbose:
            print(f'Val loss decreased ({self.val_loss_min:.6f} -> {val_loss:.6f}). Saving...')
        state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'val_loss': val_loss},
                   self.save_path)
        self.val_loss_min = val_loss


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate(model, val_loader, tokenizer, device):
    model.eval()
    total_loss = 0.0
    for batch_x, batch_y, _, _ in val_loader:
        x = batch_x.float().squeeze(-1).permute(0, 2, 1)
        y = batch_y.float().squeeze(-1).permute(0, 2, 1)
        bs, nvars, seq_len = x.shape
        x = x.reshape(bs * nvars, seq_len)
        y = y.reshape(bs * nvars, seq_len)

        token_ids, _, tokenizer_state = tokenizer.context_input_transform(x.cpu())
        target_token_ids, _ = tokenizer.label_input_transform(y.cpu(), tokenizer_state)

        _, loss = model(token_ids.to(device), target_token_ids.to(device))
        total_loss += loss.item()

    return total_loss / len(val_loader)


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_epoch(model, train_loader, optimizer, tokenizer, scaler, args,
                epoch, total_steps, early_stopping, wandb=None):
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)
    t0 = time.time()

    bar = tqdm(enumerate(train_loader), total=num_batches,
               desc=f'Epoch {epoch+1}/{args.epochs}', leave=True)

    for i, (batch_x, batch_y, _, _) in bar:
        iteration = epoch * num_batches + i
        x = batch_x.float().squeeze(-1).permute(0, 2, 1)
        y = batch_y.float().squeeze(-1).permute(0, 2, 1)
        bs, nvars, seq_len = x.shape
        x = x.reshape(bs * nvars, seq_len)
        y = y.reshape(bs * nvars, seq_len)

        # Learning rate schedule
        min_lr = args.learning_rate * args.min_lr_factor
        lr = get_lr(iteration, total_steps, args.warmup_steps,
                    args.learning_rate, min_lr, args.decay_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(args.grad_accumulation_steps):
            token_ids, _, tokenizer_state = tokenizer.context_input_transform(x)
            target_token_ids, _ = tokenizer.label_input_transform(y, tokenizer_state)
            _, loss = model(token_ids.to(args.device), target_token_ids.to(args.device))
            loss = loss / args.grad_accumulation_steps
            scaler.scale(loss).backward()
            total_loss += loss.item() * args.grad_accumulation_steps

        grad_norm = 0.0
        if args.clip_grad > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        scaler.step(optimizer)
        scaler.update()

        epoch_loss += total_loss
        avg = epoch_loss / (i + 1)
        bar.set_postfix({'loss': f'{total_loss:.4f}', 'avg': f'{avg:.4f}',
                         'lr': f'{lr:.2e}', 'p': f'{early_stopping.counter}/{args.patience}'})

        if wandb is not None:
            step = epoch * num_batches + i + 1
            wandb.log({'train/batch_loss': total_loss, 'train/lr': lr,
                       'train/grad_norm': float(grad_norm)}, step=step)

    return epoch_loss / num_batches, time.time() - t0, lr


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args = build_config(args)

    # Hardware
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    torch.set_float32_matmul_precision('high')

    # Output directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Write params.json BEFORE training so that Patcher can load it
    params = {
        "entropy_model": {
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "dropout": args.dropout,
            "bias": args.bias,
            "vocab_size": args.vocab_size,
            "block_size": args.seq_len,
        }
    }
    params_path = Path(args.checkpoint_dir) / "params.json"
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"params.json written to {params_path}")

    # Optional wandb
    wandb_run = None
    if args.use_wandb:
        import wandb as _wandb
        wandb_run = _wandb.init(
            project=args.wandb_project,
            config={k: v for k, v in vars(args).items()
                    if not callable(v) and not k.startswith('_')},
            tags=["entropy-model", "time-series"],
        )
        print(f"W&B run: {wandb_run.url}")

    # Data
    train_dataset, train_loader = data_provider(args, flag='train')
    _, val_loader = data_provider(args, flag='val')
    print(f"Dataset: {args.dataset} | train={len(train_dataset)} batches={len(train_loader)}")

    # Model
    gptconf = GPTConfig(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        block_size=args.block_size, bias=args.bias,
        vocab_size=args.vocab_size, dropout=args.dropout,
    )
    model = GPT(gptconf)
    model.to(args.device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate,
        (args.beta1, args.beta2), args.device_type
    )
    tokenizer = build_tokenizer(args)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))

    # Checkpoint path: entropy_model_checkpoints/{dataset}.pt
    ckpt_path = str(Path(args.checkpoint_dir) / f"{args.dataset}.pt")
    early_stopping = EarlyStopping(
        patience=args.patience, verbose=True, save_path=ckpt_path
    )

    num_batches = len(train_loader)
    total_steps = args.epochs * num_batches
    best_val = float('inf')
    train_start = time.time()

    print(f"Training: {args.epochs} epochs, early stopping patience={args.patience}")
    print(f"Checkpoint will be saved to: {ckpt_path}")

    for epoch in range(args.epochs):
        wandb_obj = _wandb if (args.use_wandb and wandb_run is not None) else None
        train_loss, train_time, current_lr = train_epoch(
            model, train_loader, optimizer, tokenizer, scaler,
            args, epoch, total_steps, early_stopping, wandb=wandb_obj
        )

        val_loss = evaluate(model, val_loader, tokenizer, args.device)
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Best: {best_val:.4f} | LR: {current_lr:.2e} | "
              f"Time: {train_time:.1f}s")

        if args.use_wandb and wandb_run is not None:
            _wandb.log({'epoch/train_loss': train_loss, 'epoch/val_loss': val_loss,
                        'epoch/best_val': best_val, 'epoch/lr': current_lr},
                       step=(epoch + 1) * num_batches)

        early_stopping(val_loss, model, epoch + 1)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_val:.6f}")
            break

    total_time = time.time() - train_start
    print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Best val loss: {best_val:.6f}")
    print(f"Checkpoint saved: {ckpt_path}")

    if args.use_wandb and wandb_run is not None:
        _wandb.finish()

    return model


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
