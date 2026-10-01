import argparse
import os
import random
import numpy as np
import torch
from exp.exp_main import Exp_Main


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_argument_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description='EntroPE for Time Series Forecasting'
    )
    
    # Basic configuration
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')
    parser.add_argument('--is_training', type=int, required=True, default=1, 
                        help='status: 1 for training, 0 for testing')
    parser.add_argument('--model_id', type=str, required=True, default='test', 
                        help='model identifier')
    parser.add_argument('--setting_suffix', type=str, default='',
                        help='optional unique suffix appended to the generated setting name; '
                             'useful for sweeps whose batch/dropout/seed values are otherwise '
                             'not represented in checkpoint and result paths')
    parser.add_argument('--model', type=str, default='EntroPE',
                        help='model name')
    
    # Data loader configuration
    parser.add_argument('--data', type=str, required=True, default='ETTm1', 
                        help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', 
                        help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task: M (multivariate->multivariate), '
                             'S (univariate->univariate), MS (multivariate->univariate)')
    parser.add_argument('--target', type=str, default='OT', 
                        help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='time features encoding frequency')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', 
                        help='location of model checkpoints')
    parser.add_argument('--save_test_artifacts', type=int, default=0,
                        help='save prediction arrays and test PDFs (1) or report metrics only (0)')
    
    # Forecasting task configuration
    parser.add_argument('--seq_len', type=int, default=96, 
                        help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, 
                        help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, 
                        help='prediction sequence length')
    
    # EntroPE-specific parameters
    _add_entrope_args(parser)
    
    # Traditional transformer parameters (for compatibility)
    _add_transformer_args(parser)
    
    # Optimization parameters
    _add_optimization_args(parser)
    
    # GPU configuration
    _add_gpu_args(parser)
    
    return parser


def _add_entrope_args(parser):
    """Add EntroPE-specific arguments"""
    # Tokenization
    parser.add_argument('--vocab_size', type=int, default=256, 
                        help='vocabulary size for byte-level tokenization')
    parser.add_argument('--entropy_model_checkpoint_dir', type=str, 
                        default='./entropy_model_checkpoints/dm16',
                        help='directory for entropy model checkpoints')
    parser.add_argument('--model_id_name', type=str, default='ETTm1', 
                        help='model ID for saved entropy checkpoints')
    
    # Cross-attention configuration
    parser.add_argument('--cross_attn_k', type=int, default=1, 
                        help='cross attention key dimension')
    parser.add_argument('--cross_attn_nheads', type=int, default=4, 
                        help='number of cross attention heads')
    parser.add_argument('--cross_attn_window_encoder', type=int, default=1,
                        help='cross attention window size for encoder')
    parser.add_argument('--cross_attn_window_decoder', type=int, default=1,
                        help='cross attention window size for decoder')
    parser.add_argument('--local_attention_window_len', type=int, default=96,
                        help='local attention window length. NOTE: fixed-size sliding-window '
                             'self-attention only actually applies when < seq_len (see '
                             '--encoder_self_attn_within_patch for the exact, dynamic-boundary-aware '
                             'alternative); at the default 96 == seq_len it is equivalent to full causal.')
    parser.add_argument('--encoder_self_attn_within_patch', type=int, default=1,
                        help='if 1, the Patch Encoder\'s token-level self-attention is masked to '
                             'only attend within each token\'s own (variable-length, entropy-defined) '
                             'patch, instead of full causal over the whole input. Exact and '
                             'dynamic-boundary-aware, unlike --local_attention_window_len\'s fixed-size '
                             'window. Default 1 — this is the setting used for ALL reported EntroPE '
                             'results (local encoder attention never crosses a patch boundary).')
    
    # Patching configuration
    parser.add_argument('--patch_size', type=int, default=16, 
                        help='patch length')
    parser.add_argument('--max_patch_length', type=int, default=16, 
                        help='maximum patch length')
    parser.add_argument('--patching_batch_size', type=int, default=512, 
                        help='batch size for patching')
    parser.add_argument('--patching_threshold', type=float, default=4.0, 
                        help='entropy patching threshold')
    parser.add_argument('--patching_threshold_add', type=float, default=0.2, 
                        help='additional patching threshold')
    parser.add_argument('--monotonicity', type=int, default=0,
                        help='monotonic patching (1: True, 0: False)')
    parser.add_argument('--channel_sample_size', type=int, default=None,
                        help='if set and less than enc_in, randomly sample this many '
                             'channels per training iteration (channel-independent '
                             'architectures only) to bound activation memory on '
                             'very high-channel-count datasets; validation/test '
                             'always use every channel')
    parser.add_argument('--decoder_pool_size', type=int, default=None,
                        help='if set, adaptively pool the seq_len dimension down to '
                             'this fixed size before the flatten+linear prediction '
                             'head, so the head\'s parameter count (d_model*R) is '
                             'independent of input length -- use for L=96 vs L=512 '
                             'comparisons where the unpooled head (d_model*seq_len) '
                             'would otherwise confound length with model size')
    parser.add_argument('--decoder_mode', type=str, default='flatten',
                        choices=['flatten', 'horizon_query', 'latent_resampler', 'patchtst_head'],
                        help='flatten: original Linear(d_model*seq_len -> pred_len) head '
                             '(param count scales with L; ignores --decoder_pool_size unless set). '
                             'horizon_query: fixed sinusoidal per-horizon-step queries '
                             'cross-attend directly to the backbone\'s point-level output; '
                             'decoder params are O(d_model^2), independent of L and pred_len. '
                             'latent_resampler: same, but the point-level output is first '
                             'cross-attended into --n_latent_tokens fixed latent tokens before '
                             'the horizon queries attend to those instead of the raw output. '
                             'patchtst_head: skip the fusion decoder entirely and flatten+linear '
                             'directly on patch-level global-transformer output (dim_global), '
                             'matching PatchTST\'s own head; requires --decoder_pool_size since '
                             'dynamic patching gives a variable patch count per batch.')
    parser.add_argument('--n_latent_tokens', type=int, default=32,
                        help='number of fixed latent tokens for --decoder_mode latent_resampler')
    parser.add_argument('--boundary_method', type=str, default='entropy',
                        choices=['entropy', 'static', 'local_diff', 'variance_cp',
                                 'cusum', 'random', 'empirical_entropy', 'frequency_based',
                                 'cmi_threshold', 'learned_cmi'],
                        help='boundary detection method: entropy (EntroPE default), '
                             'static, local_diff, variance_cp, cusum, random, '
                             'empirical_entropy (EAPformer-style), frequency_based, '
                             'cmi_threshold (MVG+CMI+threshold, v2 fallback), '
                             'learned_cmi (end-to-end learned, requires --model EntroPE_v2)')

    # ---- EntroPE_v2 / MVG model ----
    parser.add_argument('--mvg_layers',   type=int,   default=2,
                        help='MVG causal transformer layers')
    parser.add_argument('--mvg_embd',     type=int,   default=64,
                        help='MVG embedding dimension')
    parser.add_argument('--mvg_heads',    type=int,   default=4,
                        help='MVG attention heads')
    parser.add_argument('--mvg_dropout',  type=float, default=0.1,
                        help='MVG dropout')
    parser.add_argument('--mvg_cov_rank', type=int,   default=None,
                        help='MVG covariance rank (None=full Cholesky, int=low-rank+diag)')
    parser.add_argument('--mvg_checkpoint', type=str, default=None,
                        help='Path to pretrained MVG checkpoint (from pretrain_mvg.py)')

    # ---- Boundary scorer ----
    parser.add_argument('--scorer_hidden', type=int, default=64,
                        help='Hidden dim of LearnedBoundaryScorer MLP')

    # ---- Gumbel-sigmoid patcher ----
    parser.add_argument('--gumbel_tau_init',  type=float, default=5.0,
                        help='Initial Gumbel-sigmoid temperature')
    parser.add_argument('--gumbel_tau_min',   type=float, default=0.5,
                        help='Minimum Gumbel-sigmoid temperature (after annealing)')
    parser.add_argument('--gumbel_tau_decay', type=int,   default=5000,
                        help='Decay steps for Gumbel temperature annealing')

    # ---- Patch-rate regularisation ----
    parser.add_argument('--target_avg_patch_len', type=int,   default=8,
                        help='Target average patch length (controls boundary rate)')
    parser.add_argument('--min_patch_len',        type=int,   default=3,
                        help='Minimum allowed patch length (penalised if violated)')
    parser.add_argument('--lambda_count', type=float, default=10.0,
                        help='Weight for patch-rate regulariser')
    parser.add_argument('--lambda_min',   type=float, default=0.5,
                        help='Weight for min-patch-length regulariser')
    parser.add_argument('--lambda_div',   type=float, default=0.1,
                        help='Weight for boundary-diversity regulariser')
    parser.add_argument('--lambda_nll',   type=float, default=0.1,
                        help='Weight for MVG calibration NLL term')
    parser.add_argument('--beta_nll',     type=float, default=0.5,
                        help='beta for Seitzer 2022 beta-NLL (0=MSE-like, 1=standard NLL)')

    # Regularization
    parser.add_argument('--fc_dropout', type=float, default=0.1,
                        help='fully connected layer dropout')
    parser.add_argument('--head_dropout', type=float, default=0.1, 
                        help='prediction head dropout')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='general dropout rate')
    
    # Other EntroPE settings
    parser.add_argument('--stride', type=int, default=8, 
                        help='stride for patching')
    parser.add_argument('--padding_patch', type=str, default='end', 
                        help='padding strategy (None or end)')
    
    # RevIN and decomposition
    parser.add_argument('--revin', type=int, default=1, 
                        help='use RevIN (1: True, 0: False)')
    parser.add_argument('--affine', type=int, default=1, 
                        help='RevIN affine transformation (1: True, 0: False)')
    parser.add_argument('--subtract_last', type=int, default=0, 
                        help='RevIN normalization (0: subtract mean, 1: subtract last)')
    parser.add_argument('--decomposition', type=int, default=0, 
                        help='use decomposition (1: True, 0: False)')
    parser.add_argument('--kernel_size', type=int, default=25, 
                        help='decomposition kernel size')
    parser.add_argument('--individual', type=int, default=0, 
                        help='individual head per variable (1: True, 0: False)')


def _add_transformer_args(parser):
    """Add traditional transformer arguments for compatibility"""
    parser.add_argument('--embed_type', type=int, default=0, 
                        help='embedding type')
    parser.add_argument('--enc_in', type=int, default=7, 
                        help='encoder input size (number of channels)')
    parser.add_argument('--dec_in', type=int, default=7, 
                        help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, 
                        help='output size')
    parser.add_argument('--d_model', type=int, default=512, 
                        help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=4,
                        help='number of attention heads')
    parser.add_argument('--e_layers', type=int, default=2, 
                        help='number of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, 
                        help='number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256,
                        help='dimension of feedforward network')
    parser.add_argument('--global_d_model', type=int, default=None,
                        help='override d_model for the Global Transformer only '
                             '(e.g. to match a baseline like PatchTST); None = use --d_model')
    parser.add_argument('--global_e_layers', type=int, default=None,
                        help='override e_layers for the Global Transformer only; None = use --e_layers')
    parser.add_argument('--global_n_heads', type=int, default=None,
                        help='override n_heads for the Global Transformer only; None = use --n_heads')
    parser.add_argument('--global_d_ff', type=int, default=None,
                        help='override the FFN rounding unit for the Global Transformer only; '
                             'None = use --d_ff (shared)')
    parser.add_argument('--moving_avg', type=int, default=25, 
                        help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, 
                        help='attention factor')
    parser.add_argument('--distil', action='store_false', default=True,
                        help='disable distilling in encoder')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding type')
    parser.add_argument('--activation', type=str, default='gelu', 
                        help='activation function')
    parser.add_argument('--output_attention', action='store_true', 
                        help='output attention weights in encoder')
    parser.add_argument('--do_predict', action='store_true', 
                        help='predict unseen future data')


def _add_optimization_args(parser):
    """Add optimization-related arguments"""
    parser.add_argument('--num_workers', type=int, default=4,
                        help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, 
                        help='number of experiment iterations')
    parser.add_argument('--train_epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch size for training')
    parser.add_argument('--patience', type=int, default=10, 
                        help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', 
                        help='experiment description')
    parser.add_argument('--loss', type=str, default='mse', 
                        help='loss function')
    parser.add_argument('--lradj', type=str, default='TST', 
                        help='learning rate adjustment strategy')
    parser.add_argument('--pct_start', type=float, default=0.3, 
                        help='percentage of cycle spent increasing LR')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='use automatic mixed precision training')


def _add_gpu_args(parser):
    """Add GPU-related arguments"""
    parser.add_argument('--use_gpu', type=bool, default=True, 
                        help='use GPU if available')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU device ID')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1,2,3', 
                        help='device IDs for multiple GPUs')
    parser.add_argument('--test_flop', action='store_true', default=False,
                        help='test FLOPs calculation')


def configure_gpu(args):
    """Configure GPU settings"""
    args.use_gpu = torch.cuda.is_available() and args.use_gpu
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    return args


def generate_setting_name(args):
    """Generate experiment setting name: EntroPE_<dataset>_SL<seq_len>_PL<pred_len>.

    This is the checkpoint directory name (checkpoints/<setting>/checkpoint.pth).
    We use model_id_name (set uniquely per dataset in every launch script), not
    args.data, because several datasets (Electricity, weather, Traffic, ...)
    share args.data=='custom'. Each (dataset, horizon) cell has exactly one best
    config, so no extra hyperparameters are needed to keep the name unique.
    """
    return f"{args.model}_{args.model_id_name}_SL{args.seq_len}_PL{args.pred_len}"


def run_training(args, exp, setting):
    """Run training pipeline"""
    print(f'>>>>>>>start training: {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print(f'>>>>>>>testing: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test(setting)
    
    if args.do_predict:
        print(f'>>>>>>>predicting: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.predict(setting, True)
    
    torch.cuda.empty_cache()


def run_testing(args, exp, setting):
    """Run testing only"""
    print(f'>>>>>>>testing: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test(setting, test=1)
    torch.cuda.empty_cache()


def main():
    """Main execution function"""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.random_seed)
    
    # Configure GPU
    args = configure_gpu(args)
    
    # Print configuration
    print('=' * 80)
    print('Experiment Configuration:')
    print('=' * 80)
    print(args)
    print('=' * 80)
    
    # Run experiments
    if args.is_training:
        for iteration in range(args.itr):
            setting = generate_setting_name(args)
            exp = Exp_Main(args)
            
            if args.itr > 1:
                print(f'\n>>> Iteration {iteration + 1}/{args.itr} <<<')
            
            run_training(args, exp, setting)
    else:
        setting = generate_setting_name(args)
        exp = Exp_Main(args)
        run_testing(args, exp, setting)


if __name__ == '__main__':
    main()
