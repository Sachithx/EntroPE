import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='EntroPE',
                        help='model name, options: [Autoformer, Informer, Transformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTm1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


    # DLinear
    #parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

    # CAPE-TST
    parser.add_argument('--vocab_size', type=int, default=256, help='vocabulary size for byte-level tokenization')
    parser.add_argument('--quant_range', type=int, default=6, help='quantization range for byte-level tokenization')
    parser.add_argument('--entropy_model_checkpoint_dir', type=str, default='./entropy_model_checkpoints/', help='directory for entropy model checkpoints')
    parser.add_argument('--model_id_name', type=str, default='ETTm1', help='model ID name for saved entropy checkpoints')
    parser.add_argument('--n_layers_local_encoder', type=int, default=2, help='number of local encoder layers')
    parser.add_argument('--n_layers_local_decoder', type=int, default=2, help='number of local decoder layers')
    parser.add_argument('--n_layers_global', type=int, default=2, help='number of global layers')   
    parser.add_argument('--n_heads_local_encoder', type=int, default=4, help='number of local encoder heads')
    parser.add_argument('--n_heads_local_decoder', type=int, default=4, help='number of local decoder heads')
    parser.add_argument('--n_heads_global', type=int, default=4, help='number of global heads')
    parser.add_argument('--dim_global', type=int, default=32, help='dimension of global representation')
    parser.add_argument('--dim_local_encoder', type=int, default=16, help='dimension of local encoder representation')
    parser.add_argument('--dim_local_decoder', type=int, default=16, help='dimension of local decoder representation')
    parser.add_argument('--cross_attn_window_encoder', type=int, default=96, help='cross attention window size for encoder')
    parser.add_argument('--cross_attn_window_decoder', type=int, default=96, help='cross attention window size for decoder')
    parser.add_argument('--local_attention_window_len', type=int, default=96, help='local attention window length')
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--multiple_of', type=int, default=128, help='multiple of 128 for efficient attention')
    parser.add_argument('--head_dropout', type=float, default=0.05, help='head dropout')
    parser.add_argument('--patch_size', type=int, default=16, help='patch length')
    parser.add_argument('--max_patch_length', type=int, default=16, help='maximum patch length')
    parser.add_argument('--patching_batch_size', type=int, default=512, help='batch size for patching')
    parser.add_argument('--patching_threshold', type=float, default=4, help='patching threshold')
    parser.add_argument('--patching_threshold_add', type=float, default=0.2, help='additional patching threshold')
    parser.add_argument('--monotonicity', type=int, default=0, help='monotonic patching; True 1 False 0')
    parser.add_argument('--cross_attn_k', type=int, default=1, help='cross attention key dimension')
    parser.add_argument('--cross_attn_nheads', type=int, default=4, help='number of cross attention heads')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=1, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

    # Formers 
    parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = (
                f"{args.model}_{args.data}"          # PatchTST_ETTm1
                f"_SL{args.seq_len}"                 # SL96
                f"_PL{args.pred_len}"                # PL96 / PL192 …
                f"_GL{args.n_layers_global}"         # GL2   (global layers)
                f"_EL{args.n_layers_local_encoder}"  # EL2   (local‑enc layers)
                f"_DL{args.n_layers_local_decoder}"  # DL2   (local‑dec layers)
                f"_GD{args.dim_global}"              # GD32  (global dim)
                f"_DD{args.dim_local_decoder}"       # LD16  (local‑dec dim)
                f"_ED{args.dim_local_encoder}"       # LE16  (local‑enc
                f"_GH{args.n_heads_global}"       # GH4   (global heads)
                f"_EH{args.n_heads_local_encoder}"    # EH4   (local‑enc heads)
                f"_DH{args.n_heads_local_decoder}"    # DH4   (
            )


            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = (
            f"{args.model}_{args.data}"          # PatchTST_ETTm1
            f"_SL{args.seq_len}"                 # SL96
            f"_PL{args.pred_len}"                # PL96 / PL192 …
            f"_GL{args.n_layers_global}"         # GL2   (global layers)
            f"_EL{args.n_layers_local_encoder}"  # EL2   (local‑enc layers)
            f"_DL{args.n_layers_local_decoder}"  # DL2   (local‑dec layers)
            f"_GD{args.dim_global}"              # GD32  (global dim)
            f"_DD{args.dim_local_decoder}"       # LD16  (local‑dec dim)
            f"_ED{args.dim_local_encoder}"       # LE16  (local‑enc
            f"_GH{args.n_heads_global}"       # GH4   (global heads)
            f"_EH{args.n_heads_local_encoder}"    # EH4   (local‑enc heads)
            f"_DH{args.n_heads_local_decoder}"    # DH4   (
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        