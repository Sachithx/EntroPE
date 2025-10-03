from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, EntroPE
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import wandb

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        # Prepare wandb config before calling super().__init__()
        self.wandb_config = {
            'model': args.model,
            'seq_len': args.seq_len,
            'label_len': args.label_len,
            'pred_len': args.pred_len,
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'train_epochs': args.train_epochs,
            'patience': args.patience,
            'use_amp': args.use_amp,
            'features': args.features,
            'target': args.target,
            'freq': args.freq,
            'checkpoints': args.checkpoints,
            'use_multi_gpu': args.use_multi_gpu,
            'devices': args.devices if hasattr(args, 'devices') else None,
            'random_seed': args.random_seed if hasattr(args, 'random_seed') else None,
            'enc_in': args.enc_in if hasattr(args, 'enc_in') else None,
            'lradj': args.lradj if hasattr(args, 'lradj') else None,
            'pct_start': args.pct_start if hasattr(args, 'pct_start') else None,
        }
        
        # Add model-specific parameters (original transformer-based models)
        if hasattr(args, 'd_model'):
            self.wandb_config.update({
                # 'd_model': args.d_model,
                # 'n_heads': args.n_heads,
                # 'd_ff': args.d_ff,
                # 'e_layers': args.e_layers,
                # 'd_layers': args.d_layers,
                'dropout': args.dropout,
                'multiple_of': args.multiple_of if hasattr(args, 'multiple_of') else None,
                # 'activation': args.activation,
            })
            
        # # Add PatchTST-specific parameters
        # if hasattr(args, 'patch_len') and 'TST' in args.model:
        #     self.wandb_config.update({
        #         # 'patch_len': args.patch_len,
        #         # 'stride': args.stride,
        #     })
            
        # Add new PatchTST parameters from the script
        if 'PatchTST' in args.model or 'TST' in args.model or 'EntroPE' in args.model:
            patch_params = {
                'vocab_size': getattr(args, 'vocab_size', None),
                'quant_range': getattr(args, 'quant_range', None),
                'n_layers_local_encoder': getattr(args, 'n_layers_local_encoder', None),
                'n_layers_local_decoder': getattr(args, 'n_layers_local_decoder', None),
                'n_layers_global': getattr(args, 'n_layers_global', None),
                'n_heads_local_encoder': getattr(args, 'n_heads_local_encoder', None),
                'n_heads_local_decoder': getattr(args, 'n_heads_local_decoder', None),
                'n_heads_global': getattr(args, 'n_heads_global', None),
                'dim_global': getattr(args, 'dim_global', None),
                'dim_local_encoder': getattr(args, 'dim_local_encoder', None),
                'dim_local_decoder': getattr(args, 'dim_local_decoder', None),
                'cross_attn_k': getattr(args, 'cross_attn_k', None),
                'cross_attn_nheads': getattr(args, 'cross_attn_nheads', None),
                'fc_dropout': getattr(args, 'fc_dropout', None),
                'head_dropout': getattr(args, 'head_dropout', None),
                'patch_size': getattr(args, 'patch_size', None),
                'max_patch_length': getattr(args, 'max_patch_length', None),
                'patching_threshold': getattr(args, 'patching_threshold', None),
                'patching_threshold_add': getattr(args, 'patching_threshold_add', None),
                'monotonicity': getattr(args, 'monotonicity', None),
                'patching_batch_size': getattr(args, 'patching_batch_size', None),
                'cross_attn_window_encoder': getattr(args, 'cross_attn_window_encoder', None),
                'cross_attn_window_decoder': getattr(args, 'cross_attn_window_decoder', None),
                'local_attention_window_len': getattr(args, 'local_attention_window_len', None)
            }
            # Only add non-None parameters
            patch_params = {k: v for k, v in patch_params.items() if v is not None}
            self.wandb_config.update(patch_params)
        
        # Add general parameters that might be present
        general_params = {
            # 'd_ff': getattr(args, 'd_ff', None),
            'dropout': getattr(args, 'dropout', None),
            'des': getattr(args, 'des', None),
            'itr': getattr(args, 'itr', None),
        }
        # Only add non-None parameters
        general_params = {k: v for k, v in general_params.items() if v is not None}
        self.wandb_config.update(general_params)
        
        # Flag to track if wandb is initialized
        self.wandb_initialized = False
        
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'EntroPE': EntroPE,  
            'EntroPE': EntroPE,  
        }
        model = model_dict[self.args.model].Model(self.args).float()
        # print model architecture
        # print(model)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        # Log model parameters to wandb (only if wandb is initialized)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Store model info for later logging
        self.model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }
        
        # Only update wandb config if wandb is initialized
        if self.wandb_initialized:
            wandb.config.update(self.model_info)
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'EntroPE' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'EntroPE' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # Initialize wandb run
        wandb.init(
            project=f"time-series-forecasting-{self.args.data_path}",
            name=setting,
            config=self.wandb_config,
            reinit=True
        )
        self.wandb_initialized = True
        
        # Now update with model info
        if hasattr(self, 'model_info'):
            wandb.config.update(self.model_info)
        
        # Watch model gradients and parameters
        wandb.watch(self.model, log='all', log_freq=100)
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)

        # Log dataset info and training configuration
        dataset_info = {
            'train_samples': len(train_data),
            'val_samples': len(vali_data),
            'test_samples': len(test_data),
            'train_steps_per_epoch': train_steps,
            'scheduler_type': 'OneCycleLR',
            'optimizer_type': 'Adam',
            'criterion_type': 'MSELoss'
        }
        
        # Add batch size information
        if hasattr(self.args, 'patching_batch_size'):
            dataset_info['patching_batch_size'] = self.args.patching_batch_size
            
        wandb.log(dataset_info)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                if epoch == 0 and i == 0:
                    from ptflops import get_model_complexity_info    
                    with torch.cuda.device(0):
                        macs, params = get_model_complexity_info(self.model.cuda(), (batch_x.shape[1], batch_x.shape[2]), as_strings=True, print_per_layer_stat=False)
                        # print('Flops:' + flops)
                        # print('Params:' + params)
                        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
                        print('{:<30}  {:<8}'.format('Number of parameters: ', params))

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'EntroPE' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'EntroPE' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # Log batch-level metrics
                batch_metrics = {
                    'batch_loss': loss.item(),
                    'epoch': epoch,
                    'batch': i,
                    'learning_rate': model_optim.param_groups[0]['lr']
                }
                
                # Add gradient norm if available
                if hasattr(self.model, 'parameters'):
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    batch_metrics['gradient_norm'] = total_norm
                
                wandb.log(batch_metrics, step=epoch * train_steps + i)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    
                    # Log training progress
                    progress_metrics = {
                        'training_speed_s_per_iter': speed,
                        'estimated_time_left_s': left_time,
                        'progress_percent': ((epoch * train_steps + i) / (self.args.train_epochs * train_steps)) * 100
                    }
                    wandb.log(progress_metrics, step=epoch * train_steps + i)
                    
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            epoch_duration = time.time() - epoch_time
            print("Epoch: {} cost time: {}".format(epoch + 1, epoch_duration))
            
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # Log epoch-level metrics
            epoch_metrics = {
                'epoch_train_loss': train_loss,
                'epoch_val_loss': vali_loss,
                'epoch_test_loss': test_loss,
                'epoch_duration': epoch_duration,
                'epoch': epoch + 1,
                'current_lr': model_optim.param_groups[0]['lr']
            }
            
            # Add scheduler-specific metrics
            if hasattr(scheduler, 'get_last_lr'):
                epoch_metrics['scheduler_lr'] = scheduler.get_last_lr()[0]
                
            wandb.log(epoch_metrics)
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                wandb.log({
                    'early_stopping_epoch': epoch + 1,
                    'best_val_loss': early_stopping.val_loss_min,
                    'training_completed': False
                })
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # Log training completion
        if not early_stopping.early_stop:
            wandb.log({'training_completed': True, 'final_epoch': self.args.train_epochs})

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        # Save model to wandb
        wandb.save(best_model_path)
        
        return self.model

    def test(self, setting, test=0):
        # Initialize wandb for testing if not already initialized
        if not self.wandb_initialized:
            wandb.init(
                project=f"time-series-forecasting-{self.args.data}",
                name=f"{setting}_test",
                config=self.wandb_config,
                reinit=True
            )
            self.wandb_initialized = True
            
            # Update with model info if available
            if hasattr(self, 'model_info'):
                wandb.config.update(self.model_info)
        
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'EntroPE' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'EntroPE' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    
                    # Create and log visualization to wandb
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(gt, label='Ground Truth', alpha=0.8)
                    ax.plot(pd, label='Prediction', alpha=0.8)
                    ax.legend()
                    ax.set_title(f'Prediction vs Ground Truth - Batch {i}')
                    ax.set_xlabel('Time Steps')
                    ax.set_ylabel('Value')
                    
                    wandb.log({f'prediction_plot_batch_{i}': wandb.Image(fig)})
                    plt.close(fig)
                    
                    # Also save locally
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        # Ensure correlation is scalar where needed and also keep the vector
        corr_vec = np.asarray(corr).reshape(-1)           # (H,) e.g., (96,)
        corr_mean = float(np.mean(corr_vec))               # scalar for summaries
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        
        # Log test metrics to wandb with PatchTST context
        test_metrics = {
            'test_mse': mse,
            'test_mae': mae,
            'test_rmse': rmse,
            'test_mape': mape,
            'test_mspe': mspe,
            'test_rse': rse,
            'test_correlation_mean': corr_mean,
            'test_samples_count': len(preds),
            'prediction_horizon': self.args.pred_len,
            'input_sequence_length': self.args.seq_len
        }
        wandb.log(test_metrics)
        
        # Log summary table
        metrics_table = wandb.Table(
            columns=["Metric", "Value"],
            data=[
                ["MSE", mse],
                ["MAE", mae],
                ["RMSE", rmse],
                ["MAPE", mape],
                ["MSPE", mspe],
                ["RSE", rse],
                ["Correlation_mean", corr_mean]
            ]
        )
        wandb.log({"test_metrics_table": metrics_table})
        # Log per-horizon correlation as its own table (plottable)
        corr_table = wandb.Table(columns=["Horizon", "Correlation"])
        for h, v in enumerate(corr_vec, start=1):
            corr_table.add_data(h, float(v))
        wandb.log({"correlation_by_horizon": corr_table})
        
        # Create error distribution plot
        errors = preds - trues
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Error histogram
        ax1.hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax1.set_title('Prediction Error Distribution')
        ax1.set_xlabel('Error')
        ax1.set_ylabel('Frequency')
        
        # Scatter plot: predicted vs actual
        sample_size = min(1000, len(preds.flatten()))
        indices = np.random.choice(len(preds.flatten()), sample_size, replace=False)
        ax2.scatter(trues.flatten()[indices], preds.flatten()[indices], alpha=0.5)
        ax2.plot([trues.min(), trues.max()], [trues.min(), trues.max()], 'r--', lw=2)
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        ax2.set_title('Predicted vs Actual')
        
        plt.tight_layout()
        wandb.log({"error_analysis": wandb.Image(fig)})
        plt.close(fig)
        
        # Save files and log to wandb
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # Save predictions and upload to wandb
        pred_file = folder_path + 'pred.npy'
        np.save(pred_file, preds)
        wandb.save(pred_file)
        
        # Create and upload a table with sample predictions
        sample_indices = np.random.choice(len(preds), min(100, len(preds)), replace=False)
        pred_table_data = []
        for idx in sample_indices:
            pred_table_data.append([
                idx,
                float(np.mean(preds[idx])),
                float(np.mean(trues[idx])),
                float(np.mean(np.abs(preds[idx] - trues[idx])))
            ])
        
        pred_table = wandb.Table(
            columns=["Sample_ID", "Predicted_Mean", "Actual_Mean", "MAE"],
            data=pred_table_data
        )
        wandb.log({"sample_predictions": pred_table})
        
        # Finish wandb run
        wandb.finish()
        
        return

    def predict(self, setting, load=False):
        # Initialize wandb for prediction if not already initialized
        if not self.wandb_initialized:
            wandb.init(
                project=f"time-series-forecasting-{self.args.data}",
                name=f"{setting}_prediction",
                config=self.wandb_config,
                reinit=True
            )
            self.wandb_initialized = True
            
            # Update with model info if available
            if hasattr(self, 'model_info'):
                wandb.config.update(self.model_info)
        
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model or 'EntroPE' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model or 'EntroPE' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)
                
                # Log sample predictions periodically
                if i % 50 == 0 and i > 0:
                    sample_pred = pred[0, :, 0] if len(pred.shape) > 2 else pred[0]
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(sample_pred, label=f'Prediction Batch {i}')
                    ax.set_title(f'Sample Prediction - Batch {i}')
                    ax.set_xlabel('Time Steps')
                    ax.set_ylabel('Predicted Value')
                    ax.legend()
                    
                    wandb.log({f'sample_prediction_batch_{i}': wandb.Image(fig)})
                    plt.close(fig)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # Log prediction statistics with PatchTST-specific context
        pred_stats = {
            'prediction_mean': float(np.mean(preds)),
            'prediction_std': float(np.std(preds)),
            'prediction_min': float(np.min(preds)),
            'prediction_max': float(np.max(preds)),
            'num_predictions': len(preds),
            'prediction_shape': list(preds.shape),
            'model_type': self.args.model
        }
        
        # Add PatchTST-specific prediction stats
        if 'PatchTST' in self.args.model or 'TST' in self.args.model or 'EntroPE' in self.args.model:
            pred_stats.update({
                'patch_based_prediction': True,
                'effective_patch_size': getattr(self.args, 'patch_size', None),
                'max_patch_length': getattr(self.args, 'max_patch_length', None)
            })
        
        wandb.log(pred_stats)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        pred_file = folder_path + 'real_prediction.npy'
        np.save(pred_file, preds)
        
        # Upload predictions to wandb
        wandb.save(pred_file)
        
        # Create prediction analysis plots
        if len(preds) > 0:
            # Plot prediction distribution across time steps
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Time series of mean predictions
            mean_preds = np.mean(preds, axis=0)
            if len(mean_preds.shape) > 1:
                for feature_idx in range(min(3, mean_preds.shape[1])):
                    axes[0, 0].plot(mean_preds[:, feature_idx], 
                                   label=f'Feature {feature_idx}', alpha=0.7)
            else:
                axes[0, 0].plot(mean_preds, label='Mean Prediction', alpha=0.7)
            axes[0, 0].set_title('Mean Predictions Over Time')
            axes[0, 0].set_xlabel('Time Steps')
            axes[0, 0].set_ylabel('Predicted Value')
            axes[0, 0].legend()
            
            # Prediction variance over time
            var_preds = np.var(preds, axis=0)
            if len(var_preds.shape) > 1:
                for feature_idx in range(min(3, var_preds.shape[1])):
                    axes[0, 1].plot(var_preds[:, feature_idx], 
                                   label=f'Feature {feature_idx}', alpha=0.7)
            else:
                axes[0, 1].plot(var_preds, label='Prediction Variance', alpha=0.7)
            axes[0, 1].set_title('Prediction Variance Over Time')
            axes[0, 1].set_xlabel('Time Steps')
            axes[0, 1].set_ylabel('Variance')
            axes[0, 1].legend()
            
            # Distribution of all predictions
            axes[1, 0].hist(preds.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Distribution of All Predictions')
            axes[1, 0].set_xlabel('Predicted Value')
            axes[1, 0].set_ylabel('Frequency')
            
            # Sample predictions (first few samples)
            sample_size = min(5, len(preds))
            for i in range(sample_size):
                if len(preds[i].shape) > 1:
                    axes[1, 1].plot(preds[i, :, 0], alpha=0.6, label=f'Sample {i}')
                else:
                    axes[1, 1].plot(preds[i], alpha=0.6, label=f'Sample {i}')
            axes[1, 1].set_title('Sample Individual Predictions')
            axes[1, 1].set_xlabel('Time Steps')
            axes[1, 1].set_ylabel('Predicted Value')
            axes[1, 1].legend()
            
            plt.tight_layout()
            wandb.log({"prediction_analysis": wandb.Image(fig)})
            plt.close(fig)
        
        # Finish wandb run
        wandb.finish()

        return