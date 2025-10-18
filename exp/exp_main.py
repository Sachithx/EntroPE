from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import EntroPE
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
import wandb

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        self.wandb_config = self._build_wandb_config(args)
        self.wandb_initialized = False
        super(Exp_Main, self).__init__(args)

    def _build_wandb_config(self, args):
        """Build wandb configuration from args"""
        config = {
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
        }
        
        # Add optional parameters if present
        optional_params = ['devices', 'random_seed', 'enc_in', 'lradj', 'pct_start', 
                          'dropout', 'multiple_of', 'des', 'itr']
        for param in optional_params:
            if hasattr(args, param):
                config[param] = getattr(args, param)
        
        # Add EntroPE-specific parameters
        if 'EntroPE' in args.model:
            entrope_params = {
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
            config.update({k: v for k, v in entrope_params.items() if v is not None})
        
        return config

    def _build_model(self):
        model_dict = {
            'EntroPE': EntroPE
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        # Store model info for later logging
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.model_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        }
        
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

    def _is_simple_model(self):
        """Check if model uses simplified forward pass"""
        return 'Linear' in self.args.model or 'EntroPE' in self.args.model

    def _forward_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        """Forward pass handling both simple and complex models"""
        if self._is_simple_model():
            return self.model(batch_x)
        else:
            if self.args.output_attention:
                return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Extract predictions
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
        # Initialize wandb
        wandb.init(
            project=f"time-series-forecasting-{self.args.data_path}",
            name=setting,
            config=self.wandb_config,
            reinit=True
        )
        self.wandb_initialized = True
        
        if hasattr(self, 'model_info'):
            wandb.config.update(self.model_info)
        
        wandb.watch(self.model, log='all', log_freq=100)
        
        # Get data loaders
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Setup training
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate
        )

        # Log dataset info
        dataset_info = {
            'train_samples': len(train_data),
            'val_samples': len(vali_data),
            'test_samples': len(test_data),
            'train_steps_per_epoch': train_steps,
        }
        if hasattr(self.args, 'patching_batch_size'):
            dataset_info['patching_batch_size'] = self.args.patching_batch_size
        wandb.log(dataset_info)

        # Training loop
        time_now = time.time()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Log FLOPs on first batch
                if epoch == 0 and i == 0:
                    try:
                        from ptflops import get_model_complexity_info
                        with torch.cuda.device(0):
                            macs, params = get_model_complexity_info(
                                self.model.cuda(), 
                                (batch_x.shape[1], batch_x.shape[2]), 
                                as_strings=True, 
                                print_per_layer_stat=False
                            )
                            print(f'Computational complexity: {macs}')
                            print(f'Number of parameters: {params}')
                    except:
                        pass

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())

                # Log batch metrics
                if i % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    
                    wandb.log({
                        'batch_loss': loss.item(),
                        'learning_rate': model_optim.param_groups[0]['lr'],
                        'training_speed_s_per_iter': speed,
                        'estimated_time_left_s': left_time,
                    }, step=epoch * train_steps + i)
                    
                    iter_count = 0
                    time_now = time.time()

                # Backward pass
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

            # Epoch evaluation
            epoch_duration = time.time() - epoch_time
            print(f"Epoch: {epoch + 1} cost time: {epoch_duration}")
            
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "
                  f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            
            wandb.log({
                'epoch_train_loss': train_loss,
                'epoch_val_loss': vali_loss,
                'epoch_test_loss': test_loss,
                'epoch_duration': epoch_duration,
                'epoch': epoch + 1,
            })
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                wandb.log({
                    'early_stopping_epoch': epoch + 1,
                    'best_val_loss': early_stopping.val_loss_min,
                })
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        wandb.save(best_model_path)
        
        return self.model

    def test(self, setting, test=0):
        if not self.wandb_initialized:
            wandb.init(
                project=f"time-series-forecasting-{self.args.data}",
                name=f"{setting}_test",
                config=self.wandb_config,
                reinit=True
            )
            self.wandb_initialized = True
            
            if hasattr(self, 'model_info'):
                wandb.config.update(self.model_info)
        
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('Loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            )

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)
                
                # Visualize predictions
                if i % 20 == 0:
                    input_data = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input_data[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input_data[0, :, -1], outputs[0, :, -1]), axis=0)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(gt, label='Ground Truth', alpha=0.8)
                    ax.plot(pd, label='Prediction', alpha=0.8)
                    ax.legend()
                    ax.set_title(f'Prediction vs Ground Truth - Batch {i}')
                    
                    wandb.log({f'prediction_plot_batch_{i}': wandb.Image(fig)})
                    plt.close(fig)
                    
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # Calculate metrics
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        corr_mean = float(np.mean(corr))
        
        print(f'MSE: {mse}, MAE: {mae}, RSE: {rse}')
        
        # Log test metrics
        test_metrics = {
            'test_mse': mse,
            'test_mae': mae,
            'test_rmse': rmse,
            'test_mape': mape,
            'test_mspe': mspe,
            'test_rse': rse,
            'test_correlation_mean': corr_mean,
        }
        wandb.log(test_metrics)
        
        # Create visualization
        errors = preds - trues
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.hist(errors.flatten(), bins=50, alpha=0.7, edgecolor='black')
        ax1.set_title('Prediction Error Distribution')
        ax1.set_xlabel('Error')
        ax1.set_ylabel('Frequency')
        
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
        
        # Save results
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        
        with open("result.txt", 'a') as f:
            f.write(f"{setting}\n")
            f.write(f'MSE: {mse}, MAE: {mae}, RSE: {rse}\n\n')

        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        
        wandb.finish()
        return

    def predict(self, setting, load=False):
        if not self.wandb_initialized:
            wandb.init(
                project=f"time-series-forecasting-{self.args.data}",
                name=f"{setting}_prediction",
                config=self.wandb_config,
                reinit=True
            )
            self.wandb_initialized = True
            
            if hasattr(self, 'model_info'):
                wandb.config.update(self.model_info)
        
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            self.model.load_state_dict(torch.load(os.path.join(path, 'checkpoint.pth')))

        preds = []
        self.model.eval()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self._forward_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            
                preds.append(outputs.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # Log prediction statistics
        wandb.log({
            'prediction_mean': float(np.mean(preds)),
            'prediction_std': float(np.std(preds)),
            'num_predictions': len(preds),
        })

        # Save results
        folder_path = './results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)
        
        wandb.finish()
        return