import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.train_utils import build_tokenizer, create_static_patch_lengths, build_dataloader

def evaluate_model(model, input_len, pred_len, test_loader, test_tokenizer, eval_batch_size, dataset_name, features, device='cuda'):
    """
    Evaluate the model with comprehensive metrics and visualizations
    """
    
    # Initialize lists to store metrics across batches
    all_mse = []
    all_mae = []
    all_predictions = []
    all_actuals = []
    batch_forecasts = []
    failed_batches = 0
    
    print(f"Starting evaluation with {len(test_loader)} batches...")
    
    # Progress bar for batches
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), 
                desc="Evaluating", unit="batch", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for i, (batch_x, batch_y, _, _) in pbar:
        try:
            # Move the batch to the same device as the model
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            x = batch_x.float().squeeze(-1)
            y = batch_y.float().squeeze(-1)
            
            # Tokenize input
            token_ids_start, _, tokenizer_state = test_tokenizer.context_input_transform(x.to('cpu'))
            target_token_ids, _ = test_tokenizer.label_input_transform(y.to('cpu'), tokenizer_state.to('cpu'))
            
            # Debug shapes on first batch
            if i == 0:
                print(f"\nDebug - First batch shapes:")
                print(f"  Input x: {x.shape}")
                print(f"  Target y: {y.shape}")
                print(f"  Token IDs start: {token_ids_start.shape}")
                print(f"  Target token IDs: {target_token_ids.shape}")
            
            with torch.no_grad():
                token_ids = token_ids_start
                forecast = torch.zeros((eval_batch_size, pred_len), dtype=torch.long).to(device)
                
                # Progress bar for prediction steps (only show for first few batches)
                pred_range = range(pred_len)
                if i < 3:  # Show detailed progress for first 3 batches
                    pred_range = tqdm(pred_range, desc=f"Batch {i} predictions", 
                                    leave=False, unit="step")
                
                for j in pred_range:
                    # Get predictions for the next token
                    logits, _ = model(token_ids.to(device), None)
                    pred_tokens = torch.argmax(logits[:, -1, :], dim=-1)
                    next_token = pred_tokens.unsqueeze(1)
                    
                    # Store the predicted token
                    forecast[:, j] = next_token.squeeze(-1)
                    
                    # Update token sequence for next iteration
                    all_tokens = torch.cat([token_ids.to(device), next_token], dim=1)
                    # Keep only the last input_len tokens to maintain context window
                    token_ids = all_tokens[:, -input_len:] 
                
                # Handle target tokens - check if we have enough
                if target_token_ids.shape[1] < pred_len:
                    print(f"Warning: Target tokens ({target_token_ids.shape[1]}) < pred_len ({pred_len})")
                    # Pad target tokens or truncate forecast
                    min_len = min(target_token_ids.shape[1], pred_len)
                    actual = target_token_ids[:, :min_len]
                    forecast_eval = forecast[:, :min_len]
                else:
                    actual = target_token_ids[:, :pred_len]
                    forecast_eval = forecast
                
                # Convert tokens back to values using inverse transform
                actual_values = test_tokenizer.output_transform(actual.to('cpu').unsqueeze(1), tokenizer_state.to('cpu'))
                forecast_values = test_tokenizer.output_transform(forecast_eval.to('cpu').unsqueeze(1), tokenizer_state.to('cpu'))
                
                # Remove sample dimension if exists
                if actual_values.dim() == 3 and actual_values.shape[1] == 1:
                    actual_values = actual_values.squeeze(1)
                if forecast_values.dim() == 3 and forecast_values.shape[1] == 1:
                    forecast_values = forecast_values.squeeze(1)
                
                # Final shape check
                if actual_values.shape != forecast_values.shape:
                    print(f"Shape mismatch in batch {i}: {actual_values.shape} vs {forecast_values.shape}")
                    # Align shapes
                    min_shape = [min(actual_values.shape[d], forecast_values.shape[d]) 
                               for d in range(len(actual_values.shape))]
                    if len(min_shape) == 2:
                        actual_values = actual_values[:min_shape[0], :min_shape[1]]
                        forecast_values = forecast_values[:min_shape[0], :min_shape[1]]
                    else:
                        actual_values = actual_values[:min_shape[0]]
                        forecast_values = forecast_values[:min_shape[0]]
                
                # Calculate MSE and MAE
                mse = torch.mean((actual_values - forecast_values) ** 2)
                mae = torch.mean(torch.abs(actual_values - forecast_values))
                
                # Store metrics
                all_mse.append(mse.item())
                all_mae.append(mae.item())
                
                # Store data for plotting
                all_predictions.extend(forecast_values.cpu().numpy().flatten())
                all_actuals.extend(actual_values.cpu().numpy().flatten())
                
                # Store sample forecasts for plotting
                if i < 5:
                    batch_forecasts.append({
                        'batch_id': i,
                        'actual': actual_values.cpu().numpy(),
                        'forecast': forecast_values.cpu().numpy(),
                        'input_seq': x.cpu().numpy()
                    })
                
                # Update progress bar description with current metrics
                pbar.set_postfix({
                    'MSE': f'{mse.item():.4f}',
                    'MAE': f'{mae.item():.4f}',
                    'Avg_MSE': f'{np.mean(all_mse):.4f}' if all_mse else 'N/A'
                })
                
        except Exception as e:
            failed_batches += 1
            print(f"\nError in batch {i}: {e}")
            pbar.set_postfix({
                'Failed': failed_batches,
                'Success': len(all_mse)
            })
            continue
    
    pbar.close()
    
    if not all_mse:
        print("❌ No batches processed successfully!")
        return None
    
    # Calculate final statistics
    avg_mse = np.mean(all_mse)
    avg_mae = np.mean(all_mae)
    std_mse = np.std(all_mse)
    std_mae = np.std(all_mae)
    
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name} | Features: {features}")
    print(f"Input Length: {input_len} | Prediction Length: {pred_len}")
    print(f"Successful Batches: {len(all_mse)}/{len(all_mse) + failed_batches}")
    print(f"\n📈 METRICS:")
    print(f"  Average MSE:  {avg_mse:.6f} ± {std_mse:.6f}")
    print(f"  Average MAE:  {avg_mae:.6f} ± {std_mae:.6f}")
    print(f"  RMSE:         {np.sqrt(avg_mse):.6f}")
    
    if all_actuals and all_predictions:
        correlation = np.corrcoef(all_actuals, all_predictions)[0, 1]
        r_squared = correlation ** 2
        mape = np.mean(np.abs((np.array(all_actuals) - np.array(all_predictions)) / np.array(all_actuals))) * 100
        
        print(f"  Correlation:  {correlation:.6f}")
        print(f"  R²:           {r_squared:.6f}")
        print(f"  MAPE:         {mape:.2f}%")
    
    # Create comprehensive visualizations
    create_evaluation_plots(all_actuals, all_predictions, all_mse, all_mae, 
                          batch_forecasts, dataset_name, features, input_len, pred_len)
    
    # Save detailed results
    results = {
        "dataset_info": {
            "dataset_name": dataset_name,
            "features": features,
            "input_len": input_len,
            "pred_len": pred_len,
            "eval_batch_size": eval_batch_size
        },
        "metrics": {
            "mse_per_batch": all_mse,
            "mae_per_batch": all_mae,
            "average_mse": float(avg_mse),
            "average_mae": float(avg_mae),
            "std_mse": float(std_mse),
            "std_mae": float(std_mae),
            "rmse": float(np.sqrt(avg_mse))
        },
        "summary": {
            "successful_batches": len(all_mse),
            "failed_batches": failed_batches,
            "total_predictions": len(all_predictions)
        }
    }
    
    if all_actuals and all_predictions:
        results["metrics"].update({
            "correlation": float(correlation),
            "r_squared": float(r_squared),
            "mape": float(mape)
        })
    
    filename = f"metrics_{dataset_name}_{features}_{input_len}_{pred_len}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n💾 Results saved to: {filename}")
    print(f"🎨 Plots saved to: evaluation_{dataset_name}_{features}_{input_len}_{pred_len}.png")
    
    return results

def create_evaluation_plots(all_actuals, all_predictions, all_mse, all_mae, 
                          batch_forecasts, dataset_name, features, input_len, pred_len):
    """
    Create comprehensive evaluation plots
    """
    if not all_actuals or not all_predictions:
        print("⚠️  Insufficient data for plotting")
        return
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Time Series Forecasting Evaluation\n'
                f'{dataset_name} | {features} features | Input: {input_len}, Pred: {pred_len}', 
                fontsize=16, fontweight='bold')
    
    # Plot 1: Actual vs Predicted scatter
    ax = axes[0, 0]
    ax.scatter(all_actuals, all_predictions, alpha=0.6, s=2, c='blue')
    min_val, max_val = min(all_actuals), max(all_actuals)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Actual vs Predicted')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add R² annotation
    if len(all_actuals) > 1:
        r_squared = np.corrcoef(all_actuals, all_predictions)[0, 1] ** 2
        ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Plot 2: Error distribution
    ax = axes[0, 1]
    errors = np.array(all_predictions) - np.array(all_actuals)
    ax.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(np.mean(errors), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Metrics per batch
    ax = axes[0, 2]
    batch_nums = range(len(all_mse))
    ax.plot(batch_nums, all_mse, 'o-', label='MSE', markersize=4, linewidth=1.5)
    ax.plot(batch_nums, all_mae, 's-', label='MAE', markersize=4, linewidth=1.5)
    ax.set_xlabel('Batch Number')
    ax.set_ylabel('Error Value')
    ax.set_title('Error Trends Across Batches')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plots 4-6: Sample forecasts
    for idx in range(min(3, len(batch_forecasts))):
        ax = axes[1, idx]
        sample_data = batch_forecasts[idx]
        
        # Get first sample from batch
        if len(sample_data['actual'].shape) > 1:
            actual_sample = sample_data['actual'][0]
            forecast_sample = sample_data['forecast'][0]
            input_sample = sample_data['input_seq'][0]
        else:
            actual_sample = sample_data['actual']
            forecast_sample = sample_data['forecast']
            input_sample = sample_data['input_seq']
        
        # Create time axis
        input_time = np.arange(len(input_sample))
        forecast_time = np.arange(len(input_sample), len(input_sample) + len(forecast_sample))
        
        # Plot sequences
        ax.plot(input_time, input_sample, 'b-', label='Input Sequence', linewidth=2.5)
        ax.plot(forecast_time, actual_sample, 'g-', label='Actual Future', linewidth=2.5)
        ax.plot(forecast_time, forecast_sample, 'r--', label='Predicted Future', linewidth=2.5)
        
        # Add separation line
        ax.axvline(len(input_sample)-0.5, color='black', linestyle=':', alpha=0.7, linewidth=2)
        
        ax.set_title(f'Sample Forecast #{idx+1}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Value')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'evaluation_{dataset_name}_{features}_{input_len}_{pred_len}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def evaluation(model, dataset_name, features, quant_range, vocab_size, input_len, pred_len, eval_batch_size, device='cuda'):
    """
    Main evaluation function with progress tracking
    """
    print(f"🚀 Starting evaluation...")
    print(f"📝 Configuration:")
    print(f"   Dataset: {dataset_name}")
    print(f"   Features: {features}")
    print(f"   Input length: {input_len}")
    print(f"   Prediction length: {pred_len}")
    print(f"   Batch size: {eval_batch_size}")
    print(f"   Device: {device}")
    
    model.eval()
    
    # Build tokenizer
    print("🔧 Building tokenizer...")
    test_tokenizer = build_tokenizer(quant_range, vocab_size, input_len, pred_len)
    
    # Build data loader
    print("📁 Loading test data...")
    print(f"   Dataset: {dataset_name} | Features: {features} | Input Length: {input_len} | Prediction Length: {pred_len} | Batch Size: {eval_batch_size}")
    _, test_loader = build_dataloader(
        dataset_name, 
        features=features,
        seq_len=input_len, 
        label_len=0,
        pred_len=pred_len, 
        flag='test',
        batch_size=eval_batch_size,
        pretrain=True
    )
    
    print(f"📊 Test dataset loaded: {len(test_loader)} batches")
    
    # Create patch lengths
    # print("🔧 Creating patch lengths...")
    # patch_lengths = create_static_patch_lengths(batch_size=eval_batch_size, seq_len=input_len)
    
    # Run evaluation
    results = evaluate_model(
        model, input_len, pred_len, test_loader, test_tokenizer, 
        eval_batch_size, dataset_name, features, device
    )
    
    print("✅ Evaluation completed!")
    return results