import warnings
warnings.filterwarnings("ignore")
import torch
from chronos import MeanScaleUniformBins, ChronosConfig


class Tokenizer:
    def __init__(self, configs):
        self.quant_range = configs.quant_range
        self.vocab_size = configs.vocab_size
        self.context_length = configs.seq_len
        self.prediction_length = configs.seq_len
        
        # Build the base tokenizer
        self.tokenizer = self._build_base_tokenizer()
        
        # Store boundaries and centers on CPU for multi-GPU compatibility
        self.cpu_boundaries = self.tokenizer.boundaries.cpu()
        self.cpu_centers = self.tokenizer.centers.cpu()
    
    def _build_base_tokenizer(self):
        """
        Build the base tokenizer configuration.
        """
        low_limit = -1 * self.quant_range
        high_limit = self.quant_range

        tokenizer_config = ChronosConfig(
            tokenizer_class='MeanScaleUniformBins',
            tokenizer_kwargs={'low_limit': low_limit, 'high_limit': high_limit},
            context_length=self.context_length,
            prediction_length=self.prediction_length,   
            n_tokens=self.vocab_size,
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

        return MeanScaleUniformBins(low_limit, high_limit, tokenizer_config)
        
    def input_transform(self, context, scale=None):
        """
        Transform input context to token IDs with proper device handling.
        """
        if not isinstance(context, torch.Tensor):
            context = torch.tensor(context)
            
        context = context.to(dtype=torch.float32)
        attention_mask = ~torch.isnan(context)
        
        # Ensure we have valid data
        if not attention_mask.any():
            raise ValueError("All values in context are NaN")
        
        # Move boundaries to the same device as context
        device = context.device
        boundaries = self.cpu_boundaries.to(device)
        
        # Calculate scale if not provided
        if scale is None:
            scale = self._calculate_scale(context, attention_mask)

        # Scale the context
        scaled_context = context / scale.unsqueeze(dim=-1)
        
        # Convert to token IDs
        token_ids = (
            torch.bucketize(
                input=scaled_context,
                boundaries=boundaries,
                right=True,
            )
            + self.tokenizer.config.n_special_tokens
        )

        # Clamp to valid range and apply padding
        token_ids.clamp_(0, self.tokenizer.config.n_tokens - 1)
        token_ids[~attention_mask] = self.tokenizer.config.pad_token_id

        return token_ids, attention_mask, scale
    
    def _calculate_scale(self, context, attention_mask):
        """
        Calculate scale for normalization with proper error handling.
        """
        scale = torch.nansum(
            torch.abs(context) * attention_mask, dim=-1
        ) / torch.nansum(attention_mask, dim=-1)
        
        # Handle edge cases where scale might be 0 or invalid
        invalid_scale_mask = ~(scale > 0)
        scale[invalid_scale_mask] = 1.0
        
        return scale
        
    def output_transform(self, samples, scale):
        """
        Transform token samples back to continuous values.
        """
        if not isinstance(samples, torch.Tensor):
            samples = torch.tensor(samples)
            
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        
        device = samples.device
        centers = self.cpu_centers.to(device)
        
        # Ensure scale has the right shape
        scale_unsqueezed = scale.unsqueeze(-1)
        if samples.dim() > scale_unsqueezed.dim():
            scale_unsqueezed = scale_unsqueezed.unsqueeze(-1)
        
        # Convert token indices to center indices
        indices = torch.clamp(
            samples - self.tokenizer.config.n_special_tokens - 1,
            min=0,
            max=len(centers) - 1,
        )
        
        return centers[indices] * scale_unsqueezed
    
    def encode(self, context, scale=None):
        """
        Convenience method for encoding.
        """
        return self.input_transform(context, scale)
    
    def decode(self, samples, scale):
        """
        Convenience method for decoding.
        """
        return self.output_transform(samples, scale)
    