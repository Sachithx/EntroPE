import torch
from typing import Optional, Tuple, Any, Dict, Literal
from dataclasses import dataclass
from layers.Constants import Constants
from data_provider.data_factory import data_provider

@dataclass
class TokenizerConfig:

    tokenizer_class: str
    tokenizer_kwargs: Dict[str, Any]
    context_length: int
    prediction_length: int
    n_tokens: int
    n_special_tokens: int
    pad_token_id: int
    eos_token_id: int
    use_eos_token: bool
    model_type: Literal["causal", "seq2seq"]
    num_samples: int
    temperature: float
    top_k: int
    top_p: float

    def __post_init__(self):
        assert (
            self.pad_token_id < self.n_special_tokens
            and self.eos_token_id < self.n_special_tokens
        ), f"Special token id's must be smaller than {self.n_special_tokens=}"

class EntroPETokenizer:

    def context_input_transform(
        self,
        context: torch.Tensor,
    ) -> Tuple:

        raise NotImplementedError()

    def label_input_transform(self, label: torch.Tensor, tokenizer_state: Any) -> Tuple:

        raise NotImplementedError()

    def output_transform(
        self, samples: torch.Tensor, tokenizer_state: Any
    ) -> torch.Tensor:

        raise NotImplementedError()

class MeanScaleUniformBins(EntroPETokenizer):
    def __init__(
        self, low_limit: float, high_limit: float, config: TokenizerConfig
    ) -> None:
        self.config = config
        self.centers = torch.linspace(
            low_limit,
            high_limit,
            config.n_tokens - config.n_special_tokens - 1,
        )
        self.boundaries = torch.concat(
            (
                torch.tensor([-1e20], device=self.centers.device),
                (self.centers[1:] + self.centers[:-1]) / 2,
                torch.tensor([1e20], device=self.centers.device),
            )
        )

    def _input_transform(
        self, context: torch.Tensor, scale: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = context.to(dtype=torch.float32)
        attention_mask = ~torch.isnan(context)

        if scale is None:
            scale = torch.nansum(
                torch.abs(context) * attention_mask, dim=-1
            ) / torch.nansum(attention_mask, dim=-1)
            scale[~(scale > 0)] = 1.0

        scaled_context = context / scale.unsqueeze(dim=-1)
        token_ids = (
            torch.bucketize(
                input=scaled_context,
                boundaries=self.boundaries,
                # buckets are open to the right, see:
                # https://pytorch.org/docs/2.1/generated/torch.bucketize.html#torch-bucketize
                right=True,
            )
            + self.config.n_special_tokens
        )

        token_ids.clamp_(0, self.config.n_tokens - 1)

        token_ids[~attention_mask] = self.config.pad_token_id

        return token_ids, attention_mask, scale

    def _append_eos_token(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = token_ids.shape[0]
        eos_tokens = torch.full((batch_size, 1), fill_value=self.config.eos_token_id)
        token_ids = torch.concat((token_ids, eos_tokens), dim=1)
        eos_mask = torch.full((batch_size, 1), fill_value=True)
        attention_mask = torch.concat((attention_mask, eos_mask), dim=1)

        return token_ids, attention_mask

    def context_input_transform(
        self, context: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = context.shape[-1]

        if length > self.config.context_length:
            context = context[..., -self.config.context_length :]

        token_ids, attention_mask, scale = self._input_transform(context=context)

        if self.config.use_eos_token and self.config.model_type == "seq2seq":
            token_ids, attention_mask = self._append_eos_token(
                token_ids=token_ids, attention_mask=attention_mask
            )

        return token_ids, attention_mask, scale

    def label_input_transform(
        self, label: torch.Tensor, scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        length = label.shape[-1]

        assert length == self.config.prediction_length
        token_ids, attention_mask, _ = self._input_transform(context=label, scale=scale)

        if self.config.use_eos_token:
            token_ids, attention_mask = self._append_eos_token(
                token_ids=token_ids, attention_mask=attention_mask
            )

        return token_ids, attention_mask

    def output_transform(
        self, samples: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        scale_unsqueezed = scale.unsqueeze(-1).unsqueeze(-1)
        indices = torch.clamp(
            samples - self.config.n_special_tokens - 1,
            min=0,
            max=len(self.centers) - 1,
        )
        return self.centers[indices] * scale_unsqueezed
    

def build_tokenizer(configs):
    """
    Build a tokenizer that maps byte values to tokens.
    """
    _, data_loader = data_provider(configs, "train")
    _, quant_range_info = find_quant_range(data_loader)
    low_limit = quant_range_info["q_low"]
    high_limit = quant_range_info["q_high"]

    tokenizer_config = TokenizerConfig(
        tokenizer_class='MeanScaleUniformBins',
        tokenizer_kwargs={'low_limit': low_limit, 'high_limit': high_limit},
        context_length=configs.seq_len,
        prediction_length=configs.seq_len,   
        n_tokens=configs.vocab_size,
        n_special_tokens=4,
        pad_token_id=Constants.PAD_ID,
        eos_token_id=Constants.EOS_ID,
        use_eos_token=False,
        model_type='causal',
        num_samples=1,
        temperature=1.0,
        top_k=50,
        top_p=1.0
    )

    tokenizer = MeanScaleUniformBins(low_limit, high_limit, tokenizer_config)
    
    # Store original tensors on CPU for multi-GPU compatibility
    cpu_boundaries = tokenizer.boundaries.cpu()
    cpu_centers = tokenizer.centers.cpu()
    
    def patched_input_transform(context, scale=None):
        context = context.to(dtype=torch.float32)
        attention_mask = ~torch.isnan(context)
        
        # Create device-specific tensors for this call (don't modify original)
        device = context.device
        boundaries = cpu_boundaries.to(device)
        centers = cpu_centers.to(device)
        
        # Continue with original logic
        if scale is None:
            scale = torch.nansum(
                torch.abs(context) * attention_mask, dim=-1
            ) / torch.nansum(attention_mask, dim=-1)
            scale[~(scale > 0)] = 1.0

        scaled_context = context / scale.unsqueeze(dim=-1)
        token_ids = (
            torch.bucketize(
                input=scaled_context,
                boundaries=boundaries,  # Use local boundaries
                right=True,
            )
            + tokenizer.config.n_special_tokens
        )

        token_ids.clamp_(0, tokenizer.config.n_tokens - 1)
        token_ids[~attention_mask] = tokenizer.config.pad_token_id

        return token_ids, attention_mask, scale
    
    def patched_output_transform(samples, scale):
        device = samples.device
        centers = cpu_centers.to(device)
        
        scale_unsqueezed = scale.unsqueeze(-1).unsqueeze(-1)
        indices = torch.clamp(
            samples - tokenizer.config.n_special_tokens - 1,
            min=0,
            max=len(centers) - 1,
        )
        return centers[indices] * scale_unsqueezed
    
    # Replace the methods
    tokenizer._input_transform = patched_input_transform
    tokenizer.output_transform = patched_output_transform
    
    return tokenizer

def find_quant_range(data_loader, epsilon=0.05, max_samples=10000000):
    """
    Automatically determine the quantization range from the training data.
    
    Args:
        data_loader: DataLoader containing the training data
        epsilon: Coverage parameter (default: 0.005 for 99.5% coverage)
        max_samples: Maximum number of values to use for quantile computation
    
    Returns:
        quant_range: The symmetric quantization radius R
        quant_range_info: Dictionary containing statistics
    """
    # Collect values with limit
    all_values = []
    total_samples = 0
    collected_values = 0
    total_values = 0
    
    with torch.no_grad():
        for batch_x, batch_y, _, _ in data_loader:
            x = batch_x.float().squeeze(-1)
            x = x.permute(0, 2, 1)
            bs, nvars, seq_len = x.shape
            x = x.reshape(bs * nvars, seq_len)
            
            batch_flat = x.flatten().cpu()
            total_samples += x.shape[0]
            total_values += batch_flat.numel()
            
            # Only collect up to max_samples
            if collected_values < max_samples:
                remaining = max_samples - collected_values
                to_collect = min(remaining, batch_flat.numel())
                all_values.append(batch_flat[:to_collect])
                collected_values += to_collect
    
    # Concatenate collected values
    X_train = torch.cat(all_values, dim=0)
    
    # Compute quantiles
    q_low = torch.quantile(X_train, epsilon / 2)
    q_high = torch.quantile(X_train, 1 - epsilon / 2)
    
    # Compute symmetric quantization radius
    R = max(abs(q_low.item()), abs(q_high.item()))
    
    # Coverage statistics
    within_range = ((X_train >= -R) & (X_train <= R)).float().mean().item() * 100
    clipped = (X_train < -R).sum().item() + (X_train > R).sum().item()
    
    quant_range_info = {
        'R': R,
        'epsilon': epsilon,
        'q_low': q_low.item(),
        'q_high': q_high.item(),
        'interval_min': -R,
        'interval_max': R,
        'within_range_percent': within_range,
        'clipped_count': int(clipped),
        'clipped_percent': clipped / X_train.numel() * 100,
        'total_samples': total_samples,
        'total_values': total_values,
        'sampled_values': X_train.numel(),
    }
    
    return R, quant_range_info