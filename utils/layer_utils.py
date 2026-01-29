import torch


def fill_tokens(tokens, patch_size, fill_id):
    """Pad tokens to make sequence length divisible by patch_size."""
    batch_size, seq_len = tokens.shape
    if seq_len % patch_size == 0:
        return tokens
    else:
        remaining = patch_size - seq_len % patch_size
        final_padding = tokens.new(batch_size, remaining).fill_(fill_id)
        return torch.cat((tokens, final_padding), dim=1)
    
def get_entrope_input(
    tokens: torch.Tensor,
    enforce_patch_size_multiple: bool,
    nb_boe: int,
    patch_size: int,
    boe_id: int,
):
    """Prepare encoder and decoder tokens for EntroPE."""
    batch_size, seq_len = tokens.shape
    local_encoder_tokens = tokens
    local_decoder_tokens = tokens

    if nb_boe > 0:
        padded_patch = tokens.new(batch_size, nb_boe).fill_(boe_id)
        local_encoder_tokens = torch.cat((padded_patch, local_encoder_tokens), dim=1)

    if enforce_patch_size_multiple and local_encoder_tokens.shape[-1] % patch_size != 0:
        local_encoder_tokens = fill_tokens(local_encoder_tokens, patch_size, boe_id)

    return local_encoder_tokens, None, local_decoder_tokens

def patch_ids_from_lengths(patch_lengths, seq_len):
    """Generate patch IDs from patch lengths."""
    bs, num_patches = patch_lengths.shape
    cum_d = torch.cat(
        [
            torch.zeros(bs, 1, dtype=patch_lengths.dtype, device=patch_lengths.device),
            patch_lengths.cumsum(dim=-1),
        ],
        dim=-1,
    )
    patch_ids = (cum_d.unsqueeze(-1) <= torch.arange(seq_len, device=cum_d.device)).sum(
        dim=-2
    ) - 1
    assert not (
        torch.max(patch_ids) > patch_lengths.shape[-1] or torch.min(patch_ids) < 0
    ), f"Invalid patch_ids: max={torch.max(patch_ids)}, min={torch.min(patch_ids)}"
    return patch_ids

def cross_attn_mask(
    patch_ids,
    patch_lengths,
    N,
    patches_as_queries=False,
    cross_attn_k=1,
    window=None,
    block_mask=True,
):
    """Create cross-attention mask."""
    bs = patch_ids.shape[0]
    with torch.no_grad():
        cross_mask = create_patch_mask_from_ids(
            patch_ids,
            patch_lengths.shape[1],
            window=window,
            patches_as_queries=patches_as_queries,
        ).repeat_interleave(cross_attn_k, dim=1 if patches_as_queries else -1)
        
        q_len = patch_lengths.shape[1] * cross_attn_k if patches_as_queries else N
        kv_len = N if patches_as_queries else patch_lengths.shape[1] * cross_attn_k
        
        assert cross_mask.shape == (bs, q_len, kv_len)

        if block_mask:
            pass
        else:
            return torch.where(
                cross_mask, torch.tensor(0.0), torch.tensor(float("-inf"))
            ).unsqueeze(1)
        
def create_patch_mask_from_ids(
    patch_ids, num_patches, window=None, patches_as_queries=False
):
    """Create attention mask from patch IDs."""
    bs, seq_len = patch_ids.shape
    if not patches_as_queries:
        q_ids = patch_ids.unsqueeze(-1).expand(bs, seq_len, num_patches)
        kv_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(bs, seq_len, num_patches)
        )
    else:
        kv_ids = patch_ids.unsqueeze(1).expand(bs, num_patches, seq_len)
        q_ids = (
            torch.arange(num_patches, device=patch_ids.device)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(bs, num_patches, seq_len)
        )
    
    if window is None:
        mask = q_ids == kv_ids
    else:
        mask = (kv_ids <= q_ids) & (q_ids < kv_ids + window)
    return mask

def concat_downsample(h, patch_lengths, patch_size):
    # The assumption in this function is that seq_len = patch_size * num_patches.
    bs, seq_len, emb_dim = h.shape
    patch_end_ids = torch.cumsum(patch_lengths, dim=1)
    patch_ids = patch_end_ids.unsqueeze(-1) - torch.arange(patch_size, 0, -1).to(
        patch_end_ids.device
    )
    # Is clamp ok here?
    patch_ids = patch_ids.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, h.shape[-1])
    patch_ids = patch_ids.view(bs, -1, emb_dim)
    # after gather h.shape = [batch_size, seq_len, dim]
    h = torch.gather(h, 1, patch_ids)
    h = h.reshape(bs, patch_lengths.shape[1], patch_size * h.size(-1))
    return h

def patch_reduce(h, max_num_patches, reduction, patch_ids):
    """
    Reduce variable length patches to single embedding per patch
    Note: this works with variable number of patches for different sequences in the batch
    It handles variable length patches by assuming that patch_lengths will be 0 for any
    extra patches on the *right*. Since there can be a variable number of patches
    this function also return the number of patches for each sequence in the batch.
    Any embeddings on the right that are not allocated to a patch
    (i.e. if the sum(patch_lengths[i]) < seq_len for any i)
    will be sent to a dummy patch, which is trimmed before returning.
    """
    bs, seq_len, emb_dim = h.shape

    patch_ids = patch_ids.unsqueeze(-1).expand(-1, -1, h.shape[-1])

    reduced_embs = torch.zeros(
        (bs, max_num_patches, emb_dim), dtype=h.dtype, device=h.device
    )
    reduced_embs = reduced_embs.scatter_reduce(
        src=h,
        dim=1,
        index=patch_ids,
        reduce=reduction,
        include_self=False,
    )
    reduced_embs = reduced_embs[:, :max_num_patches, :]

    return reduced_embs

def pooling_downsample(h, max_num_patches, pooling_mode, patch_ids):
    cat = []
    if "avg" in pooling_mode or "mean" in pooling_mode:
        cat.append(patch_reduce(h, max_num_patches, "mean", patch_ids))
    if "min" in pooling_mode:
        cat.append(patch_reduce(h, max_num_patches, "amin", patch_ids))
    if "max" in pooling_mode:
        cat.append(patch_reduce(h, max_num_patches, "amax", patch_ids))
    assert len(cat) > 0
    h = torch.cat(cat, dim=-1)
    return h

def downsample(
    h,
    num_patches,
    patch_lengths=None,
    patch_ids=None,
    downsampling_by_pooling=None,
    patch_size=4,
):
    """
    Downsampling:
        a. concatenating embeddings in the patch
            Note: with dynamic patching, patch the last patch_size tokens.
        b. pooling embeddings in the patch
    """
    # input: h.shape = [batch_size, seq_len, dim]
    # input: pool h.shape = [batch_size, seq_len / patch_size, dim]
    # if we don't use the cros_attn, we pool so that we convert bytes rep to patch rep
    if downsampling_by_pooling is not None and len(downsampling_by_pooling) > 0:
        # By pooling
        max_num_patches = num_patches
        assert patch_ids is not None
        h = pooling_downsample(h, max_num_patches, downsampling_by_pooling, patch_ids)
       # print(f"Pooling downsample: {downsampling_by_pooling}, h = {h}")
        
    else:
        # TODO: remove this condition
        # By concatenating (fixed lengths patching)
        assert patch_lengths is not None
        h = concat_downsample(h, patch_lengths, patch_size)
    return h

def get_encoder_dim_token_emb(args):
    if args.dim_token is not None:
        return args.dim_token
    elif args.use_local_encoder_transformer:
        return args.dim_local_encoder
    else:
        return args.dim_global // args.patch_size
    
def get_decoder_dim_token_emb(args):
    if args.share_encoder_decoder_emb:
        return get_encoder_dim_token_emb(args)
    elif args.dim_token is not None:
        return args.dim_token
    else:
        return args.dim_local_decoder
    
def get_encoder_dim_token_emb(args):
    if args.dim_token is not None:
        return args.dim_token
    elif args.use_local_encoder_transformer:
        return args.dim_local_encoder
    else:
        return args.dim_global // args.patch_size


def get_encoder_dim_patch_emb(args):
    if args.cross_attn_encoder:
        if args.cross_attn_init_by_pooling:
            return args.dim_local_encoder
        else:
            return args.dim_global
    return None

def get_global_dim_patch_emb(args):
    dim_token_emb = get_encoder_dim_token_emb(args)
    if args.cross_attn_encoder:
        return dim_token_emb * args.cross_attn_k
    elif (
        args.downsampling_by_pooling is None
        or not args.downsampling_by_pooling
        or len(args.downsampling_by_pooling) == 0
    ):
        return dim_token_emb * args.patch_size
    else:
        return dim_token_emb * sum(
            [
                pooling in args.downsampling_by_pooling
                for pooling in ["avg", "min", "max"]
            ]
        )