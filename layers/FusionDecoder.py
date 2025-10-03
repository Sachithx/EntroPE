import torch
from torch import nn
RMSNorm = nn.RMSNorm

class LocalDecoder(LocalModelBase):
    def __init__(self, args: LocalModelArgs):
        super().__init__(args)

        # Model configuration flags
        self.cross_attn_decoder = args.cross_attn_decoder
        self.cross_attn_all_layers_decoder = args.cross_attn_all_layers_decoder
        self.cross_attn_init_by_pooling = args.cross_attn_init_by_pooling
        self.cross_attn_nheads = args.cross_attn_nheads

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        if self.cross_attn_decoder:
            self.cross_attn_layers = torch.nn.ModuleList()
            layers_to_add = args.n_layers if self.cross_attn_all_layers_decoder else 1
            for _ in range(layers_to_add):
                self.cross_attn_layers.append(
                    CrossAttention(
                        dim=self.dim,
                        head_dim=self.dim // self.cross_attn_nheads,
                        n_heads=self.cross_attn_nheads,
                        n_kv_heads=self.cross_attn_nheads,
                        norm_eps=args.norm_eps,
                    )
                )



    def forward(
        self,
        tokens: torch.Tensor,
        embeds: Optional[torch.Tensor],
        patch_embeds: Optional[torch.Tensor] = None,
        mask: Optional[Union["BlockMask", torch.Tensor, str]] = None,
        cross_mask: Optional[torch.Tensor] = None,
        cache: Optional[List[Tuple[torch.Tensor, torch.Tensor, int]]] = None,
    ):
        bs, seqlen = tokens.shape
        assert embeds is not None, "Embeddings must be provided"

        if mask is None:
            mask = create_causal_mask(
                seqlen,
                self.attn_impl,
                self.attn_bias_type,
                sliding_window=self.sliding_window,
                tokens=tokens,
                eos_id=self.eos_id,
            )

        h = embeds

        # if self.patch_embedding_projection is not None:
        #     # print(f"Patch embeddings shape before projection: {patch_embeds.shape if patch_embeds is not None else 'None'}")
        #     assert patch_embeds is not None, "Patch embeddings must be passed."
        #     patch_embeds = self.patch_embedding_projection(patch_embeds)
        #     if self.cross_attn_k is not None:
        #         patch_embeds = patch_embeds.reshape(
        #             bs, patch_embeds.shape[1] * self.cross_attn_k, self.dim
        #         )
        # if patch_embeds is not None and not self.cross_attn_decoder:
        #     h = h + patch_embeds

        # if self.use_rope:
        #     freqs_cis = self.rope(seqlen=seqlen)  
        # else: 
        #     # Suppose h: [bs, seq_len, dim]
        #     seq_len = h.size(1)
        
        device = h.device

        pos_ids = torch.arange(seqlen, device=device).unsqueeze(0)  # [1, seq_len]
        pos_emb = self.pos_embeddings(pos_ids)                       # [1, seq_len, dim]

        h = h + pos_emb


        h = F.dropout(h, p=self.dropout, training=self.training)
        for i, layer in enumerate(self.layers):
            if self.cross_attn_decoder and (
                i == 0 or self.cross_attn_all_layers_decoder
            ):
                # Use cross attention to extract info from patch_embeds into h
                h_cross = self.cross_attn_layers[i](
                    x=h,
                    kv=patch_embeds,
                    mask=cross_mask,
                )
                h = h + h_cross

            # h = layer(h, mask=mask, freq_cis=None, attn_impl=self.attn_impl)

        h_preds = self.norm(h)
        h_preds = F.dropout(h_preds, p=self.dropout, training=self.training)
        # h_preds = self.output(h_preds)
        h_preds = h_preds.float()
        return h_preds, cache
