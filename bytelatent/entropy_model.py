# # Copyright (c) Meta Platforms, Inc. and affiliates.
# import json
# import logging
# import os

# import torch

# # from bytelatent.transformer import LMTransformer, LMTransformerArgs
# from bytelatent.entropy_model_core import GPTConfig, GPT

# logger = logging.getLogger()


# def load_entropy_model(
#         entropy_model_checkpoint_dir, state_dict_path, device="cpu"):
#     with open(os.path.join(entropy_model_checkpoint_dir, "params.json")) as fr:
#         reloaded = json.loads(fr.read())
#     # print(reloaded)
#     torch.set_default_dtype(torch.bfloat16)
#     model_params = reloaded["entropy_model"]
#     logger.warning(
#         "Update checkpoint to load attn and sliding window args from checkpoint"
#     )
#     # print("Loading entropy model with params:", model_params)
#     entropy_model_args = GPTConfig(
#         n_layer=model_params["n_layer"],
#         n_head=model_params["n_head"],
#         n_embd=model_params["n_embd"],
#         dropout=model_params["dropout"],
#         bias=model_params["bias"],
#         vocab_size=model_params["vocab_size"],
#         block_size=model_params["block_size"]
#     )
#     entropy_model = GPT(entropy_model_args)

#     entropy_model.load_state_dict(torch.load(
#         state_dict_path, 
#         map_location=device, 
#         weights_only=True
#         )["model_state_dict"], strict=True)
    
#     entropy_model.to(device)
#     entropy_model = entropy_model.eval()
#     # no grads for the model:
#     for param in entropy_model.parameters():
#         param.requires_grad = False
#     return entropy_model, entropy_model_args