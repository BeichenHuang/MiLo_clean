import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
sys.path.append("/u/bhuang4/mixtral_offloading/MiLo_official")
from MiLo.models.hf.mixtral import MixtralMiLo as AutoMiLoHFModel
from MiLo.core.quantize import *

iteration = 2
model_id       = "mistralai/Mixtral-8x7B-v0.1" 
mytoken       = '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'
compute_dtype = torch.float16
device        = "cuda"

fp16_model_path = "/scratch/bcjw/bhuang4/cache/models--mistralai--Mixtral-8x7B-v0.1/snapshots/ffe1a706bacbd5abddc5ff99432ee38f7e0662fb"
quant_model_path = '/scratch/bcjw/bhuang4/MiLo_official_test/Mxitral_Moe_test'

tokenizer = AutoTokenizer.from_pretrained(model_id) 

quant_config = BaseQuantizeConfig(nbits=3, group_size=64, quant_scale=False, quant_zero=False,axis=1) 


ranks = {'self_attn': 512}
# kurtosis_rank = {}
# with open('/u/bhuang4/mixtral_offloading/HQQ_LoRC/kurtosis_deepseek.txt', 'r') as file:
#     lines = file.readlines()
# for line in lines:
#     # 以冒号分割每一行
#     parts = line.split(':')
    
#     text = parts[0].replace('.weight', '').strip()
#     if "self_attn" in text or "shared" in text or "layers.0.mlp" in text: 
#         # rank = 512
#         continue
#     else:
#         number = parts[1].strip()
#         kurtosis = round(float(number))
#         if kurtosis > 1: 
#             rank = 716 
#         else:
#             rank = 0
#         # rank = 0
#         ranks[text] = rank
#     kurtosis_rank[text] = torch.tensor(rank)

# print(ranks)



save_path = f"{quant_model_path}-iter{iteration}"

print(f"generating model={save_path}-iter{iteration}")
model = AutoModelForCausalLM.from_pretrained(fp16_model_path,torch_dtype=compute_dtype, trust_remote_code=True)
AutoMiLoHFModel.quantize_model(model, quant_config=quant_config, 
                                compute_dtype=compute_dtype, device=device, 
                                 ranks=ranks, iters=iteration,lorc_dtype="int3")
if not os.path.exists(save_path):
    os.makedirs(save_path)
AutoMiLoHFModel.save_quantized(model, save_path)




