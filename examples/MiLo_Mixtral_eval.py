
import torch
import sys
sys.path.append("/u/bhuang4/mixtral_offloading/MiLo_official")
from MiLo.core.quantize import *
from MiLo.models.hf.mixtral import MixtralMiLo as AutoMiLoHFModel
from MiLo.engine.hf import AutoTokenizer
import gc

sys.path.append('/u/bhuang4/mixtral_offloading')
sys.path.append('/u/bhuang4/mixtral_offloading/evaluation')

from evaluation.eval_perplexity import eval_perplexity
import time
torch.cuda.empty_cache()
gc.collect()
import argparse
from tqdm import tqdm

cache_path     = '/scratch/bcjw/bhuang4/cache'
device         = 'cuda:0'



def main():
   
    model_id = "mistralai/Mixtral-8x7B-v0.1" 
    ranks = {'self_attn': 32,'experts':32}
    model_path = "/scratch/bcjw/bhuang4/MiLo_official_test/Mxitral_Moe_test-iter2"
    model = AutoMiLoHFModel.from_quantized(model_path,
                                          LoRC_dtype = "int3",)

    #                                       )
    tokenizer    = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path,trust_remote_code=True)
    AutoMiLoHFModel.dequantize_UV_for_inference(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token =tokenizer.eos_token
    # prompt1 = "Write an essay about large language models."
    # warm_inputs = tokenizer(
    #         [prompt1],
    #         padding=True,
    #         add_special_tokens=True,
    #         return_tensors="pt",
    #     ).to(device)

    # print("doing warm up")
    # _ = model.generate(**warm_inputs, max_new_tokens=512, do_sample=True)
    # print("finish warm up")

    begin = time.time()
    eval_perplexity(model,tokenizer,f"wikitext2_perplexity_result.pickle",save_flag=0)
    end = time.time()
    print(f"taking {end - begin}")
    return

if __name__ == "__main__":
    main()




