import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LLM_pipline():
    def __init__(self, model_id="/path-to-m3d/M3D-LaMed-Phi-3-4B/", max_new_tokens=512, device_map="auto", dtype = torch.bfloat16):
        self.max_new_tokens = max_new_tokens
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            # _attn_implementation='flash_attention_2'
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            model_max_length=512,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )
        self.dtype = dtype
        
    def forward(self, qs, image_tensor=None, use_image=False, max_new_tokens=512):
        device_curr = self.model.device
        if image_tensor!=None and image_tensor.sum()!=0 and use_image:
            # qs = DEFAULT_IMAGE_TOKEN + qs
            qs = qs
            proj_out_num = 256
            image_tokens = "<im_patch>" * proj_out_num
            input_txt = image_tokens + ' ' + qs
            image_tensor = image_tensor.to(dtype=self.dtype, device=device_curr)
        else:
            qs = qs
            input_txt = qs
        
        input_id = self.tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device_curr)
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_id,
                images=image_tensor,
                # image_sizes=[image_tensor.size],
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                num_beams=1,
                # no_repeat_ngram_size=3,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True
                )
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    
