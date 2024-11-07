import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LLM_pipline():
    def __init__(self, model_id="/path-to-lm3/LLAMA3_8b_Instruct", max_new_tokens=512, device_map="auto"):
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )

        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
    def forward(self, qs):
        messages = [
            {"role": "system", "content": "You are a medical AI assistant, good at extracting information from medical reports and responding rigorously as required!"},
            {"role": "user", "content": qs},
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)
    
    
