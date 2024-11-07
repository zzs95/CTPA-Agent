from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LLM_pipline():
    def __init__(self, model_id="/path-to-phi3/Phi-3-mini-128k-instruct/", max_new_tokens=512, device_map="auto"):
        self.max_new_tokens = max_new_tokens
        import torch

        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map=device_map, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
            _attn_implementation='flash_attention_2'
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.pipe = pipeline( 
            "text-generation", 
            model=model, 
            tokenizer=tokenizer, 
        ) 
        self.generation_args = { 
            "max_new_tokens": max_new_tokens, 
            "return_full_text": False, 
            "temperature": 0.1, 
            "do_sample": False, 
        } 
        
    def forward(self, qs):
        messages = [
            {"role": "system", "content": "You are a medical AI assistant, good at extracting information from medical reports and responding rigorously as required!"},
            {"role": "user", "content": qs},
        ]
        
        output = self.pipe(messages, **self.generation_args) 
        return output[0]['generated_text']
    
    
