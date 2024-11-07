import argparse
import json
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # debug
# import shortuuid
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from organ_abnormality_list import abnormality_list
device = torch.device('cuda') 
dtype = torch.bfloat16

def ask_model(model, tokenizer, qs, image_tensor=None, use_image=True, max_new_tokens=512):
    if image_tensor!=None and image_tensor.sum()!=0 and use_image:
        proj_out_num = 256
        image_tokens = "<im_patch>" * proj_out_num
        input_txt = image_tokens + qs
        image_tensor = image_tensor.to(dtype=dtype, device=device)
    else:
        qs = qs
        input_txt = qs
    
    input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_id,
            images=image_tensor,
            # image_sizes=[image_tensor.size],
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs

abnormality_question_prompt = "Is there any indication of <ABN> in this image?" + " (This is true or false question, please answer 'Yes.' or 'No.'.) "

def eval_vqa(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map='cpu',
        trust_remote_code=True,
        # _attn_implementation='flash_attention_2'
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    model.get_model().seg_module = None
    model.get_model().seg_projector = None
    
    model = model.to(dtype).to(device)
    mode = 'test'
    test_data_list = pd.read_excel(os.path.join(args.text_path, "PE_ctpa_cap_"+mode+".xlsx"), index_col=False, usecols=['image', 'Findings Text', 'Impression Text'])
    test_data_list = test_data_list.sort_values(['image'])
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i in tqdm(range(len(test_data_list))):
        line = test_data_list.iloc[i]
        image_path = line["image"]
        
        image = np.load(os.path.join(args.data_root, image_path))  # nomalized 0-1, C,D,H,W
        accNum = image_path.split('/')[1].split('_')[0]
        img_modal = image_path.split('/')[1].split('_')[1]
        
        image_tensor = torch.from_numpy(image)
        image_tensor = image_tensor.unsqueeze(0).to(dtype).cuda()
        for abn_name in abnormality_list:
            qs = abnormality_question_prompt.replace('<ABN>', abn_name)
            outputs = ask_model(model, tokenizer, qs, image_tensor, use_image=True, max_new_tokens=3)
            print(outputs)
            if 'Yes' in outputs:
                outputs = 'Yes.'
            else:
                outputs = 'No.'
            
            ans_file.write(json.dumps({
                                    "image_path": image_path,
                                    "AccessionNumber_md5": accNum,
                                    "img_modal": img_modal,
                                    "abnormality": abn_name,
                                    "question": qs,
                                    "answer_pred": outputs,
                                    "metadata": {}}) + "\n")
            ans_file.flush()
    ans_file.close()
        


if __name__ == "__main__":
    from eval_path import project_root, data_root, text_path, answer_path, model_path
    answer_path = './eval_output_m3d_original_vqa'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, default=answer_path+"/vqa_answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="plain")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--square_eval", type=bool, default=False)

    parser.add_argument("--data_root", type=str, default=data_root)
    parser.add_argument("--text_path", type=str, default=text_path)
    args = parser.parse_args()
    
    eval_vqa(args)

