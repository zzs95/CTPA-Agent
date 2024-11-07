import argparse
import json
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # debug
from rewrite_process_llama3 import LLM_pipline

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from organ_abnormality_list import finding_section_list
from study_writing_prompts import finding_rewrite_prompt, impression_write_prompt

device = torch.device('cuda') 
dtype = torch.bfloat16
DEFAULT_IMAGE_TOKEN = "<image>"

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

section_question_prompt = "What findings of <FIND_SEC> do you observe in this medical image?"


def gen_text(args):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map='cuda:0',
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
    model_LLM = LLM_pipline(model_id=lm3_model_path, max_new_tokens=512, device_map='auto') # if 'auto' too slow, change to 'cuda'
    # model_LLM = LLM_pipline(model_id=m3d_model_path, max_new_tokens=512, device_map='cuda:0')
    
    mode = 'test'
    test_data_list = pd.read_excel(os.path.join(args.text_path, "PE_ctpa_cap_"+mode+".xlsx"), index_col=False, usecols=['image', 'Findings Text', 'Impression Text'])
    accNum_list = [d[1]['image'].split('/')[1].split('_')[0] for d in  test_data_list.iterrows()]
    image_modal_list = [d[1]['image'].split('/')[1].split('_')[1] for d in  test_data_list.iterrows()]
    test_data_list['AccessionNumber_md5'] = accNum_list
    test_data_list['img_modal'] = image_modal_list
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file_study = open(answers_file, "w")
    ans_file_section = open(answers_file.replace('study_findingsimpression', 'section_findings'), "w") 
    ans_file_image = open(answers_file.replace('study_findingsimpression', 'image_findings'), "w")

    for accNum in tqdm(pd.unique(accNum_list)):            
        lines = test_data_list[test_data_list['AccessionNumber_md5'] == accNum].reset_index(drop=True)
        acc_findings_text = ''
        for index, line in lines.iterrows():
            image_path = line["image"]
            # accNum = image_path.split('/')[1].split('_')[0]
            img_modal = image_path.split('/')[1].split('_')[1]
            image = np.load(os.path.join(args.data_root, image_path))  # nomalized 0-1, C,D,H,W
            ans = line["Findings Text"]
            image_tensor = torch.from_numpy(image)
            image_tensor = image_tensor.unsqueeze(0).to(dtype).to(device)
            section_findings_dict = {}
            for section_name in finding_section_list:
                qs = section_question_prompt.replace('<FIND_SEC>', section_name)
                outputs = ask_model(model, tokenizer, qs, image_tensor, use_image=True, max_new_tokens=48)
                section_findings_dict[section_name] = outputs
                
                ans_file_section.write(json.dumps({
                            "AccessionNumber_md5": accNum,
                            "img_modal": img_modal,
                            "section_name": section_name,
                            "findings_gt": line["Findings Text"],
                            "findings_section_pred": outputs,
                            # "model_id": model_name,
                            }) + "\n")
                ans_file_section.flush()
                
            image_findings_text = 'FINDINGS '+ str(index) +':\n'
            for k in section_findings_dict.keys():
                image_findings_text += k + ': ' + section_findings_dict[k] + '\n'
            acc_findings_text += image_findings_text + '\n'
            
            ans_file_image.write(json.dumps({
                "AccessionNumber_md5": accNum,
                "img_modal": img_modal,
                "findings_image_gt": line["Findings Text"],
                "findings_image_pred": image_findings_text,
                # "model_id": model_name,
                }) + "\n")
            ans_file_image.flush()
            print(ans, '\npred:', image_findings_text)
                    
        findings_text_rewrite = ''
        if finding_rewrite_prompt != '':
            qs = acc_findings_text + '\n' + finding_rewrite_prompt.replace('<f_NUM>', str(index+1))
            findings_text_rewrite = model_LLM.forward(qs)
        findings_text_rewrite = findings_text_rewrite.replace('\n\n' , '\n')
        
        impression_text_generate = ''
        if impression_write_prompt != '':
            qs = findings_text_rewrite + '\n' + impression_write_prompt
            impression_text_generate = model_LLM.forward(qs)
        
        print('*'*23)
        print(ans, '\npred:', findings_text_rewrite)
        ans_file_study.write(json.dumps({
                                    "AccessionNumber_md5": accNum,
                                    "findings_gt": line["Findings Text"],
                                    "impression_gt": line["Impression Text"],
                                    "findings_images_pred": acc_findings_text,
                                    "findings_study_pred": findings_text_rewrite,
                                    "impression_study_pred": impression_text_generate,
                                    # "model_id": model_name,
                                    }) + "\n")
        ans_file_study.flush()
    ans_file_section.close()
    ans_file_image.close()
    ans_file_study.close()
        


if __name__ == "__main__":
    from eval_path import project_root, data_root, text_path, answer_path, model_path, lm3_model_path
    answer_path = './eval_output_organ_reports/'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, default=answer_path+"study_findingsimpression.jsonl")
    parser.add_argument("--conv-mode", type=str, default="plain")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--square_eval", type=bool, default=False)

    parser.add_argument("--data_root", type=str, default=data_root)
    parser.add_argument("--text_path", type=str, default=text_path)
    args = parser.parse_args()
    
    gen_text(args)

