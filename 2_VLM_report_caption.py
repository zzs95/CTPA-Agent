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
from organ_abnormality_list import finding_section_list
device = torch.device('cuda') 
dtype = torch.bfloat16

def ask_model(model, tokenizer, qs, image_tensor=None, use_image=True, max_new_tokens=512):
    if image_tensor!=None and image_tensor.sum()!=0 and use_image:
        proj_out_num = 256
        image_tokens = "<im_patch>" * proj_out_num
        input_txt = image_tokens + ' ' + qs
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

finding_write_prompt = """Please write a radiology report consists of findings for this medical image. """
finding_write_prompt = """Please write a radiology report consists of findings for this medical image. Please include the findings of Pulmonary arteries, Lungs and Airways, Pleura, Heart, Mediastinum and Hila, Chest Wall and Lower Neck, Chest Bones."""
finding_write_prompt = """Please write a radiology report consists of findings for this medical image. Please include the findings of Pulmonary arteries, Lungs and Airways, Pleura, Heart, Mediastinum and Hila, Chest Wall and Lower Neck, Chest Bones. For example: FINDINGS: Enteric tube. Pulmonary arteries:  There is subsegmental pulmonary embolism in both lower lobes. Lungs and Airways:  Emphysema. Mild pulmonary edema. Bibasilar atelectasis. Artifact limits detection for small nodules. 18 mm focal groundglass nodule in the left lower lobe on image 81. Pleura: Small bilateral pleural effusions. No pneumothorax. Heart and mediastinum: Mild cardiomegaly. Enlarged main pulmonary artery to 3.4 cm. Chest Wall and Lower Neck:  Normal. Bones: Status post left thoracotomy.  """

impression_write_prompt = """Based on the provided FINDINGS text, write an impression for a medical image report. """

def gen_text(args):
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
    # test_data_list = test_data_list.sort_values(['image'])
    accNum_list = [d[1]['image'].split('/')[1].split('_')[0] for d in  test_data_list.iterrows()]
    image_modal_list = [d[1]['image'].split('/')[1].split('_')[1] for d in  test_data_list.iterrows()]
    test_data_list['AccessionNumber_md5'] = accNum_list
    test_data_list['img_modal'] = image_modal_list
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file_image_f = open(answers_file, "w")
    # ans_file_image_i = open(answers_file.replace('image_findings', 'image_impression'), "w")
    for accNum in tqdm(pd.unique(accNum_list)):
        lines = test_data_list[test_data_list['AccessionNumber_md5'] == accNum].reset_index(drop=True)
        acc_findings_text = ''
        for index, line in lines.iterrows():
            image_path = line["image"]
            img_modal = image_path.split('/')[1].split('_')[1]
            image = np.load(os.path.join(args.data_root, image_path))  # nomalized 0-1, C,D,H,W
            ans = line["Findings Text"]
            image_tensor = torch.from_numpy(image)
            image_tensor = image_tensor.unsqueeze(0).to(dtype).to(device)
            qs = finding_write_prompt
            image_findings_text = ask_model(model, tokenizer, qs, image_tensor, use_image=True, max_new_tokens=2048)
        
            ans_file_image_f.write(json.dumps({
                "AccessionNumber_md5": accNum,
                "img_modal": img_modal,
                "findings_image_gt": line["Findings Text"],
                "findings_image_pred": image_findings_text,
                # "model_id": model_name,
                "metadata": {}}) + "\n")
            ans_file_image_f.flush()
            
            # qs = impression_write_prompt
            # image_impression_text = ask_model(model, tokenizer, qs, image_tensor, use_image=True, max_new_tokens=2048)
        
            # ans_file_image_i.write(json.dumps({
            #     "AccessionNumber_md5": accNum,
            #     "img_modal": img_modal,
            #     "impression_image_gt": line["Impression Text"],
            #     "impression_image_pred": image_impression_text,
            #     # "model_id": model_name,
            #     "metadata": {}}) + "\n")
            # ans_file_image_i.flush()
    ans_file_image_f.close()
    # ans_file_image_i.close()
    
if __name__ == "__main__":
    from eval_path import project_root, data_root, text_path, answer_path, model_path
    answer_path = './eval_output_m3d_original_vqa'
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=model_path)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--answers-file", type=str, default=answer_path+"/image_findings.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_llama_3")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--square_eval", type=bool, default=False)

    parser.add_argument("--data_root", type=str, default=data_root)
    parser.add_argument("--text_path", type=str, default=text_path)
    args = parser.parse_args()
    
    gen_text(args)
