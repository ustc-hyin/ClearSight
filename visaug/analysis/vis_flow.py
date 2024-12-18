import argparse
import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image
import math
import random
import numpy as np

from AttnAdapter import AttnAdapter, saliency_compute, attention_compute
import torch.nn.functional as F

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answers-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Change the attention module 
    for layer in model.model.layers:
        attn_adap = AttnAdapter(layer.self_attn.config)
        attn_adap.load_state_dict(layer.self_attn.state_dict())
        attn_adap = attn_adap.half().cuda()
        layer.self_attn = attn_adap

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    questions = random.sample(questions, 50)

    results = []

    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        qs = qs + " Please just answer yes or no."

        label = line['label']
        label = tokenizer.convert_tokens_to_ids(label)
        label = torch.tensor(label, dtype=torch.int64)
        label = label.cuda()

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        model.zero_grad()
        output_ids = model.forward(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            use_cache=True,
            output_attentions=True,
            return_dict=True,
        )

        pred_logit = output_ids['logits'][:,-1,:].squeeze(0)
        loss = F.cross_entropy(pred_logit, label)
        loss.backward()

        img_flow, layers_attn = [], []
        for idx_layer, layer in enumerate(model.model.layers):
            # compute saliency scores & attention weights
            attn_grad = layer.self_attn.attn_map.grad.detach().clone().cpu()
            attn_score = output_ids['attentions'][idx_layer].detach().clone().cpu()
            saliency = torch.abs(attn_grad * attn_score)

            img_saliency = saliency_compute(saliency)
            attn_props = attention_compute(attn_score)

            img_flow.append(img_saliency)
            layers_attn.append(attn_props)

        results.append((img_flow, layers_attn))
        
    torch.save(results, args.answers_file)