import argparse
import torch
import os
import json
import ray
import time
import numpy as np
from tqdm import tqdm
import shortuuid
import fasteners

from transformers import AutoConfig
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.video_utils import VideoProcessor, merge_video_dict

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image
import math


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

@ray.remote(num_gpus=1)
def eval_model(questions, args):
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
    config = {}
    if args.lora_path is not None:
        config = AutoConfig.from_pretrained(args.lora_path)
        config = config.to_dict()
    elif args.overwrite_cfg:
        config.update({
            'tie_word_embeddings': False, 
            'use_cache': True, 
            "vocab_size": 151649
        })

    # Match the training dtype (bfloat16). fp16 inference can overflow LayerNorm
    # variance / Linear ops when bf16-trained weights are cast to fp16's narrower range.
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        overwrite_config=config,
        torch_dtype="bfloat16",
    )

    if args.lora_path is not None:
        from transformers import AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
        model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(model, args.lora_path, adapter_name="lora")
        model = model.merge_and_unload()

        # non_lora_trainables.bin was saved while the model was PEFT-wrapped, so
        # keys carry the full PEFT prefix "base_model.model.". After merge_and_unload
        # the live model is back to its raw module tree, so we strip that 17-char
        # prefix to make keys match. Cast to model dtype (bf16-trained).
        PEFT_PREFIX = "base_model.model."
        state_dict = torch.load(os.path.join(args.lora_path, 'non_lora_trainables.bin'), map_location="cpu")
        state_dict = {(k[len(PEFT_PREFIX):] if k.startswith(PEFT_PREFIX) else k): v for k, v in state_dict.items()}
        state_dict = {k: (v.to(model.dtype) if torch.is_floating_point(v) else v) for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"[non_lora_trainables] loaded {len(state_dict)} tensors; "
              f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        if msg.unexpected_keys:
            # If this is non-empty after the strip, the PEFT prefix didn't match —
            # likely a save-time format change. Surface it loudly.
            raise RuntimeError(
                f"[non_lora_trainables] {len(msg.unexpected_keys)} keys did not match the model "
                f"after stripping '{PEFT_PREFIX}'. First few: {msg.unexpected_keys[:5]}. "
                f"Trained weights would be silently ignored — aborting."
            )

    # Sanity check: if the trained ckpt expects JEPA features, the user must provide them.
    if getattr(model.config, "use_jepa_only", False) and args.jepa_feature_folder is None:
        raise ValueError(
            "Loaded checkpoint has use_jepa_only=True but --jepa-feature-folder was not provided. "
            "JEPA-only inference requires the same feature folder used at training time."
        )
    
    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "a")
    file_lock = fasteners.InterProcessLock(ans_file)

    video_processor = VideoProcessor(
        video_folder=args.video_folder,
        annotation_dir=args.embodiedscan_folder,
        frame_sampling_strategy=args.frame_sampling_strategy,
    )
    
    # Resume support: skip samples already in answer_file
    already_done = set()
    try:
        with open(answer_file, "r") as _f:
            for _line in _f:
                try:
                    already_done.add(json.loads(_line)["sample_id"])
                except Exception:
                    pass
        if already_done:
            print(f"[resume] {len(already_done)} samples already in {answer_file}, will skip them.")
    except FileNotFoundError:
        pass

    n_correct = 0
    for _i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        if idx in already_done:
            continue
        # Periodic cache cleanup to avoid slow OOM accumulation over thousands of samples.
        if _i and _i % 50 == 0:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        try:
            question_type = line["metadata"]["question_type"]
            dataset_name = line["metadata"]["dataset"]
            video_id = line["video"]

            gt = line["conversations"][1]["value"]
            qs = line["conversations"][0]["value"]
            cur_prompt = args.extra_prompt + qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            args.conv_mode = "qwen_1_5"

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = preprocess_qwen([line["conversations"][0],{'from': 'gpt','value': None}], tokenizer, has_image=True).cuda()
            img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)

            video_dict = video_processor.process_3d_video(
                video_id,
                image_processor,
                force_sample=args.force_sample,
                frames_upbound=args.max_frame_num,
            )

            if args.jepa_feature_folder is not None:
                scene_id = video_id.split("/")[-1]
                jepa_path = os.path.join(args.jepa_feature_folder, scene_id) + ".pt"
                if os.path.exists(jepa_path):
                    jepa_data = torch.load(jepa_path, map_location="cpu")
                    if isinstance(jepa_data, dict):
                        video_dict["jepa_features"] = jepa_data.get("features", jepa_data.get("jepa_features"))
                        video_dict["jepa_coords"] = jepa_data.get("coords", jepa_data.get("points"))
                    elif isinstance(jepa_data, (tuple, list)) and len(jepa_data) >= 2:
                        video_dict["jepa_features"] = jepa_data[0]
                        video_dict["jepa_coords"] = jepa_data[1]
                    else:
                        raise ValueError(f"Unsupported JEPA feature format: {jepa_path}")
                elif getattr(model.config, "use_jepa_only", False):
                    # JEPA-only mode has no fallback visual signal — record an empty prediction and skip.
                    print(f"[skip] JEPA-only run but JEPA features missing for {video_id} ({jepa_path}); writing empty prediction.")
                    with file_lock:
                        ans_file.write(json.dumps({
                            "dataset": dataset_name,
                            "sample_id": idx,
                            "prompt": cur_prompt,
                            "pred_response": "",
                            "gt_response": gt,
                            "model_id": model_name,
                            "question_type": question_type,
                            "skipped_reason": "missing_jepa_feature",
                        }) + "\n")
                        ans_file.flush()
                    continue

            video_dict = merge_video_dict([video_dict])
            # Use model.dtype (bf16) instead of hard-coded fp16 to keep input/weights consistent.
            image_tensors = video_dict.pop('images').to(dtype=model.dtype, device=model.device)
            for k in video_dict:
                video_dict[k] = video_dict[k].to(dtype=model.dtype, device=model.device)

            # Diagnostic on first sample only: log JEPA shape + prompt length so we
            # can spot truncation (visual tokens >> tokenizer.model_max_length would
            # silently lop off the trailing assistant marker, producing generic
            # captioning output that begins with "assistant ...").
            if _i == 0:
                _jf = video_dict.get("jepa_features")
                _jc = video_dict.get("jepa_coords")
                _max_len = getattr(model.config, "tokenizer_model_max_length", None)
                print(f"[diag] input_ids len={input_ids.shape[1]}  "
                      f"jepa_features shape={tuple(_jf.shape) if _jf is not None else None}  "
                      f"jepa_coords shape={tuple(_jc.shape) if _jc is not None else None}  "
                      f"tokenizer_model_max_length={_max_len}")

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    modalities="video",
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=512,
                    use_cache=True,
                    video_dict=video_dict,
                )

            # NOTE: LlavaQwen.generate() calls super().generate(inputs_embeds=...)
            # without input_ids. With inputs_embeds, HF generate returns ONLY the
            # generated tokens (not input+generated), so no slicing needed.
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            with file_lock:
                ans_file.write(json.dumps({
                                        "dataset": dataset_name,
                                        "sample_id": idx,
                                        "prompt": cur_prompt,
                                        "pred_response": outputs,
                                        "gt_response": gt,
                                        "model_id": model_name,
                                        "question_type": question_type,
                                        }) + "\n")
                ans_file.flush()
        except Exception as e:
            # Don't let one bad sample kill the whole worker. Record an empty
            # prediction so we can resume cleanly and inspect later.
            import traceback
            print(f"[error] sample {idx} (video={video_id if 'video_id' in dir() else line.get('video')}): {type(e).__name__}: {e}")
            traceback.print_exc()
            try:
                with file_lock:
                    ans_file.write(json.dumps({
                        "dataset": line.get("metadata", {}).get("dataset"),
                        "sample_id": idx,
                        "prompt": line["conversations"][0].get("value", ""),
                        "pred_response": "",
                        "gt_response": line["conversations"][1].get("value", ""),
                        "model_id": model_name,
                        "question_type": line.get("metadata", {}).get("question_type"),
                        "skipped_reason": f"{type(e).__name__}: {str(e)[:200]}",
                    }) + "\n")
                    ans_file.flush()
            except Exception:
                pass
            # Best-effort recover from CUDA OOM
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="data")
    parser.add_argument("--embodiedscan-folder", type=str, default="data/embodiedscan")
    parser.add_argument("--extra-prompt", type=str, default="The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions.\n")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--n_gpu", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    parser.add_argument("--frame_sampling_strategy", type=str, default="uniform")
    parser.add_argument("--max_frame_num", type=int, default=32)
    parser.add_argument("--force_sample", type=bool, default=True)
    parser.add_argument("--overwrite_cfg", type=bool, default=False)
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--jepa-feature-folder", type=str, default=None,
                        help="Folder of pre-extracted 3D-JEPA features (one .pt per scene). Required if the trained ckpt was use_jepa_only=True.")
    args = parser.parse_args()

    # Data
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)

    # If answer file exists, allow resume (workers will skip already-done sample_ids).
    if os.path.exists(args.answer_file):
        n_done = sum(1 for _ in open(args.answer_file))
        print(f"[resume] {args.answer_file} already exists with {n_done} lines; resuming.")
    
    ray.init()
    features = []
    for i in range(args.n_gpu):
        features.append(eval_model.remote(questions[i::args.n_gpu], args))

    ray.get(features)
