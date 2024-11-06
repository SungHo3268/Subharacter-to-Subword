import os
import sys
import hydra
import importlib
from datasets import Dataset
from safetensors import safe_open
from accelerate import Accelerator
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, DataCollatorWithPadding, GPT2ForSequenceClassification, GPT2DoubleHeadsModel, AutoTokenizer

sys.path.append(os.getcwd())
from utils.gen_utils import setup_basics
from nlu_tasks.srcs.trainer import GPTNLUTrainer
from srcs.gpt_utils import text_tokenization_for_classification, text_tokenization_for_mc
from pretraining.scripts.run_gpt_pretraining import set_logger, get_gpt2_tokenizer
from srcs.lora import make_only_lora_as_trainable, print_trainable_parameters, apply_lora_to_model, LoRA_Config
from srcs.sub2 import make_only_sub2_and_lora_as_trainable, apply_sub2_to_model, SUB2_Config


def get_config_and_nlu_model(args, tokenizer):
    if args.model.hf_model:
        if args.data.task_name in ["KB_COPA", "KB_HellaSwag"]:
            MODEL = GPT2DoubleHeadsModel
            if 'kakaobrain/kogpt' in args.model.name:
                config = AutoConfig.from_pretrained(args.model.name, revision='KoGPT6B-ryan1.5b-float16' if args.mixed_precision in ['bf16', 'float16'] else 'KoGPT6B-ryan1.5b')
            else:
                config = AutoConfig.from_pretrained(args.model.name)
        else:
            if 'kakaobrain/kogpt' in args.model.name:
                config = AutoConfig.from_pretrained(args.model.name, num_labels=args.data.num_labels, revision='KoGPT6B-ryan1.5b-float16' if args.mixed_precision in ['bf16', 'float16'] else 'KoGPT6B-ryan1.5b')
            else:
                config = AutoConfig.from_pretrained(args.model.name, num_labels=args.data.num_labels)
            MODEL = GPT2ForSequenceClassification

        if 'kakaobrain/kogpt' in args.model.name:
            model = MODEL.from_pretrained(
                args.model.name,
                revision='KoGPT6B-ryan1.5b-float16' if args.mixed_precision in ['bf16', 'float16'] else 'KoGPT6B-ryan1.5b',
                pad_token_id=tokenizer.eos_token_id,
                torch_dtype='auto', low_cpu_mem_usage=True
            ).to(device='cuda', non_blocking=True)
        else:
            model = MODEL.from_pretrained(args.model.name, config=config)

        if args.data.task_name in ["KB_COPA", "KB_HellaSwag"]:
            model.resize_token_embeddings(len(tokenizer))
    else:
        config = AutoConfig.from_pretrained(
            args.model.name,
            vocab_size=len(tokenizer),
            n_ctx=args.data.max_length,
            n_positions=args.model.n_positions,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            num_labels=args.data.num_labels
        )

        if args.data.task_name in ["KB_COPA", "KB_HellaSwag"]:
            model = GPT2DoubleHeadsModel(config)
        else:
            model = GPT2ForSequenceClassification(config)

        if args.data.task_name in ["KB_COPA", "KB_HellaSwag"]:
            model.resize_token_embeddings(len(tokenizer))

        # reload the checkpoint of the pre-trained model
        if args.model.ckpt_dir:
            print("\n")
            model_path = os.path.join(args.model.ckpt_dir, "model.safetensors")
            state_dict = {}
            with safe_open(model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key.replace("_orig_mod.transformer.", "")] = f.get_tensor(key)
            model.transformer.load_state_dict(state_dict)

    if args.model.set_lora:
        if ('skt/kogpt2' in args.model.name) or ('skt/ko-gpt-trinity' in args.model.name) or ('ai-forever/mGPT' in args.model.name):
            target_modules = ['c_attn', 'c_proj']
        elif 'EleutherAI/polyglot-ko' in args.model.name:
            target_modules = ['query_key_value', 'dense']
        elif 'kakaobrain/kogpt' in args.model.name:
            target_modules = ['k_proj', 'v_proj', 'q_proj', 'out_proj']
        else:
            raise NotImplementedError

        lora_config = LoRA_Config(
            r=args.model.lora.r,
            lora_alpha=args.model.lora.alpha,
            lora_dropout=args.model.lora.dropout,
            target_modules=target_modules,
        )

        model = apply_lora_to_model(model, lora_config)
        make_only_lora_as_trainable(model, bias='lora_only')
        _ = print_trainable_parameters(model)

    if args.model.set_sub2:
        if 'skt/kogpt2' in args.model.name:
            target_modules = ['c_attn', 'c_proj']
            args.model.sub2.hidden_dim = 768
        elif 'skt/ko-gpt-trinity' in args.model.name:
            target_modules = ['c_attn', 'c_proj']
            args.model.sub2.hidden_dim = 1920
        elif 'ai-forever/mGPT' in args.model.name:
            target_modules = ['c_attn', 'c_proj']
            args.model.sub2.hidden_dim = 2048
        elif 'EleutherAI/polyglot-ko' in args.model.name:
            target_modules = ['query_key_value', 'dense']
            args.model.sub2.hidden_dim = 2048
        elif 'kakaobrain/kogpt' in args.model.name:
            target_modules = ['k_proj', 'v_proj', 'q_proj', 'out_proj']
            args.model.sub2.hidden_dim = 4096
        else:
            raise NotImplementedError

        if args.model.set_sub2:
            trans_config = config
            trans_config.update({"embedding_norm": args.model.sub2.embedding_norm})
        else:
            trans_config = None

        lora_config = LoRA_Config(
            r=args.model.sub2.lora.r,
            lora_alpha=args.model.sub2.lora.alpha,
            lora_dropout=args.model.sub2.lora.dropout,
            target_modules=target_modules
        )
        sub2_config = SUB2_Config(
            tok_type=args.model.sub2.tok_type,
            reducer=args.model.sub2.reducer,
            hidden_dim=args.model.sub2.hidden_dim,
            sub2_max_length=args.model.sub2.max_length,
            max_length=args.data.max_length,
            do_combination=args.model.sub2.do_combination,
            combination_type=args.model.sub2.combination.combination_type,
            trans_config=trans_config,
            num_attention_heads=args.model.sub2.combination.num_attention_heads,
            intermediate_size=args.model.sub2.combination.intermediate_size,
            num_trans_layers=args.model.sub2.combination.num_trans_layers,
            add_lora=args.model.sub2.add_lora,
            is_bert=False,
            lora_config=lora_config
        )
        if args.model.sub2.tok_type == "same":
            sub2_tokenizer = tokenizer
            args.model.sub2.max_length = args.data.max_length
        else:
            sub2_tokenizer = get_gpt2_tokenizer(
                tok_type=args.model.sub2.tok_type,
                lang=args.model.sub2.lang,
                max_length=args.model.sub2.max_length,
                lowercase=True,
                clean_text=True,
                add_bos_token=False,
                bos_token='<|endoftext|>',
                eos_token='<|endoftext|>',
                unk_token='<unk>',
                pad_token='<|endoftext|>',
            )
            if (hasattr(sub2_tokenizer, "trunc_num") and
                    args.model.sub2.tok_type in ["jamo_var", "stroke_var", "cji_var", "bts_var"] and
                    args.model.sub2.max_length % sub2_tokenizer.trunc_num != 0):
                args.model.sub2.max_length = args.model.sub2.max_length - (args.model.sub2.max_length % sub2_tokenizer.trunc_num)
                sub2_tokenizer.max_length = args.model.sub2.max_length

            if args.data.task_name in ["KB_COPA", "KB_HellaSwag"] and (sub2_tokenizer.cls_token is None):
                _ = sub2_tokenizer.add_special_tokens({"cls_token": "[CLS]"})

        model = apply_sub2_to_model(model, tokenizer, sub2_tokenizer, sub2_config, None)
        make_only_sub2_and_lora_as_trainable(model, weight='sub2_lora_only', bias='sub2_lora_only')
        _, _ = print_trainable_parameters(model, None)
    return config, model

    

@hydra.main(config_path=os.path.join(os.getcwd(), "configs/gpt"), config_name="default", version_base='1.1')
def main(args):
    specific_model_type = ""
    if args.model.hf_model:
        if args.model.set_lora:
            specific_model_type += "lora_"
        if args.model.set_sub2:
            specific_model_type += "sub2_"
            if args.model.sub2.do_combination:
                specific_model_type += f"comb-{args.model.sub2.combination.combination_type}_"
                if args.model.sub2.add_lora:
                    specific_model_type += "k-lora_"
            else:
                specific_model_type += f"{args.model.sub2.tok_type}_{args.model.sub2.max_length}_red-{args.model.sub2.reducer}_"
    
    model_name = args.model.name.replace('/', '_') + specific_model_type
    
    # Get the Tokenizer
    if args.model.hf_model:
        if 'skt/kogpt2' in args.model.name or 'skt/ko-gpt-trinity' in args.model.name:
            tokenizer = AutoTokenizer.from_pretrained(args.model.name,
                                                      bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                      pad_token='<pad>', mask_token='<mask>',
                                                      padding_side='left',
                                                      )
        elif 'EleutherAI/polyglot-ko' in args.model.name or 'ai-forever/mGPT' in args.model.name:
            tokenizer = AutoTokenizer.from_pretrained(args.model.name)
        elif 'kakaobrain/kogpt' in args.model.name:
            tokenizer = AutoTokenizer.from_pretrained(
                revision='KoGPT6B-ryan1.5b-float16' if args.mixed_precision in ['bf16',
                                                                                'float16'] else 'KoGPT6B-ryan1.5b',
                bos_token='[BOS]', eos_token='[EOS]', unk_token='[UNK]', pad_token='[PAD]', mask_token='[MASK]'
            )
        else:
            raise ValueError("It's a Wrong Model Name. Please enter the right model name.")
    else:
        tokenizer = get_gpt2_tokenizer(tok_type=args.data.tok_type,
                                       lang=args.data.language,
                                       max_length=args.data.max_length,
                                       lowercase=True,
                                       clean_text=True,
                                       add_bos_token=False,
                                       bos_token='<|endoftext|>',
                                       eos_token='<|endoftext|>',
                                       unk_token='<unk>',
                                       pad_token='<|endoftext|>',
                                       )
        tokenizer.pad_token = tokenizer.eos_token
        if (hasattr(tokenizer, "trunc_num") and
                args.data.tok_type in ["jamo_var", "stroke_var", "cji_var", "bts_var"] and
                args.data.max_length % tokenizer.trunc_num != 0):
            args.data.max_length = args.data.max_length - (args.data.max_length % tokenizer.trunc_num)
            tokenizer.max_length = args.data.max_length

    if args.data.task_name in ["KB_COPA", "KB_HellaSwag"] and (tokenizer.cls_token is None):
        tokenizer.add_special_tokens({"cls_token": "[CLS]"})

    # Set the Config and Model
    config, model = get_config_and_nlu_model(args, tokenizer, None)

    print("모델 이름 : ", model_name)
    print("학습 파라미터 : ", sum(p.numel() for p in model.parameters() if p.requires_grad))

if __name__ == "__main__":
    main()