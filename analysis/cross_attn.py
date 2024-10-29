from __future__ import absolute_import

import os
import sys
import hydra
import importlib
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import (AutoConfig, AutoTokenizer,
                          GPT2LMHeadModel,
                          AutoModelWithLMHead,
                          DataCollatorForSeq2Seq,
                          GPT2DoubleHeadsModel,
                          GPT2ForSequenceClassification,
                          DataCollatorWithPadding,
                          )

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import json

sys.path.append(os.getcwd())
from utils.gen_utils import setup_basics
from srcs.gpt_utils import text_tokenization_for_casuallm
from srcs.gpt_utils import text_tokenization_for_classification, text_tokenization_for_mc

from pretraining.scripts.run_pretraining import set_logger, get_gpt2_tokenizer
from srcs.lora import make_only_lora_as_trainable, print_trainable_parameters, apply_lora_to_model, LoRA_Config
from srcs.kombo import make_only_kombo_and_lora_as_trainable, apply_kombo_to_model, KOMBO_Config

import inspectus

import transformers
transformers.logging.set_verbosity_warning()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_config_and_nlu_model(args, tokenizer, logger=None):
    if args.data.task_name in ["KorSTS"]:
        args.data.num_labels = 1
    elif args.data.task_name in ["NSMC", "PAWS_X", "KB_BoolQ", "KB_WiC", "KB_SentiNeg"]:
        args.data.num_labels = 2
    elif args.data.task_name in ["KorNLI"]:
        args.data.num_labels = 3
    elif args.data.task_name in ["KorQuAD", "KB_COPA", "KB_HellaSwag"]:
        pass
    else:
        raise ValueError("It's a Wrong Task Name. Please enter the right task name among [KorQuAD, KorNLI, KorSTS, NSMC, PAWS_X]")

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

        model = apply_lora_to_model(model, lora_config, logger)

    if args.model.set_kombo:
        if 'skt/kogpt2' in args.model.name:
            target_modules = ['c_attn', 'c_proj']
            args.model.kombo.hidden_dim = 768
        elif 'skt/ko-gpt-trinity' in args.model.name:
            target_modules = ['c_attn', 'c_proj']
            args.model.kombo.hidden_dim = 1920
        elif 'ai-forever/mGPT' in args.model.name:
            target_modules = ['c_attn', 'c_proj']
            args.model.kombo.hidden_dim = 2048
        elif 'EleutherAI/polyglot-ko' in args.model.name:
            target_modules = ['query_key_value', 'dense']
            args.model.kombo.hidden_dim = 2048
        elif 'kakaobrain/kogpt' in args.model.name:
            target_modules = ['k_proj', 'v_proj', 'q_proj', 'out_proj']
            args.model.kombo.hidden_dim = 4096
        else:
            raise NotImplementedError

        if args.model.set_kombo:
            trans_config = config
            trans_config.update({"embedding_norm": args.model.kombo.embedding_norm})
        else:
            trans_config = None

        lora_config = LoRA_Config(
            r=args.model.kombo.lora.r,
            lora_alpha=args.model.kombo.lora.alpha,
            lora_dropout=args.model.kombo.lora.dropout,
            target_modules=target_modules
        )
        kombo_config = KOMBO_Config(
            tok_type=args.model.kombo.tok_type,
            reducer=args.model.kombo.reducer,
            hidden_dim=args.model.kombo.hidden_dim,
            kombo_max_length=args.model.kombo.kombo_max_length,
            max_length=args.data.max_length,
            do_combination=args.model.kombo.do_combination,
            combination_type=args.model.kombo.combination.combination_type,
            trans_config=trans_config,
            num_attention_heads=args.model.kombo.combination.num_attention_heads,
            intermediate_size=args.model.kombo.combination.intermediate_size,
            num_trans_layers=args.model.kombo.combination.num_trans_layers,
            add_lora=args.model.kombo.add_lora,
            lora_config=lora_config
        )
        if args.model.kombo.tok_type == "same":
            kombo_tokenizer = tokenizer
            args.model.kombo.kombo_max_length = args.data.max_length
        else:
            kombo_tokenizer = get_gpt2_tokenizer(
                tok_type=args.model.kombo.tok_type,
                lang=args.model.kombo.lang,
                max_length=args.model.kombo.kombo_max_length,
                lowercase=True,
                clean_text=True,
                add_bos_token=False,
                bos_token='<|endoftext|>',
                eos_token='<|endoftext|>',
                unk_token='<unk>',
                pad_token='<|endoftext|>',
            )
            if (hasattr(kombo_tokenizer, "trunc_num") and
                    args.model.kombo.tok_type in ["jamo_var", "stroke_var", "cji_var", "bts_var"] and
                    args.model.kombo.kombo_max_length % kombo_tokenizer.trunc_num != 0):
                args.model.kombo.kombo_max_length = args.model.kombo.kombo_max_length - (args.model.kombo.kombo_max_length % kombo_tokenizer.trunc_num)
                kombo_tokenizer.max_length = args.model.kombo.kombo_max_length

                logger.info(f"Change the max_length to {args.model.kombo.kombo_max_length} for the kombo_tokenizer's truncation.")

            if args.data.task_name in ["KB_COPA", "KB_HellaSwag"] and (kombo_tokenizer.cls_token is None):
                _ = kombo_tokenizer.add_special_tokens({"cls_token": "[CLS]"})

        model = apply_kombo_to_model(model, tokenizer, kombo_tokenizer, kombo_config, logger)
    
    # reload the checkpoint of the pre-trained model
    if args.model.ckpt_dir:
        print("\n")
        logger.info(f"\nSave directory: {args.model.ckpt_dir.split('/')[-2]}")
        model_path = os.path.join(args.model.ckpt_dir, "checkpoint-best/torch.save.pytorch_model.bin")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        logger.info("Complete to reload the checkpoint of the model from above save directory.")
    return config, model

def get_config_and_nlg_model(args, tokenizer, logger=None):
    if args.model.hf_model:
        if 'skt/kogpt2' in args.model.name:
            config = AutoConfig.from_pretrained(args.model.name)
            model = GPT2LMHeadModel.from_pretrained(args.model.name, config=config)
        elif ('skt/ko-gpt-trinity' in args.model.name) or ('EleutherAI/polyglot-ko' in args.model.name) or ('ai-forever/mGPT' in args.model.name):
            config = AutoConfig.from_pretrained(args.model.name)
            model = AutoModelWithLMHead.from_pretrained(args.model.name, config=config)
        elif 'kakaobrain/kogpt' in args.model.name:
            config = AutoConfig.from_pretrained(args.model.name, revision='KoGPT6B-ryan1.5b-float16' if args.mixed_precision in ['bf16', 'float16'] else 'KoGPT6B-ryan1.5b')
            model = AutoModelWithLMHead.from_pretrained(
                args.model.name,
                revision='KoGPT6B-ryan1.5b-float16' if args.mixed_precision in ['bf16', 'float16'] else 'KoGPT6B-ryan1.5b',
                pad_token_id=tokenizer.eos_token_id,
                torch_dtype='auto', low_cpu_mem_usage=True
            ).to(device='cuda', non_blocking=True)
        else:
            raise ValueError("It's a Wrong Model Name. Please enter the right model name.")
    else:
        config = AutoConfig.from_pretrained(
            args.model.name,
            vocab_size=len(tokenizer),
            n_ctx=args.model.n_positions,
            n_positions=args.model.n_positions,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        model = GPT2LMHeadModel(config)

    if args.model.set_lora:
        if ('skt/kogpt2' in args.model.name) or ('skt/ko-gpt-trinity' in args.model.name) or ('ai-forever/mGPT' in args.model.name):        # GPT-2 src codes
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
        model = apply_lora_to_model(model, lora_config, logger)
        
    if args.model.set_kombo:
        if 'skt/kogpt2' in args.model.name:
            target_modules = ['c_attn', 'c_proj']
            args.model.kombo.hidden_dim = 768
        elif 'skt/ko-gpt-trinity' in args.model.name:
            target_modules = ['c_attn', 'c_proj']
            args.model.kombo.hidden_dim = 1920
        elif 'ai-forever/mGPT' in args.model.name:
            target_modules = ['c_attn', 'c_proj']
            args.model.kombo.hidden_dim = 2048
        elif 'EleutherAI/polyglot-ko' in args.model.name:
            target_modules = ['query_key_value', 'dense']
            args.model.kombo.hidden_dim = 2048
        elif 'kakaobrain/kogpt' in args.model.name:
            target_modules = ['k_proj', 'v_proj', 'q_proj', 'out_proj']
            args.model.kombo.hidden_dim = 4096
        else:
            raise NotImplementedError

        # if ('trans' in args.model.kombo.combination.combination_type) or (args.model.kombo.do_combination is False and args.model.kombo.reducer == 'attention_pool'):
        if args.model.set_kombo:
            trans_config = config
            trans_config.update({"embedding_norm": args.model.kombo.embedding_norm})
        else:
            trans_config = None

        lora_config = LoRA_Config(
            r=args.model.kombo.lora.r,
            lora_alpha=args.model.kombo.lora.alpha,
            lora_dropout=args.model.kombo.lora.dropout,
            target_modules=target_modules
        )
        kombo_config = KOMBO_Config(
            tok_type=args.model.kombo.tok_type,
            reducer=args.model.kombo.reducer,
            hidden_dim=args.model.kombo.hidden_dim,
            kombo_max_length=args.model.kombo.kombo_max_length,
            max_length=args.data.max_length,
            do_combination=args.model.kombo.do_combination,
            combination_type=args.model.kombo.combination.combination_type,
            trans_config=trans_config,
            num_attention_heads=args.model.kombo.combination.num_attention_heads,
            intermediate_size=args.model.kombo.combination.intermediate_size,
            num_trans_layers=args.model.kombo.combination.num_trans_layers,
            add_lora=args.model.kombo.add_lora,
            lora_config=lora_config
        )
        if args.model.kombo.tok_type == "same":
            kombo_tokenizer = tokenizer
            args.model.kombo.kombo_max_length = args.data.max_length
        else:
            kombo_tokenizer = get_gpt2_tokenizer(
                tok_type=args.model.kombo.tok_type,
                lang=args.model.kombo.lang,
                max_length=args.model.kombo.kombo_max_length,
                lowercase=True,
                clean_text=True,
                add_bos_token=False,
                # padding_side='left',
                bos_token='<|endoftext|>',
                eos_token='<|endoftext|>',
                unk_token='<unk>',
                pad_token='<|endoftext|>',
            )
            if (hasattr(kombo_tokenizer, "trunc_num") and
                    args.model.kombo.tok_type in ["jamo_var", "stroke_var", "cji_var", "bts_var"] and
                    args.model.kombo.kombo_max_length % kombo_tokenizer.trunc_num != 0):
                args.model.kombo.kombo_max_length = args.model.kombo.kombo_max_length - (args.model.kombo.kombo_max_length % kombo_tokenizer.trunc_num)
                kombo_tokenizer.max_length = args.model.kombo.kombo_max_length

                logger.info(f"Change the max_length to {args.model.kombo.kombo_max_length} for the kombo_tokenizer's truncation.")

        model = apply_kombo_to_model(model, tokenizer, kombo_tokenizer, kombo_config, logger)
        
    # reload the checkpoint of the pre-trained model
    if args.model.ckpt_dir:
        print("\n")
        logger.info(f"\nSave directory: {args.model.ckpt_dir.split('/')[-2]}")
        model_path = os.path.join(args.model.ckpt_dir, "checkpoint-best/torch.save.pytorch_model.bin")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        logger.info("Complete to reload the checkpoint of the model from above save directory.")
    # TODO: Add the loading function for fine-tuned model, not pre-trained model
    return config, model


def cross_attention_score(wte, x):
    device = x.device

    original_embedding = wte.original_layer(x)

    text_input = wte.tokenizer.batch_decode(x, skip_special_tokens=True)

    kombo_x = wte.make_kombo_input(text_input, device)                 # (B, N(=max_kombo_length), D)

    kombo_embedding = wte.kombo_combination(kombo_x, text_input)       # (B, N(=max_length), D)

    if wte.config.add_lora:
        # Apply dropout before the matrix multiplication
        A_dropout = wte.dropout(wte.kombo_lora_A)
        B_dropout = wte.dropout(wte.kombo_lora_B)
        W = wte.scaling * A_dropout @ B_dropout
        kombo_embedding = F.linear(kombo_embedding, W.T)
    else:
        pass

    # For NLU tasks, the output dimension of the original layer and the KOMBO layer may be different.
    if original_embedding.shape[1] != kombo_embedding.shape[1]:
        kombo_embedding = kombo_embedding[:, :original_embedding.shape[1]]

    hidden_states, before_pooled_states = [original_embedding, kombo_embedding.contiguous()]
    residual = hidden_states

    attn_outputs = wte.kombo_injection.attn(
        hidden_states,
        before_pooled_states,
        output_attentions=True,
    )
    cross_attention = attn_outputs[-1]
    return cross_attention
        

def plot_head_map(dir, mma, target_labels, source_labels):
    fe = fm.FontEntry(
    fname=r'/usr/share/fonts/truetype/nanum/NanumBarunpenR.ttf', # ttf 파일이 저장되어 있는 경로
    name='NanumBarunpenR')                        # 원하는 폰트 설정
    fm.fontManager.ttflist.insert(0, fe)              # Matplotlib에 폰트 추가

    plt.rcParams.update({'font.size': 18, 'font.family': 'NanumBarunpenR'}) # 폰트 설정

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(mma, cmap=plt.cm.Blues)
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(mma.shape[1]) + 0.5, minor=False) # mma.shape[1] = target seq 길이
    ax.set_yticks(np.arange(mma.shape[0]) + 0.5, minor=False) # mma.shape[0] = input seq 길이
 
    # without this I get some extra columns rows
    # http://stackoverflow.com/questions/31601351/why-does-this-matplotlib-heatmap-have-an-extra-blank-column
    ax.set_xlim(0, int(mma.shape[1]))
    ax.set_ylim(0, int(mma.shape[0]))
 
    # want a more natural, table-like display
    ax.invert_yaxis()
    ax.xaxis.tick_top()
 
    # source words -> column labels
    print(source_labels)
    ax.set_xticklabels(source_labels, minor=False)
    # target words -> row labels
    ax.set_yticklabels(target_labels, minor=False)
 
    plt.xticks(rotation=45)
 
    # plt.tight_layout()
    plt.show()
    plt.savefig(dir)
 
 
def read_plot_alignment_matrices(dir, source_labels, target_labels, alpha):
 
    mma = alpha.cpu().data.numpy()
    plot_head_map(dir, mma, target_labels, source_labels)
    

@hydra.main(config_path=os.path.join(os.getcwd(), "configs/gpt"), config_name="default", version_base='1.1')
def main(args):
    if args.mode == "nlg_ft":
        if args.model.generation_config.do_sample is None:
            del args.model.generation_config.do_sample
        if args.model.generation_config.num_beams is None:
            del args.model.generation_config.num_beams
        if args.model.generation_config.repetition_penalty is None:
            del args.model.generation_config.repetition_penalty
        if args.model.generation_config.no_repeat_ngram_size is None:
            del args.model.generation_config.no_repeat_ngram_size
        if args.model.generation_config.length_penalty is None:
            del args.model.generation_config.length_penalty

        # add 5 extra tokens for the sep_id tokens (e.g., '. ' @ KoCommonGen/ ' 요약: ' @ XL_Sum/ ' 수정: ' @ KoreanGEC)
        args.data.max_length = args.model.generation_config.max_length + args.model.generation_config.max_new_tokens + 5

        if args.model.set_kombo:
            try: assert args.model.kombo.kombo_max_length % args.data.max_length == 0
            except AssertionError: args.model.kombo.kombo_max_length = args.model.kombo.kombo_max_length - (args.model.kombo.kombo_max_length % args.data.max_length)


        if args.model.hf_model:
            specific_model_type = ""
            if args.model.set_lora:
                specific_model_type += "lora_"
            if args.model.set_kombo:
                specific_model_type += "kombo_"
                if args.model.kombo.do_combination:
                    specific_model_type += f"comb-{args.model.kombo.combination.combination_type}_"
                    if args.model.kombo.add_lora:
                        specific_model_type += "k-lora_"
                else:
                    specific_model_type += f"{args.model.kombo.tok_type}_{args.model.kombo.kombo_max_length}_red-{args.model.kombo.reducer}_"

        args.logging.log_dir = os.path.join(f"analysis/cross_attention/{args.model.name.replace('/', '_')}/nlg_tasks/{args.data.task_name}/{specific_model_type}{args.model.generation_config.max_length}+{args.model.generation_config.max_new_tokens}t_{args.optim.batch_size}b_{args.optim.grad_acc}s_{args.optim.base_lr}lr_{args.seed}rs")
        assert args.optim.train_batch_size % args.optim.grad_acc == 0, "batch size should be divisible by gradient accumulation steps."

    else:
        data_preprocess = f"{args.data.remain_lang}"
        if args.data.do_hangeulize: data_preprocess += "_dh"
        if args.data.data_remove: data_preprocess += "_dr"
        model_name = f"{args.model.name.replace('/', '_')}"
        if args.model.set_kombo: 
            model_name += "/kombo"
        elif args.model.set_lora: 
            model_name += "/lora"
        args.logging.log_dir = os.path.join(f"analysis/cross_attention/{model_name}/nlu_ft/{args.data.task_name}/{data_preprocess}/{args.data.max_length}t_{args.optim.batch_size}b_{args.optim.grad_acc}s_{args.seed}rs")
        
    # if args.data.tok_type in ["jamo_var", "stroke_var", "cji_var", "bts_var"]:
    #     args.model.generation_config.no_repeat_ngram_size = 0

    logger = set_logger(args, tqdm_handler=False)

    # ----------------------------------------
    #           Set the Trainer
    # ----------------------------------------
    # Get the Tokenizer
    if args.model.hf_model:
        if 'skt/kogpt2' in args.model.name or 'skt/ko-gpt-trinity' in args.model.name:
            tokenizer = AutoTokenizer.from_pretrained(args.model.name,
                                                      bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                      pad_token='<pad>', mask_token='<mask>',
                                                      padding_side='left',
                                                      )
        elif ('EleutherAI/polyglot-ko' in args.model.name) or ('ai-forever/mGPT' in args.model.name):
            tokenizer = AutoTokenizer.from_pretrained(args.model.name)
        elif 'kakaobrain/kogpt' in args.model.name:
            tokenizer = AutoTokenizer.from_pretrained(
                revision='KoGPT6B-ryan1.5b-float16' if args.mixed_precision in ['bf16', 'float16'] else 'KoGPT6B-ryan1.5b',
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
                                       # padding_side='left',
                                       bos_token='<|endoftext|>',
                                       eos_token='<|endoftext|>',
                                       unk_token='<unk>',
                                       pad_token='<|endoftext|>',
                                       )
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.custom_tokenizer.config.name in ["jamo_var_info", "bts_units_var_info"]:
            args.model.generation_config.max_length = args.model.generation_config.max_length - (args.model.generation_config.max_length % tokenizer.trunc_num)
            args.model.generation_config.max_new_tokens = args.model.generation_config.max_new_tokens - (args.model.generation_config.max_new_tokens % tokenizer.trunc_num)
            args.data.max_length = args.model.generation_config.max_length + args.model.generation_config.max_new_tokens
            tokenizer.max_length = args.data.max_length

        logger.info(f"\nChange the total max_length for generation to {args.data.max_length}.")
        logger.info(f"Change the context max_length for generation to {args.model.generation_config.max_length}.")
        logger.info(f"Change the max_new_tokens for generation to {args.model.generation_config.max_new_tokens}.")
        logger.info(f"Change the tokenizer_max_length to {args.model.generation_config.max_new_tokens}.\n")


    if args.mode == "nlg_ft":
        # Set the Config and Model
        config, model = get_config_and_nlg_model(args, tokenizer, logger)
        
        if args.data.task_name == 'KoCommonGen':
            from nlg_tasks.data_utils.KoCommonGen.data_utils import load_task_dataset
        elif args.data.task_name == 'XL_Sum':
            from nlg_tasks.data_utils.XL_Sum.data_utils import load_task_dataset
        elif args.data.task_name == 'WikiLingua':
            from nlg_tasks.data_utils.WikiLingua.data_utils import load_task_dataset
        elif 'KoreanGEC' in args.data.task_name:
            from nlg_tasks.data_utils.KoreanGEC.data_utils import load_task_dataset
        else:
            logger.info(
                "It's a Wrong Task Name. Please enter the right task name among [KorNLI, KorSTS, NSMC, PAWS_X]")
            raise ValueError

        dataset = load_task_dataset()
        if 'KoreanGEC' in args.data.task_name:
            dataset = dataset[args.data.task_name]
            
        # dataset['train'] = {key: dataset['train'][key][:10] for key in dataset['train']}
        # dataset['dev'] = {key: dataset['dev'][key][:10] for key in dataset['dev']}
        dataset['test'] = {key: dataset['test'][key][:10] for key in dataset['test']}
        print(args.model.generation_config.max_length)
        data = Dataset.from_dict(dataset['test'])
        batch_size = args.optim.eval_batch_size
        tokenized_datasets = data.map(text_tokenization_for_casuallm,
                                        fn_kwargs={"tokenizer": tokenizer,
                                                    "max_length": args.model.generation_config.max_length,
                                                    "max_new_tokens": args.model.generation_config.max_new_tokens,
                                                    "task_name": args.data.task_name,
                                                    "mode": 'test'},
                                        remove_columns=data.column_names,
                                        batched=True,
                                        batch_size=batch_size,
                                        )
        data_collator = DataCollatorForSeq2Seq(tokenizer)
        data_shuffle = False

        dataloader = DataLoader(
            tokenized_datasets,
            shuffle=data_shuffle,
            collate_fn=data_collator,
            batch_size=batch_size,
            num_workers=args.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    
    else:
        # Set the Config and Model
        config, model = get_config_and_nlu_model(args, tokenizer, logger)
        
        task_util_path = f"nlu_tasks.data_utils.{args.data.task_name}.data_utils"
        task_util = importlib.import_module(task_util_path, package=".")
        if args.data.task_name in ['KorNLI', 'KorSTS', 'NSMC', 'PAWS_X']:
            dataset = task_util.load_task_dataset(args.data.remain_lang, args.data.do_hangeulize, args.data.data_remove)
        elif args.data.task_name in ['KB_BoolQ', 'KB_COPA', 'KB_WiC', 'KB_HellaSwag', 'KB_SentiNeg']:
            dataset = task_util.load_task_dataset()
        else:
            raise ValueError(f"It's a Wrong Task Name (entered '{args.data.task_name}'). Please enter the right task name among "
                            "[KorNLI, KorSTS, NSMC, PAWS_X] or "
                            "[KB_BoolQ, KB_COPA, KB_WiC, KB_HellaSwag, KB_SentiNeg]")

        # dataset['train'] = {key: dataset['train'][key][:10] for key in dataset['train']}
        # dataset['dev'] = {key: dataset['dev'][key][:10] for key in dataset['dev']}
        dataset['test'] = {key: dataset['test'][key][:10] for key in dataset['test']}

        data_collator = DataCollatorWithPadding(tokenizer)

        total_dataloader = {'label_map': dataset['label_map']}
        data = Dataset.from_dict(dataset['test'])
        if args.data.task_name in ['KB_COPA', 'KB_HellaSwag']:      # For multiple choice tasks
            tokenized_datasets = data.map(text_tokenization_for_mc,
                                        fn_kwargs={"tokenizer": tokenizer,
                                                    "max_length": args.data.max_length},
                                        remove_columns=data.column_names,
                                        batched=True,
                                        batch_size=args.optim.batch_size // args.optim.grad_acc,
                                        )
        else:       # For sentence classification tasks
            tokenized_datasets = data.map(text_tokenization_for_classification,
                                        fn_kwargs={"tokenizer": tokenizer,
                                                    "max_length": args.data.max_length},
                                        remove_columns=data.column_names,
                                        batched=True,
                                        batch_size=args.optim.batch_size // args.optim.grad_acc,
                                        )
        dataloader = DataLoader(
            tokenized_datasets,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=args.optim.batch_size // args.optim.grad_acc,
            num_workers=args.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    sent_num = 0
    
    with torch.no_grad():
        for batch in dataloader:
            tokens = tokenizer.batch_decode(batch.input_ids)[sent_num]
            tokens = tokenizer.tokenize(tokens)

            cross_attn = cross_attention_score(model.transformer.wte, batch.input_ids) # batch_size, attention_heads, seq_len, seq_len

    seq_len = torch.where(batch.input_ids[sent_num]==tokenizer.pad_token_id)[0][0] + 1
    
    figure_dir = os.path.join(args.logging.log_dir, 'cross_attn.png')
    attention = cross_attn[sent_num][0] # (batch_size, attention_heads, seq_len, seq_len) * num_layers
    attention = F.normalize(attention)

    read_plot_alignment_matrices(figure_dir, tokens[:seq_len], tokens[:seq_len], attention[:seq_len, :seq_len])

    # attention = output.attentions[0][:, :, :seq_len, :seq_len]
    # inspectus.attention(attention, tokens[:seq_len], tokens[:seq_len])
    # plt.savefig("attn.png")
    
if __name__ == "__main__":
    main()
    