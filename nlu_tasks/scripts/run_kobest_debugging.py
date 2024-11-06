import os
import sys
import torch
import hydra
import importlib
from datasets import Dataset
from safetensors import safe_open
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AutoConfig, DataCollatorWithPadding, GPT2ForSequenceClassification, GPT2DoubleHeadsModel, AutoTokenizer

sys.path.append(os.getcwd())
from utils.gen_utils import setup_basics
from nlu_tasks.srcs.trainer import GPTNLUTrainer
from srcs.gpt_utils import text_tokenization_for_classification, text_tokenization_for_mc
from pretraining.scripts.run_gpt_pretraining import set_logger, get_gpt2_tokenizer
from srcs.lora import make_only_lora_as_trainable, print_trainable_parameters, apply_lora_to_model, LoRA_Config
from srcs.sub2_debug import make_only_sub2_and_lora_as_trainable, apply_sub2_to_model, SUB2_Config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_nlu_dataloader(args, tokenizer):
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

    # dataset['train'] = {key: dataset['train'][key][:50] for key in dataset['train']}
    # dataset['dev'] = {key: dataset['dev'][key][:50] for key in dataset['dev']}
    # dataset['test'] = {key: dataset['test'][key][:50] for key in dataset['test']}

    data_collator = DataCollatorWithPadding(tokenizer)

    total_dataloader = {'label_map': dataset['label_map']}
    for mode in ['train', 'dev', 'test']:
        data = Dataset.from_dict(dataset[mode])
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
            shuffle=True,
            collate_fn=data_collator,
            batch_size=args.optim.batch_size // args.optim.grad_acc,
            num_workers=args.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        total_dataloader[mode] = dataloader
    return total_dataloader


def get_config_and_nlu_model(args, tokenizer, logger=None):
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
            logger.info(f"\nSave directory: {args.model.ckpt_dir.split('/')[-2]}")
            model_path = os.path.join(args.model.ckpt_dir, "model.safetensors")
            state_dict = {}
            with safe_open(model_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key.replace("_orig_mod.transformer.", "")] = f.get_tensor(key)
            model.transformer.load_state_dict(state_dict)
            logger.info("Complete to reload the checkpoint of the model from above save directory.")
        #TODO: Add the loading function for fine-tuned model, not pre-trained model

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
        make_only_lora_as_trainable(model, bias='lora_only')
        _ = print_trainable_parameters(model, logger)

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

                logger.info(f"Change the max_length to {args.model.sub2.max_length} for the sub2_tokenizer's truncation.")

            if args.data.task_name in ["KB_COPA", "KB_HellaSwag"] and (sub2_tokenizer.cls_token is None):
                _ = sub2_tokenizer.add_special_tokens({"cls_token": "[CLS]"})

        model = apply_sub2_to_model(model, tokenizer, sub2_tokenizer, sub2_config, logger)
        make_only_sub2_and_lora_as_trainable(model, weight='sub2_lora_only', bias='sub2_lora_only')
        _, _ = print_trainable_parameters(model, logger)
    return config, model


@hydra.main(config_path=os.path.join(os.getcwd(), "configs/gpt"), config_name="default", version_base='1.1')
def main(args):
    if args.model.hf_model:
        specific_model_type = ""
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
        args.logging.log_dir = os.path.join(f"logs/debug/{args.model.name.replace('/', '_')}/{args.data.task_name}/{specific_model_type}{args.data.max_length}t_{args.optim.batch_size}b_{args.optim.grad_acc}s_{args.optim.base_lr}lr_{args.seed}rs")
        args.logging.save_dir = os.path.join(args.logging.log_dir, "ckpt")
        args.logging.tb_dir = os.path.join(args.logging.log_dir, "tb")

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

    assert args.optim.batch_size % args.optim.grad_acc == 0, "batch size should be divisible by gradient accumulation steps."

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

            logger.info(f"Change the max_length to {args.data.max_length} for the tokenizer's truncation.")

    if args.data.task_name in ["KB_COPA", "KB_HellaSwag"] and (tokenizer.cls_token is None):
        tokenizer.add_special_tokens({"cls_token": "[CLS]"})

    # Set basic settings for training
    setup_basics(args)

    # Load the Dataset & DataLoader
    dataloaders = get_nlu_dataloader(args, tokenizer)

    batch_num = len(dataloaders['train'])
    args.optim.total_steps = int((batch_num / args.optim.grad_acc) * args.optim.epochs)
    args.optim.warmup_steps = round(args.optim.total_steps * args.optim.warmup_ratio)

    # Set the Config and Model
    config, model = get_config_and_nlu_model(args, tokenizer, logger)

    # Set the Accelerator
    accelerator = Accelerator(
        cpu=(args.device == "cpu"),
        mixed_precision=args.mixed_precision
    )

    trainer = GPTNLUTrainer(args, accelerator, logger, tokenizer, model, dataloaders)

    # from accelerate.utils import set_seed
    # from srcs.functions import init_random
    # if args.seed is not None:
    #     set_seed(args.seed)
    #     init_random(args.seed)

    # ----------------------------------------
    #               Do Fine-Tuning
    # ----------------------------------------
    logger.info("\n")
    logger.info(f"========== Fine-tuning on {args.data.task_name} ==========")
    logger.info(f"model                   : {args.model.name}")
    if not args.model.hf_model:
        logger.info(f"tokenizer               : {args.data.tok_type}")
    logger.info(f"vocab size              : {len(tokenizer)}")
    logger.info(f"device                  : {args.device}")
    logger.info(f"random seed             : {args.seed}")
    logger.info(f"train data size         : {args.optim.total_steps // args.optim.epochs * (args.optim.batch_size // args.optim.grad_acc)}")
    logger.info(f"max epochs              : {args.optim.epochs}")
    logger.info(f"total steps             : {args.optim.total_steps}")
    logger.info(f"warmup steps            : {args.optim.warmup_steps}")
    logger.info(f"batch size              : {args.optim.batch_size}")
    logger.info(f"accumulation steps      : {args.optim.grad_acc}")
    logger.info(f"optimizer               : {args.optim.name}")
    logger.info(f"lr_scheduler            : {args.optim.lr_scheduler}")
    logger.info(f"learning rate           : {args.optim.base_lr}")
    logger.info(f"max length              : {args.data.max_length}\n")
    if args.model.set_lora:
        logger.info(f"LoRA Configuration")
        logger.info(f"ㄴ r                    : {args.model.lora.r}")
        logger.info(f"ㄴ alpha                : {args.model.lora.alpha}")
        logger.info(f"ㄴ dropout              : {args.model.lora.dropout}\n")
    if args.model.set_sub2:
        logger.info(f"SUB2 Configuration")
        logger.info(f"ㄴ tok_type             : {args.model.sub2.tok_type}")
        logger.info(f"ㄴ hidden_dim           : {args.model.sub2.hidden_dim}")
        logger.info(f"ㄴ sub2_max_length     : {args.model.sub2.max_length}")
        logger.info(f"ㄴ embedding_norm       : {args.model.sub2.embedding_norm}")
        logger.info(f"ㄴ do_combination       : {args.model.sub2.do_combination}")
        if args.model.sub2.do_combination:
            logger.info(f"  ㄴ num_attn_heads     : {args.model.sub2.combination.num_attention_heads}")
            logger.info(f"  ㄴ intermediate_size  : {args.model.sub2.combination.intermediate_size}")
            logger.info(f"  ㄴ num_trans_layers   : {args.model.sub2.combination.num_trans_layers}")
            logger.info(f"  ㄴ add_lora           : {args.model.sub2.add_lora}\n")
        else:
            logger.info(f"ㄴ reducer              : {args.model.sub2.reducer}\n")
        if args.model.sub2.add_lora:
            logger.info(f"LoRA in SUB2 Configuration")
            logger.info(f"ㄴ r                : {args.model.sub2.lora.r}")
            logger.info(f"ㄴ alpha            : {args.model.sub2.lora.alpha}")
            logger.info(f"ㄴ dropout          : {args.model.sub2.lora.dropout}\n")
    logger.info('\n')
    if args.model.ckpt_dir:
        logger.info(f"ckpt dir        : {args.model.ckpt_dir}")
    logger.info(f"* log dir       : {args.logging.log_dir}")
    logger.info(f"* save dir      : {args.logging.save_dir}")
    logger.info(f"* tb dir        : {args.logging.tb_dir}")
    logger.info(f"* tb interval   : {args.logging.log_steps}\n")

    # Run training
    print("\n")
    logger.info("\n")
    logger.info("Start the Training !")
    trainer.train()
    logger.info("Fine-tuning is done!")


if __name__ == "__main__":
    main()
