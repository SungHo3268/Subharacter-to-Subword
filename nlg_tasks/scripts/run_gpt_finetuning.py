import os
import sys
import hydra
from datasets import Dataset
from safetensors import safe_open
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import (AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq,
                          GPT2LMHeadModel, GenerationConfig,
                          AutoModelWithLMHead
                          )

sys.path.append(os.getcwd())
from utils.gen_utils import setup_basics
from nlg_tasks.srcs.trainer import GPTNLGTrainer
from srcs.gpt_utils import text_tokenization_for_casuallm
from pretraining.scripts.run_gpt_pretraining import set_logger, get_gpt2_tokenizer
from srcs.lora import make_only_lora_as_trainable, print_trainable_parameters, apply_lora_to_model, LoRA_Config
from srcs.sub2 import make_only_sub2_and_lora_as_trainable, apply_sub2_to_model, SUB2_Config


import transformers
transformers.logging.set_verbosity_warning()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_nlg_dataloader(args, tokenizer, logger):
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
    dataset['dev'] = {key: dataset['dev'][key][:10] for key in dataset['dev']}
    # dataset['test'] = {key: dataset['test'][key][:10] for key in dataset['test']}

    total_dataloader = dict()
    for mode in ['train', 'dev', 'test']:
        data = Dataset.from_dict(dataset[mode])
        batch_size = args.optim.train_batch_size // args.optim.grad_acc if mode == 'train' else args.optim.eval_batch_size
        tokenized_datasets = data.map(text_tokenization_for_casuallm,
                                      fn_kwargs={"tokenizer": tokenizer,
                                                 "max_length": args.model.generation_config.max_length,
                                                 "max_new_tokens": args.model.generation_config.max_new_tokens,
                                                 "task_name": args.data.task_name,
                                                 "mode": mode},
                                      remove_columns=data.column_names,
                                      batched=True,
                                      batch_size=batch_size,
                                      )
        if mode == 'train':
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
            data_shuffle = True
        else:
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

        total_dataloader[mode] = dataloader
    return total_dataloader


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
    # TODO: Add the loading function for fine-tuned model, not pre-trained model

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

        # if ('trans' in args.model.sub2.combination.combination_type) or (args.model.sub2.do_combination is False and args.model.sub2.reducer == 'attention_pool'):
        if args.model.set_sub2:
            trans_config = model.config
            trans_config.update({"is_cross_attention": False})
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
            is_bert=False,
            fusion=args.model.sub2.fusion,
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
                # padding_side='left',
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

        model = apply_sub2_to_model(model, tokenizer, sub2_tokenizer, sub2_config, logger)
        make_only_sub2_and_lora_as_trainable(model, weight='sub2_lora_only', bias='sub2_lora_only')
        _, _ = print_trainable_parameters(model, logger)
    return config, model

@hydra.main(config_path=os.path.join(os.getcwd(), "configs/gpt"), config_name="default", version_base='1.1')
def main(args):
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

    if args.model.set_sub2:
        try: assert args.model.sub2.max_length % args.data.max_length == 0
        except AssertionError: args.model.sub2.max_length = args.model.sub2.max_length - (args.model.sub2.max_length % args.data.max_length)

    if args.model.hf_model:
        specific_model_type = ""
        if args.model.set_lora:
            specific_model_type += "lora_"
        if args.model.set_sub2:
            specific_model_type += f"sub2_{args.model.sub2.tok_type}_{args.model.sub2.max_length}t_"
            if args.model.sub2.do_combination:
                specific_model_type += f"comb-{args.model.sub2.combination.combination_type}_"
            else:
                specific_model_type += f"red-{args.model.sub2.reducer}_"
            specific_model_type += f"{args.model.sub2.fusion}_"

        args.logging.log_dir = os.path.join(f"logs/{args.model.name.replace('/', '_')}/nlg_tasks/{args.data.task_name}/{specific_model_type}{args.model.generation_config.max_length}+{args.model.generation_config.max_new_tokens}t_{args.optim.batch_size}b_{args.optim.grad_acc}s_{args.optim.base_lr}lr_{args.seed}rs")
        args.logging.save_dir = os.path.join(args.logging.log_dir, "ckpt")
        args.logging.tb_dir = os.path.join(args.logging.log_dir, "tb")

    # if args.data.tok_type in ["jamo_var", "stroke_var", "cji_var", "bts_var"]:
    #     args.model.generation_config.no_repeat_ngram_size = 0
    assert args.optim.train_batch_size % args.optim.grad_acc == 0, "batch size should be divisible by gradient accumulation steps."

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

    # Set basic settings for training
    setup_basics(args)

    # Load the Dataset & DataLoader
    dataloaders = get_nlg_dataloader(args, tokenizer, logger)


    # for i, data in enumerate(dataloaders['test']):
    #     # print(tokenizer.decode(data['input_ids'][0]))
    #     # print(tokenizer.decode([idx for idx in data['labels'][0] if idx != -100]))
    #     for label in data['labels']:
    #         text = tokenizer.decode([idx for idx in label if idx != -100]).split(' = ')
    #         text = [t.strip() for t in text if t.strip() != '']
    #         if len(text) != 3:
    #             print("\n")
    #             print(f"text: {text}")
    #             cnt = 1
    #             for z in range(len(data['labels'])):
    #                 d = data['input_ids'][z]
    #                 l = data['labels'][z]
    #                 print(f"{cnt} input: {tokenizer.decode(d)}")
    #                 print(f"{cnt} label: {tokenizer.decode([idx for idx in l if idx != -100]).split(' = ')}")
    #                 cnt += 1
    #                 print("\n")
    #             # print(f"data['labels']: {data['labels']}")
    #             # print(f"data: {data}")
    #             print(f"Error in the label text number {i}")
    #             print("\n\n\n\n")
    # exit("Check the label text")


    batch_num = len(dataloaders['train'])
    args.optim.total_steps = int((batch_num / args.optim.grad_acc) * args.optim.epochs)
    args.optim.warmup_steps = round(args.optim.total_steps * args.optim.warmup_ratio)

    # Set the Config and Model
    config, model = get_config_and_nlg_model(args, tokenizer, logger)

    args.model.generation_config.pad_token_id = tokenizer.eos_token_id
    args.model.generation_config.bos_token_id = tokenizer.eos_token_id
    args.model.generation_config.eos_token_id = tokenizer.eos_token_id
    if args.model.hf_model:
        for key, value in dict(args.model.generation_config).items():
            setattr(model.generation_config, key, value)
    else:
        generation_config = GenerationConfig.from_pretrained(args.model.name)

        for key, value in dict(args.model.generation_config).items():
            setattr(generation_config, key, value)
        model.generation_config = generation_config

    # Set the Accelerator
    accelerator = Accelerator(
        cpu=(args.device == "cpu"),
        mixed_precision=args.mixed_precision
    )

    trainer = GPTNLGTrainer(args, accelerator, logger, tokenizer, model, dataloaders)

    # ----------------------------------------
    #               Do Fine-Tuning
    # ----------------------------------------
    logger.info("\n")
    logger.info(f"========== Fine-tuning on {args.data.task_name} ==========")
    logger.info(f"model                 : {args.model.name}")
    if not args.model.hf_model:
        logger.info(f"tokenizer             : {args.data.tok_type}")
    logger.info(f"vocab size            : {len(tokenizer)}")
    logger.info(f"device                : {args.device}")
    logger.info(f"random seed           : {args.seed}")
    logger.info(f"train data size       : {args.optim.total_steps // args.optim.epochs * (args.optim.train_batch_size // args.optim.grad_acc)}")
    logger.info(f"max epochs            : {args.optim.epochs}")
    logger.info(f"total steps           : {args.optim.total_steps}")
    logger.info(f"warmup steps          : {args.optim.warmup_steps}")
    logger.info(f"train batch size      : {args.optim.train_batch_size}")
    logger.info(f"train grad acc        : {args.optim.grad_acc}")
    logger.info(f"eval batch size       : {args.optim.eval_batch_size}")
    logger.info(f"optimizer             : {args.optim.name}")
    logger.info(f"lr_scheduler          : {args.optim.lr_scheduler}")
    logger.info(f"learning rate         : {args.optim.base_lr}")
    if args.optim.early_stop_patience > 0:
        logger.info(f"early stop patience   : {args.optim.early_stop_patience}")
    if "min_length" in args.model.generation_config:
        logger.info(f"min length            : {args.model.generation_config.min_length}")
    logger.info(f"max total length      : {args.data.max_length}")
    logger.info(f"ㄴ max context length  : {args.model.generation_config.max_length}")
    logger.info(f"ㄴ max new tokens      : {args.model.generation_config.max_new_tokens}")
    if 'repetition_penalty' in args.model.generation_config:
        logger.info(f"repetition_penalty    : {args.model.generation_config.repetition_penalty}")
    if 'no_repeat_ngram_size' in args.model.generation_config:
        logger.info(f"no_repeat_ngram_size  : {args.model.generation_config.no_repeat_ngram_size}")
    if 'do_sample' in args.model.generation_config:
        logger.info(f"do_sample       : {args.model.generation_config.do_sample}")
    if 'num_beams' in args.model.generation_config:
        logger.info(f"num_beams       : {args.model.generation_config.num_beams}")
    if 'length_penalty' in args.model.generation_config:
        logger.info(f"length_penalty  : {args.model.generation_config.length_penalty}")
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
        else:
            logger.info(f"ㄴ reducer              : {args.model.sub2.reducer}\n")
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
