import os
import sys
import hydra
from datasets import Dataset
from safetensors import safe_open
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, GPT2LMHeadModel, GenerationConfig

sys.path.append(os.getcwd())
from utils.gen_utils import setup_basics
from nlg_tasks.srcs.trainer import GPT2NLGTrainer
from srcs.gpt2_utils import text_tokenization_for_casuallm
from pretraining.scripts.run_pretraining import set_logger, get_gpt2_tokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_nlg_dataloader(args, tokenizer, logger):
    if args.data.task_name == 'KoCommonGen':
        from nlg_tasks.data_utils.KoCommonGen.data_utils import load_task_dataset
    elif args.data.task_name == 'XL_Sum':
        from nlg_tasks.data_utils.XL_Sum.data_utils import load_task_dataset
    else:
        logger.info(
            "It's a Wrong Task Name. Please enter the right task name among [KorNLI, KorSTS, NSMC, PAWS_X]")
        raise ValueError

    dataset = load_task_dataset()

    total_dataloader = dict()
    for mode in ['train', 'dev', 'test']:
        data = Dataset.from_dict(dataset[mode])
        batch_size = args.optim.train_batch_size if mode == 'train' else args.optim.eval_batch_size
        # grad_acc = args.optim.grad_acc if mode == 'train' else 1
        grad_acc = args.optim.grad_acc
        tokenized_datasets = data.map(text_tokenization_for_casuallm,
                                      fn_kwargs={"tokenizer": tokenizer,
                                                 "max_length": args.model.generation_config.max_length,
                                                 "max_new_tokens": args.model.generation_config.max_new_tokens,
                                                 "task_name": args.data.task_name,
                                                 "mode": mode},
                                      remove_columns=data.column_names,
                                      batched=True,
                                      batch_size=batch_size // grad_acc,
                                      )
        if mode == 'train':
            data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        else:
            data_collator = DataCollatorForSeq2Seq(tokenizer)

        dataloader = DataLoader(
            tokenized_datasets,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=batch_size // grad_acc,
            num_workers=args.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        total_dataloader[mode] = dataloader
    return total_dataloader


def get_config_and_nlg_model(args, tokenizer, logger=None):
    if args.model.hf_model:
        config = AutoConfig.from_pretrained(args.model.name)
        model = GPT2LMHeadModel.from_pretrained(args.model.name)
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
    return config, model


@hydra.main(config_path="../configs/gpt2", config_name="default", version_base='1.1')
def main(args):
    if args.model.hf_model:
        args.logging.log_dir = os.path.join(f"log/{args.model.name}/nlg_tasks/{args.data.task_name}/{args.data.max_length}t_{args.optim.batch_size}b_{args.optim.grad_acc}s_{args.optim.base_lr}lr_{args.seed}rs")
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
        if args.model.name == 'skt/kogpt2-base-v2':
            tokenizer = AutoTokenizer.from_pretrained(args.model.name,
                                                      bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                      pad_token='<pad>', mask_token='<mask>',
                                                      # padding_side='left',
                                                      )
        else:
            raise ValueError("It's a Wrong Model Name. Please enter the right model name.")
    else:
        tokenizer = get_gpt2_tokenizer(tok_type=args.data.tok_type,
                                       lang=args.data.language,
                                       max_length=args.model.generation_config.max_length,
                                       lowercase=True,
                                       clean_text=True,
                                       add_bos_token=False,
                                       padding_side='left',
                                       )
        tokenizer.pad_token = tokenizer.eos_token

    if (hasattr(tokenizer, "trunc_num") and tokenizer.custom_tokenizer.config.name in ["jamo_var_info", "bts_units_var_info"] and
            args.model.generation_config.max_length % tokenizer.trunc_num != 0 and
            args.model.generation_config.max_new_tokens % tokenizer.trunc_num != 0):
        args.model.generation_config.max_length = args.model.generation_config.max_length - (args.model.generation_config.max_length % tokenizer.trunc_num)
        args.model.generation_config.max_new_tokens = args.model.generation_config.max_new_tokens - (args.model.generation_config.max_new_tokens % tokenizer.trunc_num)

    # if (args.data.tok_type in ["jamo_var", "stroke_var", "cji_var", "bts_var"] and
    #         args.data.max_length % tokenizer.trunc_num != 0):
    #     args.data.max_length = args.data.max_length - (args.data.max_length % tokenizer.trunc_num)
    #     tokenizer.max_length = args.data.max_length

        logger.info(f"Change the max_length for generation to {args.model.generation_config.max_length}.")
        logger.info(f"Change the max_new_tokens for generation to {args.model.generation_config.max_new_tokens}.")

    # Load the Dataset & DataLoader
    dataloaders = get_nlg_dataloader(args, tokenizer, logger)

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
        setattr(model.generation_config, "use_cache", True)
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

    # Set basic settings for training
    setup_basics(args)

    trainer = GPT2NLGTrainer(args, accelerator, logger, tokenizer, model, dataloaders)

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
    logger.info(f"max length            : {args.model.generation_config.max_length}")
    logger.info(f"max new tokens        : {args.model.generation_config.max_new_tokens}")
    logger.info(f"repetition_penalty    : {args.model.generation_config.repetition_penalty}\n")
    # logger.info(f"no_repeat_ngram_size  : {args.model.generation_config.no_repeat_ngram_size}\n")

    if args.model.ckpt_dir:
        logger.info(f"ckpt dir              : {args.model.ckpt_dir}")
    logger.info(f"log dir             : {args.logging.log_dir}")
    logger.info(f"save dir            : {args.logging.save_dir}")
    logger.info(f"tb dir              : {args.logging.tb_dir}")
    logger.info(f"tb interval         : {args.logging.log_steps}\n")

    # Run training
    print("\n")
    logger.info("\n")
    logger.info("Start the Training !")
    trainer.train()
    logger.info("Fine-tuning is done!")


if __name__ == "__main__":
    main()
