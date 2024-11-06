import os
import sys
import time
import json
import torch
import hydra
from omegaconf import open_dict
from datasets import load_dataset
from transformers import GPT2LMHeadModel, AutoConfig
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator

sys.path.append(os.getcwd())
from tokenization.srcs.functions import get_tokenizer
from srcs.gpt2_tokenizers import KoGPT2Tokenizer
from srcs.gpt_utils import check_pretraining_data, doc_tokenization
from pretraining.srcs.trainer import GPT2Trainer
from utils.gen_utils import setup_basics
# from utils.logging_utils import Logger
from utils.logger import get_logger

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_gpt2_tokenizer(tok_type, lang="ko", max_length=512, lowercase=True, clean_text=True, add_bos_token=True, padding_side="right",
                       bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|endoftext|>", unk_token="<|endoftext|>"):
    if tok_type in ['jamo', 'jamo_var', 'stroke', 'stroke_var', 'cji', 'cji_var', 'bts', 'bts_var']:
        tok_vocab_size = "200"
    elif tok_type in ['char']:
        tok_vocab_size = "2k"
    elif tok_type in ['morpheme', 'subword', 'morphemeSubword']:
        tok_vocab_size = "32k"
    elif tok_type in ['word']:
        tok_vocab_size = "64k"
    else:
        raise NotImplementedError(f"Invalid tokenization type: {tok_type}")

    custom_tokenizer = get_tokenizer(tok_type)

    if tok_type in ['subword', 'morphemeSubword']:
        tok_name = f"{tok_type}_{lang}_wiki_{tok_vocab_size}"
        custom_tokenizer.load_model(f"tokenization/resources/{tok_name}/tok.model")
    else:
        tok_name = f"{tok_type}_{lang}_{tok_vocab_size}"

    vocab_file = f"tokenization/resources/{tok_name}/tok.vocab"
    tokenizer = KoGPT2Tokenizer(vocab_file=vocab_file,
                                custom_tokenizer=custom_tokenizer,
                                max_length=max_length,
                                padding_side=padding_side,
                                lowercase=lowercase,
                                clean_text=clean_text,
                                add_bos_token=add_bos_token,
                                bos_token=bos_token,
                                eos_token=eos_token,
                                pad_token=pad_token,
                                unk_token=unk_token,
                                )
    return tokenizer


def get_dataloader(args, tokenizer):
    # Check the pretraining data
    check_pretraining_data(raw_data_path=args.data.raw_data_path, doc_split=args.data.split_by_doc, toyset=args.data.is_toyset)

    # Load the dataset
    dataset_path = args.data.train_data_path
    if args.data.is_toyset:
        dataset_path = dataset_path.replace(".txt", "_toy.txt")

    raw_datasets = load_dataset("text", data_files=dataset_path, split="train", streaming=True)
    tokenized_datasets = raw_datasets.map(doc_tokenization,
                                          fn_kwargs={"tokenizer": tokenizer,
                                                     "max_length": args.data.max_length},
                                          remove_columns=["text"],
                                          batched=True,
                                          batch_size=args.optim.batch_size//args.optim.grad_acc,
                                          )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    dataloader = DataLoader(
        tokenized_datasets,
        shuffle=not isinstance(tokenized_datasets, IterableDataset),
        collate_fn=data_collator,
        batch_size=args.optim.batch_size // args.optim.grad_acc,
        num_workers=args.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader


def get_config_and_model(args, tokenizer, logger=None):
    # Set the config and model
    config = AutoConfig.from_pretrained(
        args.model.name,
        vocab_size=len(tokenizer),
        n_ctx=args.data.max_length,
        n_positions=args.model.n_positions,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)

    model_size = sum(t.numel() for t in model.parameters())
    logger.info(f"GPT-2 size: {model_size / 1000 ** 2:.1f}M parameters")

    if args.model.ckpt_dir:
        with open(os.path.join(args.model.ckpt_dir, 'args.json'), 'r') as fr:
            load_state = json.load(fr)
        fr.close()

        with open_dict(args):
            args.current_train_step = load_state["current_train_step"]
            args.current_epoch = load_state["current_epoch"]
            args.train_done = load_state["train_done"]
            args.last_log = time.time()
    else:
        with open_dict(args):
            args.current_train_step = 0
            args.current_epoch = 1
            args.train_done = False
            args.last_log = time.time()

    if args.model.compile:
        logger.info("Compile! (torch 2.0)")
        model = torch.compile(model)

    return config, model


def set_logger(args, tqdm_handler=True):
    logger = get_logger(log_path=os.path.join(args.logging.log_dir, "train_log.txt"), tqdm_handler=tqdm_handler)
    logger.info("\n")
    logger.info("This train_log.txt inform the Running Progress.\n")
    logger.info(f"Save the parser information to {args.logging.log_dir}")

    logger.info("\n")
    logger.info(f"Arguments: {args}\n")

    # with open(os.path.join(args.logging.log_dir, 'argparse.json'), 'w') as fw:
    #     json.dump(dict(args), fw, indent=2)
    #     fw.close()
    return logger



# @hydra.main(config_path="../../configs/gpt", config_name="default", version_base='1.1')
@hydra.main(config_path=os.path.join(os.getcwd(), "configs/gpt"), config_name="default", version_base='1.1')
def main(args):
    logger = set_logger(args, tqdm_handler=False)

    logger.info("\n")
    logger.info(f"* [sys] Current available # GPUs: {torch.cuda.device_count()}\n")
    logger.info(f"* [sys] Current working directory: {os.getcwd()}\n")

    # ----------------------------------------
    #           Set the Trainer
    # ----------------------------------------
    # Get the Tokenizer
    tokenizer = get_gpt2_tokenizer(tok_type=args.data.tok_type,
                                   lang=args.data.language,
                                   max_length=args.data.max_length,
                                   lowercase=True,
                                   clean_text=True)
    tokenizer.pad_token = tokenizer.eos_token

    if (args.data.tok_type in ["jamo_var", "stroke_var", "cji_var", "bts_var"] and
            args.data.max_length % tokenizer.trunc_num != 0):
        # tokenizer.max_length = args.data.max_length - (args.data.max_length % tokenizer.trunc_num)
        tokenizer.max_length = args.data.max_length

        logger.info(f"Change the max_length to {args.data.max_length} for the tokenizer's truncation.")


    # Load the Dataset & DataLoader
    dataloader = get_dataloader(args, tokenizer)

    # Set the Config and Model
    config, model = get_config_and_model(args, tokenizer, logger)

    # Set the Accelerator
    accelerator = Accelerator(
        cpu=(args.device == "cpu"),
        mixed_precision=args.mixed_precision
    )

    # Set basic settings for training
    setup_basics(args)

    # logger = Logger(args=args, accelerator=accelerator)
    # logger.log_args(config)

    trainer = GPT2Trainer(args, accelerator, logger, tokenizer, model, dataloader)


    # ----------------------------------------
    #               Do Pre-Training
    # ----------------------------------------
    logger.info("\n")
    logger.info("========== Pre-training ==========")
    logger.info(f"model                 : {args.model.name}")
    logger.info(f"tokenizer             : {args.data.tok_type}")
    logger.info(f"vocab size            : {args.data.vocab_size}")
    logger.info(f"device                : {args.device}")
    logger.info(f"random seed           : {args.seed}")
    logger.info(f"total steps           : {args.optim.total_steps}")
    logger.info(f"warmup steps          : {args.optim.warmup_steps}")
    logger.info(f"batch size            : {args.optim.batch_size}")
    logger.info(f"accumulation steps    : {args.optim.grad_acc}")
    logger.info(f"max seq len           : {args.data.max_length}")
    logger.info(f"optimizer             : {args.optim.name}")
    logger.info(f"lr_scheduler          : {args.optim.lr_scheduler}")
    logger.info(f"learning rate         : {args.optim.base_lr}\n")

    if args.model.ckpt_dir:
        logger.info(f"ckpt dir              : {args.model.ckpt_dir}")
    logger.info(f"* log dir             : {args.logging.log_dir}")
    logger.info(f"* save dir            : {args.logging.save_dir}")
    logger.info(f"* save interval       : {args.checkpoint.save_steps}")
    logger.info(f"* tb dir              : {args.logging.tb_dir}")
    logger.info(f"* tb interval         : {args.logging.log_steps}\n")

    # Run training
    logger.info("\n")
    logger.info("Start the Training !")
    trainer.train()

    # logger.finish()

    logger.info("Save the final trained model!")
    torch.save(model.state_dict(), os.path.join(args.logging.save_dir, 'pytorch_model.bin'))
    logger.info("Pre-training is done!")


if __name__ == "__main__":
    main()
