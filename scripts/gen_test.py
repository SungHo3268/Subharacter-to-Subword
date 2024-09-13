import os
import sys
import hydra
from datasets import Dataset
from safetensors import safe_open
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoConfig, DataCollatorWithPadding, GPT2LMHeadModel

sys.path.append(os.getcwd())
from utils.gen_utils import setup_basics
from nlu_tasks.srcs.trainer import GPT2NLUTrainer
from srcs.gpt2_utils import text_tokenization_for_classification
from pretraining.scripts.run_pretraining import set_logger, get_gpt2_tokenizer


def get_nlu_dataloader(args, tokenizer, logger):
    if args.data.task_name == 'KorNLI':
        from nlu_tasks.data_utils.KorNLI.data_utils import load_task_dataset
    elif args.data.task_name == 'KorSTS':
        from nlu_tasks.data_utils.KorSTS.data_utils import load_task_dataset
    elif args.data.task_name == 'NSMC':
        from nlu_tasks.data_utils.NSMC.data_utils import load_task_dataset
    elif args.data.task_name == 'PAWS_X':
        from nlu_tasks.data_utils.PAWS_X.data_utils import load_task_dataset
    else:
        logger.info(
            "It's a Wrong Task Name. Please enter the right task name among [KorNLI, KorSTS, NSMC, PAWS_X]")
        raise ValueError
    dataset = load_task_dataset(args.data.remain_lang, args.data.do_hangeulize, args.data.data_remove)

    train_data = Dataset.from_dict(dataset['train'])
    dev_data = Dataset.from_dict(dataset['dev'])
    test_data = Dataset.from_dict(dataset['test'])
    label_map = dataset['label_map']

    train_tokenized_datasets = train_data.map(text_tokenization_for_classification,
                                              fn_kwargs={"tokenizer": tokenizer,
                                                         "max_length": args.data.max_length},
                                              remove_columns=train_data.column_names,
                                              batched=True,
                                              batch_size=args.optim.batch_size // args.optim.grad_acc,
                                              )
    dev_tokenized_datasets = dev_data.map(text_tokenization_for_classification,
                                          fn_kwargs={"tokenizer": tokenizer,
                                                     "max_length": args.data.max_length},
                                          remove_columns=dev_data.column_names,
                                          batched=True,
                                          batch_size=args.optim.batch_size // args.optim.grad_acc,
                                          )
    test_tokenized_datasets = dev_data.map(text_tokenization_for_classification,
                                           fn_kwargs={"tokenizer": tokenizer,
                                                      "max_length": args.data.max_length},
                                           remove_columns=test_data.column_names,
                                           batched=True,
                                           batch_size=args.optim.batch_size // args.optim.grad_acc,
                                           )

    data_collator = DataCollatorWithPadding(tokenizer)

    train_dataloader = DataLoader(
        train_tokenized_datasets,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.optim.batch_size // args.optim.grad_acc,
        num_workers=args.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    dev_dataloader = DataLoader(
        dev_tokenized_datasets,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.optim.batch_size // args.optim.grad_acc,
        num_workers=args.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_dataloader = DataLoader(
        test_tokenized_datasets,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.optim.batch_size // args.optim.grad_acc,
        num_workers=args.data.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    total_dataloader = {"train": train_dataloader, "dev": dev_dataloader, "test": test_dataloader, "label_map": label_map}

    return total_dataloader


def get_config_and_nlu_model(args, tokenizer):
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

    model = GPT2LMHeadModel(config)

    # reload the checkpoint of the pre-trained model
    if args.model.ckpt_dir:
        print("\n")
        model_path = os.path.join(args.model.ckpt_dir, "model.safetensors")
        state_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key.replace("_orig_mod.transformer.", "")] = f.get_tensor(key)
        model.transformer.load_state_dict(state_dict)
    #TODO: Add the loading function for fine-tuned model, not pre-trained model
    return config, model


@hydra.main(config_path="../configs/gpt2", config_name="default", version_base='1.1')
def main(args):
    if args.data.task_name in ["KorSTS"]:
        args.data.num_labels = 1
    elif args.data.task_name in ["NSMC", "PAWS_X"]:
        args.data.num_labels = 2
    elif args.data.task_name in ["KorNLI"]:
        args.data.num_labels = 3
    elif args.data.task_name in ["KorQuAD"]:
        args.data.num_labels = -1
    else:
        raise ValueError("It's a Wrong Task Name. Please enter the right task name among [KorQuAD, KorNLI, KorSTS, NSMC, PAWS_X]")

    assert args.optim.batch_size % args.optim.grad_acc == 0, "batch size should be divisible by gradient accumulation steps."


    # Get the Tokenizer
    tokenizer = get_gpt2_tokenizer(tok_type=args.data.tok_type,
                                   lang=args.data.language,
                                   max_length=args.data.max_length,
                                   lowercase=True,
                                   clean_text=True)
    tokenizer.pad_token = tokenizer.eos_token
    if (args.data.tok_type in ["jamo_var", "stroke_var", "cji_var", "bts_var"] and
            args.data.max_length % tokenizer.trunc_num != 0):
        # args.data.max_length = args.data.max_length - (args.data.max_length % tokenizer.trunc_num)
        tokenizer.max_length = args.data.max_length


    # Set the Config and Model
    config, model = get_config_and_nlu_model(args, tokenizer)

    text = '근육이 커지기 위해서는'
    input_ids = tokenizer.encode(text, return_tensors='pt')
    gen_ids = model.generate(input_ids,
                             max_length=128,
                             repetition_penalty=2.0,
                             pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id,
                             bos_token_id=tokenizer.bos_token_id,
                             use_cache=True)
    generated = tokenizer.decode(gen_ids[0])
    print(generated)
    exit(-111)


if __name__ == "__main__":
    main()
