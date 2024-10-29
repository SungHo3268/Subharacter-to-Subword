import os
import sys
import hydra
import importlib
from datasets import Dataset
from safetensors import safe_open
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoConfig, DataCollatorWithPadding, GPT2ForSequenceClassification, GPT2DoubleHeadsModel, AutoTokenizer

sys.path.append(os.getcwd())
from utils.gen_utils import setup_basics
from nlu_tasks.srcs.trainer import GPTNLUTrainer
from srcs.gpt_utils import text_tokenization_for_classification, text_tokenization_for_mc
from pretraining.scripts.run_pretraining import set_logger, get_gpt2_tokenizer
from srcs.lora import make_only_lora_as_trainable, print_trainable_parameters, apply_lora_to_model, LoRA_Config
from srcs.kombo import make_only_kombo_and_lora_as_trainable, apply_kombo_to_model, KOMBO_Config

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
            config = AutoConfig.from_pretrained(args.model.name)
            model = GPT2DoubleHeadsModel.from_pretrained(args.model.name, config=config)
        else:
            config = AutoConfig.from_pretrained(args.model.name, num_labels=args.data.num_labels)
            model = GPT2ForSequenceClassification.from_pretrained(args.model.name, config=config)

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
        #TODO: Add other models for LoRA (e.g., Llamma-3)
        if 'gpt2' in args.model.name:
            target_modules = ['c_attn', 'c_proj']
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

    if args.model.set_kombo:
        if 'gpt2' in args.model.name:
            target_modules = ['c_attn', 'c_proj']
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
        make_only_kombo_and_lora_as_trainable(model, weight='kombo_lora_only', bias='kombo_lora_only')
        _, _ = print_trainable_parameters(model, logger)
    return config, model


@hydra.main(config_path=os.path.join(os.getcwd(), "configs/gpt"), config_name="default", version_base='1.1')
def main(args):
    if args.model.hf_model:
        data_preprocess = f"{args.data.remain_lang}"
        if args.data.do_hangeulize: data_preprocess += "_dh"
        if args.data.data_remove: data_preprocess += "_dr"
        model_name = f"{args.model.name.replace('/', '_')}"
        if args.model.set_kombo: 
            model_name += "/kombo"
        elif args.model.set_lora: 
            model_name += "/lora"
        args.logging.log_dir = os.path.join(f"logs/{model_name}/nlu_tasks_lrs/{args.data.task_name}/{data_preprocess}/{args.data.max_length}t_{args.optim.batch_size}b_{args.optim.grad_acc}s_{args.seed}rs")
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
    best_score = []
    
    for base_lr in args.optim.base_lrs.split('_'):
        args.optim.base_lr = float(base_lr)
        
        # ----------------------------------------
        #           Set the Trainer
        # ----------------------------------------
        # Get the Tokenizer
        if args.model.hf_model:
            if args.model.name == 'skt/kogpt2-base-v2':
                tokenizer = AutoTokenizer.from_pretrained(args.model.name,
                                                        bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                        pad_token='<pad>', mask_token='<mask>')
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
        logger.info(f"Data Preprocessing")
        logger.info(f"ㄴ remain_lang                {args.data.remain_lang}")
        logger.info(f"ㄴ do_hangeulize              {args.data.do_hangeulize}")
        logger.info(f"ㄴ data_remove                {args.data.data_remove}\n")
        
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
        if args.model.set_kombo:
            logger.info(f"KOMBO Configuration")
            logger.info(f"ㄴ tok_type             : {args.model.kombo.tok_type}")
            logger.info(f"ㄴ hidden_dim           : {args.model.kombo.hidden_dim}")
            logger.info(f"ㄴ kombo_max_length     : {args.model.kombo.kombo_max_length}")
            logger.info(f"ㄴ embedding_norm       : {args.model.kombo.embedding_norm}")
            logger.info(f"ㄴ do_combination       : {args.model.kombo.do_combination}")
            if args.model.kombo.do_combination:
                logger.info(f"  ㄴ num_attn_heads     : {args.model.kombo.combination.num_attention_heads}")
                logger.info(f"  ㄴ intermediate_size  : {args.model.kombo.combination.intermediate_size}")
                logger.info(f"  ㄴ num_trans_layers   : {args.model.kombo.combination.num_trans_layers}")
                logger.info(f"  ㄴ add_lora           : {args.model.kombo.add_lora}\n")
            else:
                logger.info(f"ㄴ reducer              : {args.model.kombo.reducer}\n")
            if args.model.kombo.add_lora:
                logger.info(f"LoRA in KOMBO Configuration")
                logger.info(f"ㄴ r                : {args.model.kombo.lora.r}")
                logger.info(f"ㄴ alpha            : {args.model.kombo.lora.alpha}")
                logger.info(f"ㄴ dropout          : {args.model.kombo.lora.dropout}\n")
        logger.info('\n')
        if args.model.ckpt_dir:
            logger.info(f"ckpt dir        : {args.model.ckpt_dir}")
        logger.info(f"* log dir       : {args.logging.log_dir}")
        logger.info(f"* save dir      : {args.logging.save_dir}")
        logger.info(f"* tb dir        : {args.logging.tb_dir}")
        logger.info(f"* tb interval   : {args.logging.log_steps}\n")

        # best_score = {'lr': [], 'dev_score': [], 'test_score': []}
        
        # Run training
        print("\n")
        logger.info("\n")
        logger.info("Start the Training !")
        trainer.train()
        logger.info("Fine-tuning is done!")
        
        # best_score['lr'].append(args.optim.base_lr)
        # best_score['dev_score'].append(trainer.best_score['best_dev_score'])
        # best_score['test_score'].append(trainer.best_score['best_test_score'])
        best_score.append({'lr': args.optim.base_lr, \
                           'dev_score': trainer.best_score['best_dev_score'], \
                           'test_score': trainer.best_score['best_test_score']})
    
    best_lr = 0
    best_dev_score = 0
    best_test_score = 0
    for i, score in enumerate(best_score):
        if score['dev_score'] + score['test_score'] >= (best_dev_score + best_test_score):
            best_lr = score['lr']
            best_dev_score = score['dev_score']
            best_test_score = score['test_score']
        
    print("\n")
    logger.info("########################  BEST LR RESULT  ########################")
    logger.info(f"- BEST LR: {best_lr}")
    logger.info(f"- DEV score: {best_dev_score * 100:.2f} [%]")
    logger.info(f"- TEST score: {best_test_score * 100:.2f} [%]")


if __name__ == "__main__":
    main()
