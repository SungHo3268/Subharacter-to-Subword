import os
import sys
import time
import evaluate
import torch
import torch.nn as nn
from tqdm import tqdm
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from srcs.functions import BAR_FORMAT
from utils.logging_utils import Averager


class GPT2NLGTrainer(nn.Module):
    def __init__(self, hparams, accelerator, logger, tokenizer, model, dataloaders: dict):
        super(GPT2NLGTrainer, self).__init__()
        self.hparams = hparams
        self.logger = logger
        self.device = torch.device('cuda' if hparams.device == 'gpu' and torch.cuda.is_available() else 'cpu')

        self.tokenizer = tokenizer

        self.train_dataloader = dataloaders['train']
        self.dev_dataloader = dataloaders['dev']
        self.test_dataloader = dataloaders['test']

        self.model = model
        self.model.to(self.device)

        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_lr_scheduler()
        self.accelerator = accelerator

        self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.dev_dataloader, self.test_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.train_dataloader, self.dev_dataloader, self.test_dataloader
        )

        self.tb_writer = SummaryWriter(self.hparams.logging.tb_dir)

        self.current_train_step = 0
        self.current_epoch = 0

        self.epoch_done = False
        self.last_log = time.time()

        self.best_score = {
            "best_epoch": 0,
            "best_dev_score": {},
            "best_test_score": {},
        }

    def get_optimizer(self):
        no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.optim.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.hparams.optim.name == 'adamw':
            from transformers import AdamW
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.hparams.optim.base_lr,
            )
        elif self.hparams.optim.name == 'adamwscale':
            from srcs.functions import AdamWScale
            optimizer = AdamWScale(
                optimizer_grouped_parameters,
                lr=self.hparams.optim.base_lr,
            )
        elif self.hparams.optim.name == 'adafactor':
            from transformers import Adafactor
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.hparams.optim.base_lr,
                relative_step=False,
            )
        else:
            raise NotImplementedError

        return optimizer

    def get_lr_scheduler(self):
        if self.hparams.optim.lr_scheduler == 'cosine':
            from torch.optim.lr_scheduler import (
                SequentialLR,
                LinearLR,
                CosineAnnealingLR,
            )
            scheduler1 = LinearLR(
                self.optimizer,
                start_factor=0.5,
                end_factor=1,
                total_iters=self.hparams.optim.warmup_steps,
                last_epoch=-1,
            )
            scheduler2 = CosineAnnealingLR(
                self.optimizer,
                T_max=self.hparams.optim.total_steps - self.hparams.optim.warmup_steps,
                eta_min=self.hparams.optim.final_cosine,
            )
            lr_scheduler = SequentialLR(
                self.optimizer,
                schedulers=[scheduler1, scheduler2],
                milestones=[self.hparams.optim.warmup_steps]
            )

        elif self.hparams.optim.lr_scheduler == 'legacy':
            import math
            from torch.optim.lr_scheduler import (
                SequentialLR,
                LinearLR,
                LambdaLR,
            )

            num_steps_optimizer1 = math.ceil(self.hparams.optim.total_steps * 0.9)
            iters_left_for_optimizer2 = self.hparams.optim.total_steps - num_steps_optimizer1

            scheduler1 = LambdaLR(
                self.optimizer,
                lambda step: min(
                    1e-2, 1.0 / math.sqrt(step)
                ) / self.hparams.optim.base_lr if step else 1e-2 / self.hparams.optim.base_lr
            )
            scheduler2 = LinearLR(
                self.optimizer,
                start_factor=(
                        min(1e-2, 1.0 / math.sqrt(num_steps_optimizer1)) / self.hparams.optim.base_lr
                ),
                end_factor=0,
                total_iters=iters_left_for_optimizer2,
                last_epoch=-1,
            )
            lr_scheduler = SequentialLR(
                self.optimizer,
                schedulers=[scheduler1, scheduler2],
                milestones=[num_steps_optimizer1]
            )
        elif self.hparams.optim.lr_scheduler == 'constant':
            from transformers import get_scheduler
            lr_scheduler = get_scheduler(
                name=self.hparams.optim.lr_scheduler,
                optimizer=self.optimizer,
            )
        elif self.hparams.optim.lr_scheduler == 'linear':
            from transformers import get_linear_schedule_with_warmup

            lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.hparams.optim.warmup_steps, num_training_steps=self.hparams.optim.total_steps
            )
        else:
            raise NotImplementedError

        return lr_scheduler

    def maybe_logging(self, averager, mode='train'):
        if (mode == 'train') and (len(averager.total) != 0) and ((self.current_train_step % self.hparams.logging.log_steps == 0) or self.epoch_done):
            stats = self.extra_stats(self.model, self.optimizer)
            averager.update(stats)

            averaged_stats = averager.average()     # it includes the reset status function

            messages = (f"Step: {self.current_train_step}/{self.hparams.optim.total_steps}  |  "
                        f"Loss: {averaged_stats['loss']:.4f}  |  "
                        f"Accuracy: {averaged_stats['accuracy'] * 100:.2f} [%]  |  "
                        f"Seq Length: {averaged_stats['seq_len']:.1f}"
                        )
            print("")
            self.logger.info(messages)
            self.last_log = time.time()

            self.tb_writer.add_scalar(f"{mode}_loss/step", averaged_stats['loss'], self.current_train_step)
            self.tb_writer.add_scalar(f"{mode}_acc/step", averaged_stats['accuracy'], self.current_train_step)
            self.tb_writer.add_scalar(f"{mode}_seq_len/step", averaged_stats['seq_len'], self.current_train_step)
            self.tb_writer.add_scalar(f"{mode}_lr/step", averaged_stats['lr'], self.current_train_step)
            self.tb_writer.add_scalar(f"{mode}_seconds_per_step/step", averaged_stats['seconds_per_step'], self.current_train_step)
            if self.hparams.logging.grad_l2:
                self.tb_writer.add_scalar(f"{mode}_grad_l2/step", averaged_stats['grad_l2'], self.current_train_step)
            if self.hparams.logging.weights_l2:
                self.tb_writer.add_scalar(f"{mode}_weights_l2/step", averaged_stats['weights_l2'], self.current_train_step)
            self.tb_writer.flush()

            if self.epoch_done: self.epoch_done = False
            return None

        elif mode in ['dev', 'test']:
            averaged_stats = averager.average()     # it includes the reset status function

            self.logger.info(f"########################  {mode.upper()} REPORT #EP{self.current_epoch}  ########################")
            messages = (f"Rouge2: {averaged_stats['rouge2'] * 100:.2f} [%]  |  "
                        f"RougeL: {averaged_stats['rougeL'] * 100:.2f} [%]  |  "
                        f"Evaluation Time: {averaged_stats['time']:.2f} [s]"
                        )

            self.logger.info(messages)
            self.last_log = time.time()

            self.tb_writer.add_scalar(f"{mode}_rouge2/step", averaged_stats['rouge2'], self.current_epoch)
            self.tb_writer.add_scalar(f"{mode}_rougeL/step", averaged_stats['rougeL'], self.current_epoch)
            self.tb_writer.add_scalar(f"{mode}_evaluation_time", averaged_stats['time'], self.current_epoch)
            self.tb_writer.flush()
            
            return {'rouge2': averaged_stats['rouge2'],
                    'rougeL': averaged_stats['rougeL']}

    def maybe_grad_clip_and_grad_calc(self, model):
        if self.hparams.optim.grad_clip > 0:
            grad_l2 = self.accelerator.clip_grad_norm_(
                parameters=model.parameters(),
                max_norm=self.hparams.optim.grad_clip,
                norm_type=2,
            )
        else:
            grad_l2 = None

        if self.hparams.logging.grad_l2:
            if grad_l2 is None:
                grad_l2 = (
                        sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
                )
            return {'grad_l2': grad_l2}
        else:
            return {}

    def extra_stats(self, model, optimizer):
        stats = {}
        if self.hparams.logging.weights_l2:
            weights_l2 = sum(p.detach().norm(2).item() ** 2 for p in model.parameters()) ** 0.5
            stats['weights_l2'] = weights_l2

        stats['lr'] = optimizer.param_groups[0]['lr']
        stats['seconds_per_step'] = (time.time() - self.last_log) / self.hparams.logging.log_steps
        return stats

    def forward(self, batch, calc_acc=False):
        outputs = self.model(**batch, output_hidden_states=True)
        loss = outputs.loss
        seq_len = outputs.hidden_states[-2].shape[1]
        stats = {'loss': loss.detach().float().item(),
                 'seq_len': seq_len
                 }
        if calc_acc:
            correct = (outputs.logits.argmax(-1) == batch["labels"]).sum().item()
            accuracy = correct / batch["labels"].numel()
            stats['accuracy'] = accuracy
        return loss, stats

    def predict(self, eval_dataloader, mode):
        self.last_log = time.time()
        metric = evaluate.load('rouge')

        def decode(preds):
            preds[preds == -100] = self.tokenizer.pad_token_id
            preds = self.tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            preds = [_pred.strip() for _pred in preds]
            return preds

        # for step, batch in enumerate(eval_dataloader):
        for batch in tqdm(eval_dataloader, desc=f"Evaluating {mode}...", bar_format=BAR_FORMAT):
            predictions = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                generation_config=self.model.generation_config,
            )
            given_text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            predictions = decode(predictions)

            only_predictions = []
            for i in range(len(predictions)):
                pred = predictions[i]
                only_pred = pred[len(given_text[i]):].strip()

                if self.hparams.data.task_name == 'KoCommonGen':
                    end_of_text = only_pred.find('.')
                    if end_of_text == 0:
                        only_pred = only_pred[1:]
                        end_of_text = only_pred.find('.')
                        if (end_of_text == -1) or (end_of_text > len(only_pred) - 1):
                            end_of_text = len(only_pred) - 1
                    elif end_of_text == -1:
                        end_of_text = len(only_pred) - 1
                    else:
                        pass
                    only_pred = only_pred[: end_of_text + 1].strip()
                elif self.hparams.data.task_name == 'XL_Sum':
                    pass
                else:
                    raise NotImplementedError("This task is not supported. Please check the generation code in the predict function.")

                only_predictions.append(only_pred)

            references = decode(batch["labels"])

            for i in range(len(only_predictions)):
                print("\n")
                print(f"Prediction: {only_predictions[i]}")
                if only_predictions[i] == '':
                    print(f"Original_Prediction: {predictions[i]}")
                    only_predictions[i] = predictions[i]
                print(f"Reference: {references[i]}")

            metric.add_batch(
                predictions=only_predictions,
                references=references,
            )

        if not self.hparams.model.hf_model and self.hparams.data.tok_type in ['jamo_var', 'stroke_var', 'cji_var', 'bts_var']:
            eval_metric = metric.compute(tokenizer=lambda x: [tok for tok in self.tokenizer.tokenize(x) if tok != self.tokenizer.custom_tokenizer.empty_jamo], use_stemmer=True, use_aggregator=False)
        else:
            eval_metric = metric.compute(tokenizer=lambda x: self.tokenizer.tokenize(x), use_stemmer=True, use_aggregator=False)
        # eval_metric = metric.compute(tokenizer=lambda x: x.split(), use_stemmer=True, use_aggregator=False)
        rougeL = sum(eval_metric["rougeL"]) / len(eval_metric["rougeL"])
        rouge2 = sum(eval_metric["rouge2"]) / len(eval_metric["rouge2"])
        eval_stats = {"rougeL": rougeL,
                      "rouge2": rouge2,
                      "time": time.time() - self.last_log}
        return eval_stats

    def evaluation(self, eval_dataloader, mode='dev'):
        self.model.eval()
        with torch.no_grad():
            self.last_log = time.time()
            eval_averager = Averager()

            eval_stats = self.predict(eval_dataloader, mode)
            eval_averager.update(eval_stats)

            eval_score = self.maybe_logging(eval_averager, mode=mode)
        return eval_score

    def train(self, **kwargs):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        train_averager = Averager()

        current_stack = 0
        for _ in range(self.hparams.optim.epochs):
            self.model.train()
            self.current_epoch += 1
            print("\n")
            self.logger.info(f"\n[{self.current_epoch} Epoch]")
            for batch in tqdm(self.train_dataloader, total=len(self.train_dataloader), desc=f"Fine-Tuning...", bar_format=BAR_FORMAT):
                loss, stats = self.forward(batch, calc_acc=True)

                self.accelerator.backward(loss / self.hparams.optim.grad_acc)
                train_averager.update(stats)

                current_stack += 1
                if current_stack == self.hparams.optim.grad_acc:
                    stats = self.maybe_grad_clip_and_grad_calc(self.model)
                    train_averager.update(stats)

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.current_train_step += 1
                    current_stack = 0

                    self.maybe_logging(train_averager, mode='train')
                    # self.maybe_save_checkpoint()

            self.epoch_done = True
            self.maybe_logging(train_averager, mode='train')

            dev_score = self.evaluation(self.dev_dataloader, mode='dev')
            test_score = self.evaluation(self.test_dataloader, mode='test')

            if (sum(list(dev_score.values())) + sum(list(test_score.values()))) >= (sum(list(self.best_score['best_dev_score'].values())) + sum(list(self.best_score['best_test_score'].values()))):
                self.logger.info(f"\nSave new Best Score (Epoch: {self.current_epoch})")
                self.best_score['best_epoch'] = self.current_epoch
                self.best_score['best_dev_score'] = dev_score
                self.best_score['best_test_score'] = test_score

                # self.logger.info(f"\nSave new Best Model (Epoch: {self.current_epoch})")
                # save_file(self.model.state_dict(), os.path.join(self.hparams.logging.save_dir, "model.safetensors"))

        print("\n")
        self.logger.info("########################  BEST RESULT  ########################")
        self.logger.info(f"Epoch: {self.best_score['best_epoch']}")
        self.logger.info("")
        for key in self.best_score['best_dev_score']:
            self.logger.info(f"DEV - {key.upper()}: {self.best_score['best_dev_score'][key] * 100:.2f} [%]")
        self.logger.info("")
        for key in self.best_score['best_test_score']:
            self.logger.info(f"TEST - {key.upper()}: {self.best_score['best_test_score'][key] * 100:.2f} [%]")
        self.logger.info("##############################################################")