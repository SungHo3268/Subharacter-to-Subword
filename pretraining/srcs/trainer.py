import os
import sys
import time
import json
import shutil
import evaluate
import torch
import torch.nn as nn
from datasets.iterable_dataset import IterableDataset
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from utils.logging_utils import Averager


class GPT2Trainer(nn.Module):
    def __init__(self, hparams, accelerator, logger, tokenizer, model, dataloader):
        super(GPT2Trainer, self).__init__()
        self.hparams = hparams
        self.logger = logger
        self.device = torch.device('cuda' if hparams.device == 'gpu' and torch.cuda.is_available() else 'cpu')

        self.tokenizer = tokenizer

        self.dataloader = dataloader

        self.model = model
        self.model.to(self.device)

        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_lr_scheduler()
        self.accelerator = accelerator

        self.model, self.optimizer, self.lr_scheduler, self.dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler, self.dataloader
        )

        if hparams.model.ckpt_dir:
            accelerator.load_state(hparams.model.ckpt_dir)
            self.reload = True
        else:
            self.reload = False

        if not os.path.exists(self.hparams.logging.tb_dir):
            os.makedirs(self.hparams.logging.tb_dir)
        self.tb_writer = SummaryWriter(self.hparams.logging.tb_dir)

        self.current_train_step = hparams.current_train_step
        self.current_epoch = hparams.current_epoch

        self.last_log = hparams.last_log

        self.train_done = hparams.train_done

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

    def maybe_save_checkpoint(self):
        if self.current_train_step % self.hparams.checkpoint.save_steps == 0:
            output_dir = os.path.join(self.hparams.logging.save_dir, f'checkpoint-{self.current_train_step}')

            self.accelerator.save_state(output_dir=output_dir)
            if os.path.exists(os.path.join(self.hparams.logging.save_dir, f'checkpoint-{self.current_train_step - (self.hparams.checkpoint.save_steps * self.hparams.checkpoint.max_number)}')):
                shutil.rmtree(os.path.join(self.hparams.logging.save_dir, f'checkpoint-{self.current_train_step - (self.hparams.checkpoint.save_steps * self.hparams.checkpoint.max_number)}'))

            # torch.save(self.model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

            with open(os.path.join(output_dir, 'args.json'), 'w') as fw:
                args = {
                    "current_train_step": self.current_train_step,
                    "current_epoch": self.current_epoch,
                    "train_done": self.train_done,
                    "last_log": self.last_log,
                }
                json.dump(args, fw)

    def maybe_eval_predict(self, model, eval_dataloader):
        if (
                self.current_train_step > self.optim.total_steps
                or self.current_train_step % self.hparams.eval.eval_steps == 0
        ):
            model.eval()

            with torch.no_grad():
                self.evaluation(eval_dataloader)

            self.last_log = time.time()
            model.train()

    def maybe_logging(self, averager, model, optimizer):
        if self.current_train_step % self.hparams.logging.log_steps == 0:
            stats = self.extra_stats(model, optimizer)

            averager.update(stats)
            averaged_stats = averager.average()     # it includes the reset status function

            # self.logger.log_stats(
            #     stats=averaged_stats,
            #     step=self.current_train_step,
            #     total_steps=self.hparams.optim.total_steps,
            #     prefix='train/'
            # )
            self.logger.info(f"* Step: {self.current_train_step}/{self.hparams.optim.total_steps} | "
                             f"Loss: {averaged_stats['loss']:.4f} | "
                             f"LR: {averaged_stats['lr']:.4f} | "
                             f"Seconds per step: {averaged_stats['seconds_per_step']:.4f}")

            self.last_log = time.time()

            self.tb_writer.add_scalar(f"pretraining_loss/step", averaged_stats['loss'], self.current_train_step)
            self.tb_writer.add_scalar(f"pretraining_lr/step", averaged_stats['lr'], self.current_train_step)
            self.tb_writer.add_scalar(f"pretraining_seconds_per_step/step", averaged_stats['seconds_per_step'],
                                      self.current_train_step)
            self.tb_writer.flush()

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
        outputs = self.model(**batch)
        loss = outputs.loss
        stats = {'loss': loss.detach().float().item()}
        if calc_acc:
            correct = (outputs.logits.argmax(-1) == batch["labels"]).sum().item()
            accuracy = correct / batch["labels"].numel()
            stats['accuracy'] = accuracy
        return loss, stats

    def evaluation(self, eval_dataloader):
        self.model.eval()
        with torch.no_grad():
            self.last_log = time.time()
            averager = Averager()
            for batch_id, batch in enumerate(eval_dataloader, start=1):
                if batch_id == self.hparams.eval.corrected_steps * self.hparams.optim.grad_acc:
                    break
                _, stats = self.forward(batch, calc_acc=True)
                averager.update(stats)

            averager.update({'time': time.time() - self.last_log})
            averaged_stats = averager.average()

            # self.logger.log_stats(
            #     stats=averaged_stats,
            #     step=self.current_train_step,
            #     total_steps=self.hparams.optim.total_steps,
            #     prefix='eval/'
            # )
            self.logger.info(f"* Step: {self.current_train_step}/{self.hparams.optim.total_steps} | "
                             f"Loss: {averaged_stats['loss']:.4f} | "
                             f"LR: {averaged_stats['lr']:.4f} | "
                             f"Seconds per step: {averaged_stats['seconds_per_step']:.4f}")

            self.tb_writer.add_scalar(f"pretraining_loss/step", averaged_stats['loss'], self.current_train_step)
            self.tb_writer.add_scalar(f"pretraining_lr/step", averaged_stats['lr'], self.current_train_step)
            self.tb_writer.add_scalar(f"pretraining_seconds_per_step/step", averaged_stats['seconds_per_step'], self.current_train_step)
            self.tb_writer.flush()


    def predict(self, dataloader):
        self.last_log = time.time()
        metric = evaluate.load('rouge')
        samples_seen = 0

        def decode(preds):
            preds[preds == -100] = self.tokenizer.pad_token_id
            preds = self.tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            preds = [pred.strip() for pred in preds]
            return preds

        for step, batch in enumerate(dataloader):
            predictions = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=self.hparams.data.max_target_len,
                generation_config=self.model.generation_config,
            )
            predictions = decode(predictions)
            references = decode(batch["labels"])

            # If we are in a multiprocess environment, the last batch has duplicates
            if step == len(dataloader) - 1:
                predictions = predictions[: len(dataloader.dataset) - samples_seen]
                references = references[: len(dataloader.dataset) - samples_seen]
            else:
                samples_seen += len(references)

            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute(use_stemmer=True, use_aggregator=False)
        rougeL = sum(eval_metric["rougeL"]) / len(eval_metric["rougeL"]) * 100

        # self.logger.log_stats(
        #     stats={
        #         "rougeL": rougeL,
        #         "time": time.time() - self.last_log,
        #     },
        #     step=self.current_train_step,
        #     total_steps=self.hparams.optim.total_steps,
        #     prefix="test/",
        # )

        self.logger.info(f"* Step: {self.current_train_step}/{self.hparams.optim.total_steps} | "
                         f"rougeL: {rougeL:.4f} | "
                         f"time: {time.time() - self.last_log:.4f} | "
                         )

    def train(self, **kwargs):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        train_averager = Averager()

        current_stack = 0
        self.train_done = False
        if isinstance(self.dataloader.dataset, IterableDataset):
            self.dataloader.dataset.set_epoch(1)

        while self.current_train_step <= self.hparams.optim.total_steps:
            if self.reload:
                skip_batch_num = self.current_train_step * self.hparams.optim.grad_acc
                skipped_dataloader = self.accelerator.skip_first_batches(self.dataloader, skip_batch_num)

                for batch in skipped_dataloader:
                    if self.current_train_step >= self.hparams.optim.total_steps - 1:
                        self.train_done = True
                        break

                    loss, stats = self.forward(batch)

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

                        self.maybe_logging(train_averager, self.model, self.optimizer)
                        self.maybe_save_checkpoint()

                self.reload = False

            else:
                for batch in self.dataloader:
                    if self.current_train_step > self.hparams.optim.total_steps:
                        self.train_done = True
                        break

                    loss, stats = self.forward(batch)

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

                        self.maybe_logging(train_averager, self.model, self.optimizer)
                        self.maybe_save_checkpoint()

            if not self.train_done:
                self.current_epoch += 1

        if self.train_done:
            self.maybe_save_checkpoint()
