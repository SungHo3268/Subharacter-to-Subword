import os
import sys
import time
import json
import shutil
from omegaconf import OmegaConf
import evaluate
import torch
import torch.nn as nn
from numpy.ma.extras import average
from sympy.testing.tests.test_code_quality import message_eof
from tqdm import tqdm
from safetensors.torch import save_file
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from srcs.functions import BAR_FORMAT
from nlg_tasks.srcs.evaluation_metrics import eval_main
from utils.m2scorer.scripts.m2scorer import m2score_main
from utils.gleu_scorer.gleumodule import run_gleu
from utils.logging_utils import Averager


class GPTNLGTrainer(nn.Module):
    def __init__(self, hparams, accelerator, logger, tokenizer, model, dataloaders: dict):
        super(GPTNLGTrainer, self).__init__()
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

        self.stop_cnt = 0
        self.early_stop = False

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
            if 'grad_l2' in averaged_stats:
                self.tb_writer.add_scalar(f"{mode}_grad_l2/step", averaged_stats['grad_l2'], self.current_train_step)
            if 'weights_l2' in averaged_stats:
                self.tb_writer.add_scalar(f"{mode}_weights_l2/step", averaged_stats['weights_l2'], self.current_train_step)
            self.tb_writer.flush()

            if self.epoch_done: self.epoch_done = False
            return None

        elif mode in ['dev', 'test']:
            averaged_stats = averager.average()     # it includes the reset status function

            self.logger.info(f"########################  {mode.upper()} REPORT #EP{self.current_epoch}  ########################")
            messages = ""
            for key in averaged_stats:
                if key == 'time':
                    messages += f"Evaluation Time: {averaged_stats[key]:.2f} [s]"
                else:
                    messages += f"{key.upper()}: {averaged_stats[key] * 100:.2f} [%]  |  "

            self.logger.info(messages)
            self.last_log = time.time()

            for key in averaged_stats:
                if key == 'time':
                    self.tb_writer.add_scalar(f"{mode}_evaluation_time", averaged_stats[key], self.current_epoch)
                else:
                    self.tb_writer.add_scalar(f"{mode}_{key}/step", averaged_stats[key], self.current_epoch)

            averaged_stats.pop('time')
            return averaged_stats

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

    def maybe_save_checkpoint(self):
        output_dir = os.path.join(self.hparams.logging.save_dir, f'checkpoint-best')

        self.accelerator.save_state(output_dir=output_dir)
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'torch.save.pytorch_model.bin'))
        json.dump(OmegaConf.to_container(self.hparams, resolve=True), open(os.path.join(output_dir, 'args.json'), 'w'))

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
        # metric = evaluate.load('rouge')

        def decode(preds):
            preds[preds == -100] = self.tokenizer.pad_token_id
            preds = self.tokenizer.batch_decode(
                preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            preds = [_pred.strip() for _pred in preds]
            return preds

        given_text_list = []
        preds_list = []
        refs_list = []
        concepts_list = []
        # for step, batch in enumerate(eval_dataloader):
        for batch in tqdm(eval_dataloader, desc=f"Evaluating {mode}...", bar_format=BAR_FORMAT):
            given_text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)
            if self.hparams.data.task_name == 'KoCommonGen':
                concepts = [[morph.strip() for morph in text.split(",")] for text in given_text]
                concepts = ["#".join(text) for text in concepts]
                concepts_list.extend(concepts)
            elif 'KoreanGEC' in self.hparams.data.task_name:
                trimmed_given_text = [text.split("수정:")[0].strip() for text in given_text]
                given_text_list.extend(trimmed_given_text)

            predictions = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                generation_config=self.model.generation_config,
            )
            predictions = decode(predictions)

            only_predictions = []
            for i in range(len(predictions)):
                pred = predictions[i]
                only_pred = pred[len(given_text[i]):].strip()
                if self.hparams.data.task_name in ['KoCommonGen', 'XL_Sum'] or 'KoreanGEC' in self.hparams.data.task_name:
                    if self.hparams.data.task_name == 'KoCommonGen':
                        end_of_text = only_pred.find('.')
                    elif self.hparams.data.task_name in ['XL_Sum']:
                        end_of_text = min(only_pred.find('.'), only_pred.find('?'))
                        if end_of_text == -1:
                            end_of_text = only_pred.find('.')
                    elif 'KoreanGEC' in self.hparams.data.task_name:
                        end_of_text = [only_pred.find('.'), only_pred.find('?'), only_pred.find('!'), only_pred.find('\n'), only_pred.find('\t')]
                        end_of_text = [end for end in end_of_text if end != -1]
                        if len(end_of_text) == 0:
                            end_of_text = -1
                        else:
                            end_of_text = min(end_of_text)
                    else:
                        raise NotImplementedError

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
                elif self.hparams.data.task_name == 'WikiLingua':
                    pass
                else:
                    raise NotImplementedError("This task is not supported. Please check the generation code in the predict function.")

                only_predictions.append(only_pred)
            preds_list.extend(only_predictions)

            references = decode(batch["labels"])
            if self.hparams.data.task_name == 'KoCommonGen' and mode == 'test':
                references = [ref.split(' = ') for ref in references]

            if type(references[0]) == str:
                references = [[ref] for ref in references]
            refs_list.extend(references)

            print("\n\n")
            print(f"given_text[0]: {given_text[0]}")
            print(f"references[0]: {references[0]}")
            print(f"only_predictions[0]: {only_predictions[0]}")

        if self.hparams.data.task_name == 'KoCommonGen':
            results = eval_main(refs_list, preds_list, concepts_list)
            eval_stats = results['total_avg']

        # length limit
        elif 'KoreanGEC' in self.hparams.data.task_name:
            refs_list = [ref[0] for ref in refs_list]
            if self.hparams.data.task_name == 'KoreanGEC_union':
                for k in range(len(preds_list)):
                    ref = refs_list[k]
                    pred = preds_list[k]
                    if len(pred) > len(ref) * 1.5:
                        preds_list[k] = pred[:len(ref)]

            # M2 Score
            paths = self.hparams.data.task_name.split('_')
            path_1 = paths[0]
            path_2 = '_'.join(paths[1:])
            if mode == 'dev':
                mode = 'val'

            json.dump({"preds": preds_list,
                       "refs": refs_list,
                       "srcs": given_text_list},
                      open(os.path.join(self.hparams.logging.log_dir, f"{mode}_{self.current_epoch}.json"), "w"),
                      ensure_ascii=True,
                      indent=2
                      )

            p, r, f1 = m2score_main(preds_list, f"datasets/nlg_tasks/{path_1}/{path_2}/{path_2}_{mode}.m2")
            # GLEU Score
            gleu_score = float(run_gleu(refs_list, given_text_list, preds_list))
            eval_stats = {'m2_precision': p, 'm2_recall': r, 'm2_f1_half': f1, 'gleu': gleu_score}
        else:
            results = eval_main(refs_list, preds_list, None)
            eval_stats = results['total_avg']
        eval_stats['time'] = time.time() - self.last_log
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
            self.logger.info(f"\n[{self.current_epoch}/ {self.hparams.optim.epochs} Epoch]")
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

            self.epoch_done = True
            self.maybe_logging(train_averager, mode='train')

            # dev_score = self.evaluation(self.dev_dataloader, mode='dev')
            test_score = self.evaluation(self.test_dataloader, mode='test')

            # if (sum(list(dev_score.values())) + sum(list(test_score.values()))) >= (sum(list(self.best_score['best_dev_score'].values())) + sum(list(self.best_score['best_test_score'].values()))):
            if sum(list(test_score.values())) >= sum(list(self.best_score['best_test_score'].values())):
                print("")
                self.logger.info(f"Save new Best Score (Epoch: {self.current_epoch})")
                self.best_score['best_epoch'] = self.current_epoch
                # self.best_score['best_dev_score'] = dev_score
                self.best_score['best_test_score'] = test_score
                self.stop_cnt = 0
                self.logger.info(f"The Best score is renewed. Stop Count Reset to 0")
                self.maybe_save_checkpoint()
            else:
                self.stop_cnt += 1
                print("")
                self.logger.info(f"Stop Count: {self.stop_cnt}")
                if self.stop_cnt == self.hparams.optim.early_stop_patience:
                    self.early_stop = True
                    self.logger.info(f"Early Stop !!!! Training Done.\n")
                    break

        print("\n")
        self.logger.info("########################  BEST RESULT  ########################")
        self.logger.info(f"Epoch: {self.best_score['best_epoch']}")
        self.logger.info("")
        for key in self.best_score['best_dev_score']:
            self.logger.info(f"DEV - {key.upper()}: {self.best_score['best_dev_score'][key] * 100:.2f} [%]")
        self.logger.info("")
        self.logger.info("")
        for key in self.best_score['best_test_score']:
            self.logger.info(f"TEST - {key.upper()}: {self.best_score['best_test_score'][key] * 100:.2f} [%]")
        self.logger.info("##############################################################")