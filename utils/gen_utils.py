import os
import sys
import torch
from omegaconf import open_dict
from accelerate.utils import set_seed
sys.path.append(os.getcwd())
from srcs.functions import init_random

def check_dir(args):
    os.makedirs(args.logging.log_dir, exist_ok=True)
    os.makedirs(args.logging.tb_dir, exist_ok=True)
    os.makedirs(args.logging.save_dir, exist_ok=True)

def check_args_and_env(args):
    assert args.optim.batch_size % args.optim.grad_acc == 0

    # Train log must happen before eval log
    # assert args.eval.eval_steps % args.logging.log_steps == 0

    if args.device == 'gpu':
        assert torch.cuda.is_available(), 'We use GPU to train/eval the model'

    assert not (args.eval_only and args.predict_only)


def opti_flags():
    # This lines reduce training step by 2.4x
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def update_args_with_env_info(args):
    with open_dict(args):
        slurm_id = os.getenv('SLURM_JOB_ID')

        if slurm_id is not None:
            args.slurm_id = slurm_id
        else:
            args.slurm_id = 'none'

        args.working_dir = os.getcwd()


def setup_basics(args):
    check_dir(args)
    check_args_and_env(args)
    update_args_with_env_info(args)
    opti_flags()
    if args.seed is not None:
        set_seed(args.seed)
        init_random(args.seed)
