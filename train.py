# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import numpy as np

from datetime import timedelta

import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from net import TiDE
from utils.data_utils import get_loader
from utils.basic_utils import count_parameters, set_seed, save_model
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule

logger = logging.getLogger(__name__)


# notice simple_metric and LossMeter are not implemented yet
def simple_metric(preds, labels):
    # TODO: implement simple_metric
    return np.mean(np.abs(preds - labels))


class LossMeter(object):
    # TODO: implement LossMeter
    pass


def setup(args):
    # Prepare model
    # parser yaml
    import yaml
    with open(args.model_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = config[args.model_name]

    model = TiDE(config)

    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


def valid(args, model, writer, test_loader, global_step):
    eval_losses = LossMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.MSELoss()
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(all_preds[0],
                                     preds.detach().cpu().numpy(),
                                     axis=0)
            all_label[0] = np.append(all_label[0],
                                     y.detach().cpu().numpy(),
                                     axis=0)
        epoch_iterator.set_description("Validating... (loss=%2.5f)" %
                                       eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    metric = simple_metric(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid metric: %2.5f" % metric)

    writer.add_scalar("test/metric",
                      scalar_value=metric,
                      global_step=global_step)
    return metric


def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.adam(model.parameters(),
                                 lr=args.learning_rate,
                                 momentum=0.9,
                                 weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer,
                                         warmup_steps=args.warmup_steps,
                                         t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer,
                                         warmup_steps=args.warmup_steps,
                                         t_total=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps *
        (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(
        args)  # Added here for reproducibility (even between python 2 and 3)

    losses = LossMeter()
    global_step, best_metric = 0, 0
    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" %
                    (global_step, t_total, losses.val))
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss",
                                      scalar_value=losses.val,
                                      global_step=global_step)
                    writer.add_scalar("train/lr",
                                      scalar_value=scheduler.get_lr()[0],
                                      global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [
                        -1, 0
                ]:
                    metric = valid(args, model, writer, test_loader,
                                   global_step)
                    if best_metric < metric:
                        save_model(args, model, logger)
                        best_metric = metric
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best metric: \t%f" % best_metric)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name",
                        required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset",
                        choices=["cifar10", "cifar100"],
                        default="cifar10",
                        help="Which downstream task.")
    parser.add_argument("--model_config",
                        default="config/model.yaml",
                        help="Model config.")
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="The output directory where checkpoints will be written.")

    parser.add_argument("--train_batch_size",
                        default=512,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument(
        "--eval_every",
        default=100,
        type=int,
        help="Run prediction on validation set every so many steps."
        "Will always run one evaluation at the end of training.")

    parser.add_argument("--learning_rate",
                        default=3e-2,
                        type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay",
                        default=0,
                        type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps",
                        default=10000,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type",
                        choices=["cosine", "linear"],
                        default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument(
        "--warmup_steps",
        default=500,
        type=int,
        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass."
    )
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s" %
        (args.local_rank, args.device, args.n_gpu,
         bool(args.local_rank != -1)))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    # Training
    train(args, model)


if __name__ == "__main__":
    main()