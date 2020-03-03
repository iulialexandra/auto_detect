import argparse
import logging
import os

import torch
import torch.distributed as dist

from ssd.engine.inference import do_evaluation
from ssd.config import cfg
from ssd.data.build import make_data_loader
from ssd.modeling.detector import build_detection_model
from ssd.solver.build import make_optimizer, make_lr_scheduler
from ssd.utils import dist_util, mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
from ssd.utils.misc import str2bool

import collections
import datetime
import logging
import os
import time
import torch
import torch.distributed as dist

from ssd.engine.inference import do_evaluation
from ssd.utils import dist_util
from ssd.utils.metric_logger import MetricLogger


def write_metric(eval_result, prefix, summary_writer, global_step):
    for key in eval_result:
        value = eval_result[key]
        tag = '{}/{}'.format(prefix, key)
        if isinstance(value, collections.Mapping):
            write_metric(value, tag, summary_writer, global_step)
        else:
            summary_writer.add_scalar(tag, value, global_step=global_step)


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = dist_util.get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def inc_train(cfg, model,
              data_loader,
              optimizer,
              scheduler,
              checkpointer,
              device,
              arguments,
              args):
    logger = logging.getLogger("SSD.trainer")
    logger.info("Start training ...")
    meters = MetricLogger()

    model.train()
    save_to_disk = dist_util.get_rank() == 0
    if args.use_tensorboard and save_to_disk:
        import tensorboardX

        summary_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(cfg.OUTPUT_DIR, 'tf_logs'))
    else:
        summary_writer = None

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = targets.to(device)
        loss_dict = model(images, targets=targets)
        loss = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(total_loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)
        if iteration % args.log_step == 0:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                meters.delimiter.join([
                    "iter: {iter:06d}",
                    "lr: {lr:.5f}",
                    '{meters}',
                    "eta: {eta}",
                    'mem: {mem}M',
                ]).format(
                    iter=iteration,
                    lr=optimizer.param_groups[0]['lr'],
                    meters=str(meters),
                    eta=eta_string,
                    mem=round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                )
            )
            if summary_writer:
                global_step = iteration
                summary_writer.add_scalar('losses/total_loss', losses_reduced, global_step=global_step)
                for loss_name, loss_item in loss_dict_reduced.items():
                    summary_writer.add_scalar('losses/{}'.format(loss_name), loss_item, global_step=global_step)
                summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if iteration % args.save_step == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)

        if args.eval_step > 0 and iteration % args.eval_step == 0 and not iteration == max_iter:
            eval_results = do_evaluation(cfg, model, distributed=args.distributed, iteration=iteration)
            if dist_util.get_rank() == 0 and summary_writer:
                for eval_result, dataset in zip(eval_results, cfg.DATASETS.TEST):
                    write_metric(eval_result['metrics'], 'metrics/' + dataset, summary_writer, iteration)
            model.train()  # *IMPORTANT*: change to train mode after eval.

    checkpointer.save("model_final", **arguments)
    # compute training time
    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
    return model


def train(cfg, args):
    logger = logging.getLogger('SSD.trainer')
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    optimizer = make_optimizer(cfg, model, lr)

    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    arguments = {"iteration": 0}
    save_to_disk = dist_util.get_rank() == 0
    checkpointer = CheckPointer(model, optimizer, scheduler, cfg.OUTPUT_DIR, save_to_disk, logger)
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
    train_loader = make_data_loader(cfg, is_train=True, distributed=args.distributed, max_iter=max_iter, start_iter=arguments['iteration'])

    model = inc_train(cfg, model, train_loader, optimizer, scheduler, checkpointer, device, arguments, args)
    return model


def main():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=2500, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=2500, type=int, help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    args.config_file = "configs/vgg_ssd300_voc0712.yaml"

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    model = train(cfg, args)

    if not args.skip_test:
        logger.info('Start evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(cfg, model, distributed=args.distributed)


if __name__ == '__main__':
    main()
