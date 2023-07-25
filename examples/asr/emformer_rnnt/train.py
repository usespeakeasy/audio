#!/usr/bin/env python3
import datetime
import logging
import pathlib
from argparse import ArgumentParser
from speak.lightning import SpeakRNNTModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def get_trainer(args):
    checkpoint_dir = args.exp_dir / "checkpoints"
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/val_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
        verbose=True,
    )
    train_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        monitor="Losses/train_loss",
        mode="min",
        save_top_k=5,
        save_weights_only=False,
        verbose=True,
    )
    callbacks = [
        checkpoint,
        train_checkpoint,
    ]
    return Trainer(
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs, 
        num_nodes=args.num_nodes,
        devices=args.gpus,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_false",
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=WandbLogger(project="emformer_rnnt"),
        accumulate_grad_batches=16 * int(8/args.gpus),
        
    )


def get_lightning_module(args):
    if args.checkpoint_path:
        return SpeakRNNTModule.load_from_checkpoint(
            args.checkpoint_path,
            data_path={"train": {"csv_files": [f"/data/home/ec2-user/raw_data/validated/{args.dataset_name}_train.csv"], "data_dir": [f"{args.datacache_dir}/{args.dataset_name}"]},
                       "val": {"csv_files": [f"/data/home/ec2-user/raw_data/validated/{args.dataset_name}_val.csv"], "data_dir": [f"{args.datacache_dir}/{args.dataset_name}"]}},
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    return SpeakRNNTModule(
            data_path={"train": {"csv_files": [f"/data/home/ec2-user/raw_data/validated/{args.dataset_name}_train.csv"], "data_dir": [f"{args.datacache_dir}/{args.dataset_name}"]},
                       "val": {"csv_files": [f"/data/home/ec2-user/raw_data/validated/{args.dataset_name}_val.csv"], "data_dir": [f"{args.datacache_dir}/{args.dataset_name}"]}},
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path("librispeech/global_stats.json"),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--datacache-dir",
        type=pathlib.Path,
        default=pathlib.Path("/data/home/ec2-user/data_cache"), 
        help="Path to audio cache directory. (Default: 'data_cache')",
        required=False,
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Name of dataset to train on", # TODO: Add support for aggregation of multiple datasets
        
    )
    parser.add_argument(
        "--sp-model-path",
        default=pathlib.Path("spm_bpe_4096_librispeech.model"),
        type=pathlib.Path,
        help="Path to SentencePiece model.",
    )
    parser.add_argument(
        "--exp-dir",
        default=pathlib.Path("./exp/" + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))),
        type=pathlib.Path,
        help="Directory to save checkpoints and logs to. (Default: './exp')",
    )
    parser.add_argument(
        "--num-nodes",
        default=1,
        type=int,
        help="Number of nodes to use for training. (Default: 1)",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=4,
        type=int,
        help="Number of epochs to train for. (Default: 120)",
    )
    parser.add_argument(
        "--gradient-clip-val", default=10.0, type=float, help="Value to clip gradient values to. (Default: 10.0)"
    )
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    
    parser.add_argument("--checkpoint-path", type=pathlib.Path, help="Path to checkpoint to continue training from.")
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    model = get_lightning_module(args)
    trainer = get_trainer(args)
    trainer.fit(model)


if __name__ == "__main__":
    cli_main()
