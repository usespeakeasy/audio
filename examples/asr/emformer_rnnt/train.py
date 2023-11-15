#!/usr/bin/env python3
import datetime
import logging
import os
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
        every_n_train_steps=1000,
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
        strategy="ddp",
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=WandbLogger(project="emformer_rnnt"),
        accumulate_grad_batches=16,
        limit_val_batches=0.2,
        # overfit_batches=0.001,
        
    )


def get_lightning_module(args):
    # train_csv_folder = "/data/home/ec2-user/asr_data/validated_csv/train"
    # val_csv_folder = "/data/home/ec2-user/asr_data/validated_csv/val"
    # # List all 
    # train_csv_files = [f"{train_csv_folder}/{file}" for file in os.listdir(train_csv_folder)]
    val_csv_files = ["/data/home/ec2-user/asr_data/validated_csv/val/ko-KR_2023-07-24_val.csv"] # TODO: FIX nnumpy error while loading data
    
    train_csv_files = ["/data/home/ec2-user/asr_data/aggregated_csv/emformer_v1/train.csv"]
    # val_csv_files = ["/data/home/ec2-user/asr_data/aggregated_csv/emformer_v1/val.csv"]
    
    if args.checkpoint_path:
        return SpeakRNNTModule.load_from_checkpoint(
            args.checkpoint_path,
            data_path={"train": {"csv_files": train_csv_files, "data_dir": args.datacache_dir},
                       "val": {"csv_files": val_csv_files, "data_dir": args.datacache_dir}},
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )
    return SpeakRNNTModule(
            data_path={"train": {"csv_files": train_csv_files, "data_dir": args.datacache_dir},
                       "val": {"csv_files": val_csv_files, "data_dir": args.datacache_dir}},
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
        default=pathlib.Path("/data/home/ec2-user/audio_files"), 
        help="Path to audio cache directory. (Default: 'audio_files')",
        required=False,
    )
    # parser.add_argument(
    #     "--dataset-name",
    #     type=str,
    #     help="Name of dataset to train on", # TODO: Add support for aggregation of multiple datasets via command line
        
    # )
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
        default=4,
        type=int,
        help="Number of GPUs per node to use for training. (Default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=10,
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
