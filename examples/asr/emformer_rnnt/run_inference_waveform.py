#!/usr/bin/env python3
import logging
import os
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter
import time


import torch
import torchaudio

from common import post_process_hypos
from speak.lightning import SpeakRNNTModule
from opensource.audio.torchaudio.models import RNNTBeamSearch

logger = logging.getLogger(__name__)

def run_eval_subset(model, wavefile, num_samples_segment=2560, num_samples_context=640, topk=5):
    model.eval()
    model.model.transcriber.eval()
    model = model.cpu()
    model.model = model.model.cpu()
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")


    decoder = RNNTBeamSearch(model.model, model.blank_idx)
    waveform, samplerate = torchaudio.load(wavefile)
    assert samplerate == 16000
    waveform = waveform[0]
    with torch.no_grad():
        state = None
        hypotheses = None
        for idx in (range(0, len(waveform), num_samples_segment)):
            segment = torch.Tensor(waveform[idx : idx + num_samples_segment + num_samples_context])
            segment = torch.nn.functional.pad(segment, (0, num_samples_segment + num_samples_context - len(segment)))
            # normalize this segment
            segment_mean = segment.mean()
            segment_std = segment.std()
            segment = (segment - segment_mean) / segment_std


            features, length = model.streaming_extract_features(segment)
            start = time.time()
            hypotheses, state = decoder.infer(features.to(model.device), length.to(model.device), topk, state=state, hypothesis=hypotheses)
            hypos = post_process_hypos(hypotheses, model.sp_model)     
            stop = time.time()
            print(f"Time taken: {stop - start}")       
            print(f"\rPredicted: {hypos[0][0]}")


def run_eval(model, args):
    run_eval_subset(model, args.wavefile_path)

def get_lightning_module(args):
    return SpeakRNNTModule.load_from_checkpoint(
            args.checkpoint_path,
            data_path="",
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )

def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--checkpoint-path", type=pathlib.Path, help="Path to checkpoint to use for evaluation.",
                        default="/data/home/ec2-user/audio/examples/asr/emformer_rnnt/exp/20230427-023514/checkpoints/epoch=0-step=5215-v1.ckpt")
    parser.add_argument("--global-stats-path", type=pathlib.Path, help="Path to JSON file containing feature means and stddevs.",
                        default="/data/home/ec2-user/audio/examples/asr/emformer_rnnt/librispeech/global_stats.json")
    parser.add_argument("--sp-model-path", type=pathlib.Path, help="Path to SentencePiece model.", 
                        default= "/data/home/ec2-user/audio/examples/asr/emformer_rnnt/spm_bpe_4096_librispeech.model")
        
    parser.add_argument("--wavefile-path", type=pathlib.Path, help="Path to wav file.", required=True)
    parser.add_argument("--use-cuda", action="store_true", default=False, help="Run using CUDA.")
    parser.add_argument("--device-id", default=0, help="GPU device id to use.", type=int)
    parser.add_argument("--debug", action="store_true", help="whether to use debug level for logging")
    return parser.parse_args()

def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")

def cli_main():
    args = parse_args()
    init_logger(args.debug)
    model = get_lightning_module(args)
    device = torch.device("cpu")
    model = model.to(device)
    run_eval(model, args)

if __name__ == "__main__":
    cli_main()

