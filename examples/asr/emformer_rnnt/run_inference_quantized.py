#!/usr/bin/env python3
""" Run inference on quantized model"""
import logging
import os
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter

import pandas as pd
import torch
from tqdm import tqdm

from audio.examples.asr.emformer_rnnt.speak.lightning import SpeakRNNTModule



logger = logging.getLogger(__name__)

def run_eval_subset(model, dataloader, run_id, num_samples_segment=2560, num_samples_context=640, topk=20):
    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    results = {"filename": [], "actual": [], "streaming_prediction": []}

    with torch.no_grad():
        for sample_idx, (filename, transcripts, waveform) in tqdm(enumerate(dataloader)):
            state = None
            hypotheses = None
            actual = transcripts[0]
            waveform = waveform[0]
            for idx in (range(0, len(waveform), num_samples_segment)):
                segment = torch.Tensor(waveform[idx : idx + num_samples_segment + num_samples_context])
                segment = torch.nn.functional.pad(segment, (0, num_samples_segment + num_samples_context - len(segment)))

                with torch.no_grad():
                    streaming_prediction, probs, hypotheses, state = model(segment, prev_hypo=hypotheses, prev_state=state)
                   
        
            if sample_idx % 10 == 0:
                print(f"\rPredicted: {streaming_prediction[0]}")
                print(f"Actual: {actual}")

            results["filename"].append(filename[0])
            results["actual"].append(actual)
            results["streaming_prediction"].append(streaming_prediction)

    df = pd.DataFrame(results)
    df.to_csv(f"results_streaming_{run_id}.csv")
    print(f"Saved results to results_streaming_{run_id}.csv")

def get_lightning_module(args):
    return SpeakRNNTModule(
            data_path=str(args.dataset_path),
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )

def run_eval(model, args):
    run_id = os.path.basename(args.dataset_path).split(".")[0]
    audio_data_dir =  "/data/home/ec2-user/data_cache/" + run_id
    dataloader = get_lightning_module(args).test_dataloader(csv_files = [args.dataset_path], data_dirs = [audio_data_dir], batch_size=1)
    run_eval_subset(model, dataloader, run_id)

def load_quantized_model(args):
    return torch.jit.load(args.model_path, map_location=torch.device("cpu"))

def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument("--model-path", type=pathlib.Path, help="Path to checkpoint to use for evaluation.",
                        default="/data/home/ec2-user/before_emformer_linear_quantised_20230427-023514-03.ptl")
    
    parser.add_argument("--global-stats-path", type=pathlib.Path, help="Path to JSON file containing feature means and stddevs.",
                        default="/data/home/ec2-user/audio/examples/asr/emformer_rnnt/librispeech/global_stats.json")
    parser.add_argument("--sp-model-path", type=pathlib.Path, help="Path to SentencePiece model.", 
                        default= "/data/home/ec2-user/audio/examples/asr/emformer_rnnt/spm_bpe_4096_librispeech.model")
        
    parser.add_argument("--dataset-path", type=pathlib.Path, help="Path to dataset.", required=True)
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
    model = load_quantized_model(args)
    # Print device and model
    # print("Model loaded on device: ", next(model.parameters()).device)
    run_eval(model, args)

if __name__ == "__main__":
    cli_main()

