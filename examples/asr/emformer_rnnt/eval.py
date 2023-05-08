#!/usr/bin/env python3
"""Evaluate the lightning module by loading the checkpoint, the SentencePiece model, and the global_stats.json.

Example:
python eval.py --model-type tedlium3 --checkpoint-path ./experiments/checkpoints/epoch=119-step=254999.ckpt
    --dataset-path ./datasets/tedlium --sp-model-path ./spm_bpe_500.model
"""
import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter
import sys
import numpy as np
sys.path.append("/data/home/ec2-user/")
import torch
import torchaudio
import wandb
import pandas as pd
from speak.lightning import SpeakRNNTModule
from common import post_process_hypos
# import audio.torchaudio as torchaudio
from torchaudio.models import RNNTBeamSearch

# TODO: Whisper normalisation and WER computation
logger = logging.getLogger(__name__)

# set wandb project name
# wandb.init(project="emformer_rnnt")
GLOBAL_STATS_PATH = "/data/home/ec2-user/audio/examples/asr/emformer_rnnt/librispeech/global_stats.json"
SP_MODEL_PATH = "/data/home/ec2-user/audio/examples/asr/emformer_rnnt/spm_bpe_4096_librispeech.model"
# CKPT_DEFAULT = "/data/home/ec2-user/audio/examples/asr/emformer_rnnt/exp/20230412-164625/checkpoints/epoch=0-step=2760-v1.ckpt"
CKPT_DEFAULT = "/data/home/ec2-user/audio/examples/asr/emformer_rnnt/exp/20230427-023514/checkpoints/epoch=0-step=5215-v1.ckpt"
# Extract timestamp from checkpoint path
run_id = CKPT_DEFAULT.split("/")[-3]

DATA_DIR_DEFAULT = "/data/home/ec2-user/data_cache/test_ai_tutor_04_10"
def compute_word_level_distance(seq1, seq2):
    # remove punctuation
    seq1 = seq1.replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(";", "").replace("-", "").replace("'", "").replace('"', "")
    seq1 = seq1.replace(".", "").replace(",", "").replace("?", "").replace("!", "").replace(":", "").replace(";", "").replace("-", "").replace("'", "").replace('"', "")
    return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())


def run_eval_subset(model, dataloader, subset):
    hop_length = 160 # 160 samples = 10ms frame shift
    segment_length = 16 # 8 * 10ms = 80ms
    right_context_length = 4 # 4 * 10ms = 40ms
    # left_context_length = 32 # 32 * 10ms = 40ms # TODO: introduce left context
    # TODO: Check if the memory is used in the Librispeech model.

    num_samples_segment = segment_length * hop_length
    num_samples_context = right_context_length * hop_length
    num_samples_segment_right_context = num_samples_segment + num_samples_context

    total_edit_distance = 0
    total_edit_distance_streaming = 0
    total_length = 0
    results = {"filename": [], "actual": [], "streaming_prediction": [], "streaming_prediction_probs": []}
    # TODO: Move this to lightning module"
    # Set model to eval mode
    model.eval()
    model.model.transcriber.eval()
    # model.model.transcriber.transformer.segment_length = int(segment_length / 4)
    # model.model.transcriber.transformer.right_context_length = int(right_context_length / 4)
    # model.model.transcriber.transformer.transformer_left_context_length = 50
    # model.model.transcriber.transformer.left_context_length = int(left_context_length / 4)

    with torch.no_grad():
        for sample_idx, (filename, transcripts, waveform) in enumerate(dataloader):

            # Dump the waveform to a binary file
            # np_waveform = np.array(waveform[0])
            # np_waveform.tofile(f"eval_{sample_idx}.bin")
            # print(f"Processing sample {sample_idx}")
            decoder = RNNTBeamSearch(model.model, model.blank_idx) # Why is model.model needed? - cos it encapsulates the predictor and joiner
            actual = transcripts[0]
            # batch inference
            # predicted = model(batch)
            # predicted = ""
            streaming_prediction = ""
            # Extract frames and run streaming inference
            state = None
            hypotheses = None
            waveform = waveform[0]
            predicted_transcript = ""
            for idx in (range(0, len(waveform), num_samples_segment)):
                # print(f"Processing segment {idx} to {idx + num_samples_segment_right_context}")
                segment = torch.Tensor(waveform[idx : idx + num_samples_segment_right_context])
                segment = torch.nn.functional.pad(segment, (0, num_samples_segment_right_context - len(segment)))

                with torch.no_grad():
                    features, length = model.streaming_extract_features(segment) # features: (1, 21, 80) # Extract mel features
                    hypotheses, state = decoder.infer(features.to(model.device), length.to(model.device), 20, state=state, hypothesis=hypotheses)
                    
                    hypos = post_process_hypos(hypotheses, model.sp_model)
                    hypotheses = hypotheses
            predicted_transcript = hypos[0][0]
            # print(hypos[0][0], end="", flush=True)
            
            # streaming_probs = [h[1] for h in hypos][:30]
            # streaming_prediction = [h[0] for h in hypos][:30]
            if sample_idx % 1 == 0:
                sys.stdout.write(f"\rPredicted: {predicted_transcript}")
                sys.stdout.flush()
                print(f"\t Actual: {actual}")
            # total_edit_distance += compute_word_level_distance(actual, predicted)
            # total_edit_distance_streaming += compute_word_level_distance(actual, streaming_prediction)
            # Pick the minimum edit distance
            current_edit_distance = np.inf
            for hyp in streaming_prediction:
                current_edit_distance = min(compute_word_level_distance(actual, hyp), current_edit_distance)
                if current_edit_distance == 0:
                    break
            total_edit_distance_streaming += current_edit_distance
            results["filename"].append(filename[0])
            results["actual"].append(actual)
            # results["predicted"].append(predicted)
            results["streaming_prediction"].append(streaming_prediction)
            # results["streaming_prediction_probs"].append(streaming_probs)
            total_length += len(actual.split())            
            # if sample_idx % 100 == 0:
            #     # Save results to csv
            #     df = pd.DataFrame(results)
            #     df.to_csv(f"results_{subset}_streaming_{run_id}.csv")
            #     logger.info(f"Processed elem {sample_idx}; WER: {total_edit_distance_streaming / total_length}")
    # Save results to csv
    # df = pd.DataFrame(results)
    # df.to_csv(f"results_{subset}_streaming_{run_id}.csv")
    # logger.info(f"Final WER for {subset} set: {total_edit_distance_streaming / total_length}")


def run_eval(model, device, args):
    dataloader = model.test_dataloader(csv_files = [args.dataset_path], data_dirs = [DATA_DIR_DEFAULT], batch_size=1)
    run_eval_subset(model, dataloader, f"test_device_{device}")


def get_lightning_module(args):
    return SpeakRNNTModule.load_from_checkpoint(
            args.checkpoint_path,
            dataset_path=str(args.dataset_path),
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )



def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        # default=pathlib.Path("/data/home/ec2-user/audio/examples/asr/emformer_rnnt/exp/checkpoints/epoch=116-step=53001.ckpt"),
        default=pathlib.Path(CKPT_DEFAULT),
        help="Path to checkpoint to use for evaluation.",
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path(GLOBAL_STATS_PATH),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--dataset-path",
        default=pathlib.Path(""),
        type=pathlib.Path,
        help="Path to dataset.",
    )
    parser.add_argument(
        "--sp-model-path",
        default=pathlib.Path(SP_MODEL_PATH),
        type=pathlib.Path,
        help="Path to SentencePiece model.",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        default=True,
        help="Run using CUDA.",
    )
    parser.add_argument(
        "--device-id",
        default=0,
        help="GPU device id to use.",
        type=int
    )
    
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
    if args.use_cuda:
        model = model.to(device=f"cuda:{args.device_id}")
    run_eval(model,args.device_id, args)


if __name__ == "__main__":
    cli_main()
