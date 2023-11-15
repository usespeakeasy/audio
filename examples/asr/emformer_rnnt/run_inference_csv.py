# Run inference on entire test set, save results to csv
import argparse
import pathlib
import datetime
import random
import numpy as np

import pandas as pd
import torch
from speak.lightning import SpeakRNNTModule
from audio.torchaudio.models import RNNTBeamSearch
from common import post_process_hypos

GLOBAL_STATS_PATH = "/data/home/ec2-user/audio/examples/asr/emformer_rnnt/librispeech/global_stats.json"
SP_MODEL_PATH = "/data/home/ec2-user/audio/examples/asr/emformer_rnnt/spm_bpe_4096_librispeech.model"
DATA_DIR_DEFAULT = "/data/home/ec2-user/audio_files"
    
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--ckpt-path",
        type=str,
        help="Path to checkpoint to use for evaluation.",
    )
    parser.add_argument(
        "--test-csv-path",
        type=str,
        help="Path to test csv file.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to output csv file.",
        default=None,
    )
    
    parser.add_argument(
        "--device_id",
        type=int,
        help="GPU device id to use.",
        default=0,
    )
    parser.add_argument(
        "--global-stats-path",
        default=pathlib.Path(GLOBAL_STATS_PATH),
        type=pathlib.Path,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--sp-model-path",
        default=pathlib.Path(SP_MODEL_PATH),
        type=pathlib.Path,
        help="Path to SentencePiece model.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to run inference on.",
        default=None,
    )
    parser.add_argument(
        "--start-sample",
        type=int,
        help="Sample index to start inference from.",
        default=0,
    )
    return parser.parse_args()

def get_lightning_module(args):
    return SpeakRNNTModule.load_from_checkpoint(
            args.ckpt_path,
            data_path={}, # train and val data paths are not needed for inference
            sp_model_path=str(args.sp_model_path),
            global_stats_path=str(args.global_stats_path),
        )

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    args = parse_args()
    if args.output_path is None:
        ckpt = pathlib.Path(args.ckpt_path).name
        args.output_path = f"/data/home/ec2-user/asr_data/predictions_csv/{ckpt}_predictions_test_ai_tutor_04_10_{args.num_samples}.csv"
    model = get_lightning_module(args)
    model = model.to(device=f"cuda:{args.device_id}")
    if args.num_samples is not None:
        shuffle = True
    else:
        shuffle = False
    set_seed(42)
    dataloader = model.test_dataloader(csv_files = [args.test_csv_path], data_dirs=DATA_DIR_DEFAULT, batch_size=1, shuffle=shuffle)
    hop_length = 160 # 160 samples = 10ms frame shift
    segment_length = 16 # 8 * 10ms = 80ms
    right_context_length = 4 # 4 * 10ms = 40ms
    num_samples_segment = segment_length * hop_length
    num_samples_context = right_context_length * hop_length
    num_samples_segment_right_context = num_samples_segment + num_samples_context
    results = {"filename": [], "actual": [], "streaming_prediction": []}
    
    model.eval()
    model.model.transcriber.eval()
    samples_processed = 0
    
    with torch.no_grad():
        for sample_idx, (filename, transcripts, waveform) in enumerate(dataloader):
            
            if sample_idx < args.start_sample:
                print(f"Skipping sample {sample_idx}.")
                continue  # Skip until reaching the starting sample
            
            if args.num_samples is not None:
                # Break if the number of samples processed reaches the total number of samples specified
                if args.num_samples == (args.start_sample + samples_processed):
                    print(f"Processed {args.num_samples} samples.")
                    break
            decoder = RNNTBeamSearch(model.model, model.blank_idx) # Why is model.model needed? - cos it encapsulates the predictor and joiner
            try:
                actual = transcripts[0]
                if actual.lower() in ("thank you", "thank you.", "thank you for watching.", "thank you for watching"):
                    continue
            except:
                print(f"Error getting transcript for {filename}")
                continue
            
            streaming_prediction = ""
            state = None
            hypotheses = None
            waveform = waveform[0]
            for idx in (range(0, len(waveform), num_samples_segment)):
                segment = torch.Tensor(waveform[idx : idx + num_samples_segment_right_context])
                segment = torch.nn.functional.pad(segment, (0, num_samples_segment_right_context - len(segment)))

                with torch.no_grad():
                    features, length = model.streaming_extract_features(segment) # features: (1, 21, 80) # Extract mel features
                    hypotheses, state = decoder.infer(features.to(model.device), length.to(model.device), 20, state=state, hypothesis=hypotheses)
            hypos = post_process_hypos(hypotheses, model.sp_model)
            streaming_predictions = [h[0] for h in hypos][:10]
            if sample_idx % 500 == 0:
                print(f"Predicted: {hypos[0][0]}")
                print(f"Actual: {actual}")

            results["filename"].append(filename[0])
            results["actual"].append(actual)
            results["streaming_prediction"].append(streaming_predictions)
            
            samples_processed += 1
            print(f"Processed {samples_processed} samples.")
    df = pd.DataFrame(results)
    print(f"Saving results to {args.output_path}.")
    df.to_csv(args.output_path, index=False)