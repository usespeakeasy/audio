#!/usr/bin/env python3
""" Run inference on scripted model."""
import logging
import pathlib
from argparse import ArgumentParser, RawTextHelpFormatter
import time
import torch
import torchaudio

logger = logging.getLogger(__name__)
# torch._C._set_graph_executor_optimize(False)
from torch.jit.mobile import _load_for_lite_interpreter



def infer(model, wav_file, num_samples_segment=2560, num_samples_context=640, topk=5):
    with torch.no_grad():
        waveform, samplerate = torchaudio.load(wav_file)
        assert samplerate == 16000
        waveform = waveform[0]
        state = None
        hypotheses = None

        for idx in range(0, len(waveform), num_samples_segment):
            segment = torch.Tensor(
                waveform[idx : idx + num_samples_segment + num_samples_context]
            )
            segment = torch.nn.functional.pad(
                segment, (0, num_samples_segment + num_samples_context - len(segment))
            )
            start = time.time()
            pred, prob, state, hypotheses = model(segment, state, hypotheses)
            stop = time.time()
            print(f"Time taken: {stop - start}")
            print("Probs: ", prob)
            print(f"Predicted: {pred}")


def load_quantized_model(args):
    return _load_for_lite_interpreter(args.model_path.as_posix(), map_location=torch.device("cpu"))


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "--model-path",
        type=pathlib.Path,
        help="Path to checkpoint to use for evaluation.",
        default="/data/home/ec2-user/vanilla_emformer_linear_quantised_test.ptl",
    )

    parser.add_argument(
        "--wavefile-path", type=pathlib.Path, help="Path to wav file.", required=True
    )
    return parser.parse_args()


def cli_main():
    args = parse_args()
    model = load_quantized_model(args)
    infer(model, args.wavefile_path)


if __name__ == "__main__":
    cli_main()
