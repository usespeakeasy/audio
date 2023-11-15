import argparse
from hashlib import md5
import os
import sys

sys.path.append("/data/home/ec2-user/audio/examples/asr/emformer_rnnt")
sys.path.append("/data/home/ec2-user/")


import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sentencepiece as sp
import torch
from fairseq.data import Dictionary
from speak.lightning import SpeakRNNTModule
from torch.quantization import (
    default_dynamic_qconfig,
    float_qparams_weight_only_qconfig,
    quantize_dynamic,
)
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchaudio.transforms import MelSpectrogram

from audio.torchaudio.models import Hypothesis, RNNTBeamSearch




def get_hypo_tokens(hypo: Hypothesis) -> List[int]:
    return hypo[0]


def get_hypo_score(hypo: Hypothesis) -> float:
    return hypo[3]


def to_string(
    input: List[int],
    tgt_dict: List[str],
    bos_idx: int = 0,
    eos_idx: int = 2,
    separator: str = "",
) -> str:
    # torchscript dislikes sets
    extra_symbols_to_ignore: Dict[int, int] = {}
    extra_symbols_to_ignore[eos_idx] = 1
    extra_symbols_to_ignore[bos_idx] = 1

    # it also dislikes comprehensions with conditionals
    filtered_idx: List[int] = []
    for idx in input:
        if idx not in extra_symbols_to_ignore:
            filtered_idx.append(idx)

    return separator.join([tgt_dict[idx] for idx in filtered_idx]).replace(
        "\u2581", " "
    )


def post_process_hypos(
    hypos: List[Hypothesis],
    tgt_dict: List[str],
) -> Tuple[List[str], List[float]]:
    post_process_remove_list = [
        3,  # unk
        2,  # eos
        1,  # pad
    ]
    hypos_str: List[str] = []
    for h in hypos:
        filtered_tokens: List[int] = []
        for token_index in get_hypo_tokens(h)[1:]:
            if token_index not in post_process_remove_list:
                filtered_tokens.append(token_index)
        string = to_string(filtered_tokens, tgt_dict)
        hypos_str.append(string)

    hypos_score = [math.exp(get_hypo_score(h)) for h in hypos]
    return hypos_str, hypos_score


def _piecewise_linear_log(x):
    x[x > math.e] = torch.log(x[x > math.e])
    x[x <= math.e] = x[x <= math.e] / math.e
    return x


# Export fairseq dictionary from sentencepiece model
def export_fairseq_dictionary(output_path_prefix: Path, spm_path: str):
    UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
    BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
    EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
    PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1
    spm = sp.SentencePieceProcessor()
    spm.Load(spm_path)
    vocab = {i: spm.IdToPiece(i) for i in range(spm.GetPieceSize())}
    assert (
        vocab.get(UNK_TOKEN_ID) == UNK_TOKEN
        and vocab.get(PAD_TOKEN_ID) == PAD_TOKEN
        and vocab.get(BOS_TOKEN_ID) == BOS_TOKEN
        and vocab.get(EOS_TOKEN_ID) == EOS_TOKEN
    )
    vocab = {
        i: s
        for i, s in vocab.items()
        if s not in {UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN}
    }
    with open(output_path_prefix.as_posix(), "w") as f_out:
        for _, s in sorted(vocab.items(), key=lambda x: x[0]):
            f_out.write(f"{s} 1\n")
    return output_path_prefix.as_posix()


def get_lightning_module(
    checkpoint_path: str, global_stats_path: str, sp_model_path: str
):
    return SpeakRNNTModule.load_from_checkpoint(
        Path(checkpoint_path),
        map_location="cpu",
        data_path=Path(""),
        sp_model_path=sp_model_path,
        global_stats_path=Path(global_stats_path),
        device="cpu",
    )


# Print checksum md5 for the file
def get_checksum(filename):
    with open(filename, "rb") as f:
        readable_hash = md5(f.read()).hexdigest()  # Use hashlib to create md5 hash
    return readable_hash


class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        tgt_dict: List[str],
        global_stats_path: str,
        checkpoint_path: str,
        sp_model_path: str,
    ):
        super().__init__()
        self.transform = MelSpectrogram(
            sample_rate=16000, n_fft=400, n_mels=80, hop_length=160
        )
        model = get_lightning_module(checkpoint_path, global_stats_path, sp_model_path)
        model.eval()
        self.decoder = RNNTBeamSearch(model.model, model.blank_idx)

        self.tgt_dict = tgt_dict

        with open(global_stats_path) as f:
            blob = json.loads(f.read())

        self.mean = torch.tensor(blob["mean"])
        self.invstddev = torch.tensor(blob["invstddev"])

        self.decibel = 2 * 20 * math.log10(32767)
        self.gain = pow(10, 0.05 * self.decibel)

    def forward(
        self,
        input: torch.Tensor,
        prev_state: Optional[List[List[torch.Tensor]]],
        prev_hypo: Optional[List[Hypothesis]],
    ) -> Tuple[
        List[str],
        List[float],
        Optional[List[List[torch.Tensor]]],
        Optional[List[Hypothesis]],
    ]:
        spectrogram = self.transform(input).transpose(1, 0)
        features = _piecewise_linear_log(spectrogram * self.gain).unsqueeze(0)[:, :-1]
        features = (features - self.mean) * self.invstddev
        length = torch.tensor([features.shape[1]])

        if prev_hypo is None:
            if prev_state is not None:
                hypotheses, state = self.decoder.infer(
                    features, length, beam_width=20, state=prev_state
                )
            else:
                hypotheses, state = self.decoder.infer(features, length, beam_width=20)
        else:
            hypotheses, state = self.decoder.infer(
                features, length, beam_width=20, state=prev_state, hypothesis=prev_hypo
            )

        transcript, probs = post_process_hypos(hypotheses, self.tgt_dict)
        # Remove any hypothesis that has a nan probability
        non_nan_indices: List[int] = []
        for i, p in enumerate(probs):
            if not torch.tensor(p).isnan():
                non_nan_indices.append(i)
        hypotheses = [hypotheses[i] for i in non_nan_indices]
        probs = [probs[i] for i in non_nan_indices]
        transcript = [transcript[i] for i in non_nan_indices]

        return transcript, probs, state, hypotheses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument(
        "--global-stats-path",
        type=str,
        required=False,
        default="/data/home/ec2-user/audio/examples/asr/emformer_rnnt/librispeech/global_stats.json",
    )
    parser.add_argument(
        "--sp-model-path",
        type=str,
        required=False,
        default="/data/home/ec2-user/audio/examples/asr/emformer_rnnt/spm_bpe_4096_librispeech.model",
    )
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--quantize", action="store_true", default=False, required=False
    )  # TODO: Fix the boolean flag
    args = parser.parse_args()

    # Construct wrapper model
    sp_model_txt = Path("spm_bpe_4096.txt")
    tgt_dict_file = export_fairseq_dictionary(sp_model_txt, args.sp_model_path)
    tgt_dict = Dictionary.load(tgt_dict_file)
    model = ModelWrapper(
        tgt_dict.symbols,
        global_stats_path=args.global_stats_path,
        sp_model_path=args.sp_model_path,
        checkpoint_path=args.checkpoint_path,
    ).eval()

    # Run inference on CPU, set cuda visible devices to -1
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Move model to CPU
    model = model.cpu()
    # save model before quantization
    before_quantized = torch.jit.script(model)
    before_quantized.save("before_" + args.output_path)
    model.decoder.model.predictor.qconfig = None

    # Quantize the model
    if args.quantize:
        quantize_dynamic(
            model.decoder.model.transcriber,
            {torch.nn.Linear},
            dtype=torch.qint8,
            inplace=True,
        )
        print(model)

    # Script the model and optimize for mobile
    if args.quantize:
        scripted_quantised_model = torch.jit.script(model)
        scripted_quantised_model.eval()
        optimized_model = optimize_for_mobile(scripted_quantised_model)

    else:
        scripted_model = torch.jit.script(model)
        scripted_model.eval()
        optimized_model = optimize_for_mobile(scripted_model)

    # Save the model
    optimized_model.save("quantised_scripted_" + args.output_path)
    print("Saved model to: ", "quantised_scripted_" + args.output_path)
    print(
        "Size of model in MB: ",
        os.path.getsize("quantised_scripted_" + args.output_path) / 1e6,
    )

    # Save for lite interpreter
    optimized_model._save_for_lite_interpreter(args.output_path)
    print("Saved model for lite interpreter to: ", args.output_path)
    print("Size of model in MB: ", os.path.getsize(args.output_path) / 1e6)

    checksum = get_checksum(args.output_path)
    print("Checksum: ", checksum)


if __name__ == "__main__":
    main()
