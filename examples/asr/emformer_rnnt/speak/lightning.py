
from functools import partial
from typing import List
import torchmetrics


import sentencepiece as spm
import torch

import torchaudio
from common import (
    Batch,
    batch_by_token_count,
    FunctionalModule,
    GlobalStatsNormalization,
    piecewise_linear_log,
    post_process_hypos,
    spectrogram_transform,
    WarmupLR,
)
from pytorch_lightning import LightningModule
from torchaudio.models import emformer_rnnt_base, RNNTBeamSearch

from speak_datasets.data_pytorch import AudioDataset
MAX_TOKENS_PACKED = 512

class CustomDataset(torch.utils.data.Dataset):
    r"""Sort samples by target length and batch to max token count."""

    def __init__(self, base_dataset, max_token_limit, idx_target_lengths):
        super().__init__()
        self.base_dataset = base_dataset
        print("base_dataset: ", base_dataset)

        print("Sorting samples by token count...")
        idx_target_lengths = sorted(idx_target_lengths, key=lambda x: x[1], reverse=True)

        # assert max_token_limit >= idx_target_lengths[0][1], f"max_token_limit: {max_token_limit}, idx_target_lengths[0][1]: {idx_target_lengths[0][1]}"
        print("Batching samples by token count...")

        # Filter out samples that are too long
        idx_target_lengths = [x for x in idx_target_lengths if x[1] <= max_token_limit]

        self.batches = batch_by_token_count(idx_target_lengths, max_token_limit)

    def __getitem__(self, idx):
        return [self.base_dataset[subidx] for subidx in self.batches[idx]]

    def __len__(self):
        return len(self.batches)


class SpeakRNNTModule(LightningModule):
    def __init__(
        self,
        *,
        data_path: str,
        sp_model_path: str,
        global_stats_path: str,
    ):
        super().__init__()
        self.data_path = data_path
        # self.model = torch.compile(emformer_rnnt_base(num_symbols=4097))
        self.model = emformer_rnnt_base(num_symbols=4097)
        # Load weights from pretrained model
        # https://download.pytorch.org/torchaudio/models/emformer_rnnt_base_librispeech.pt
        # download the pretrained model from TorchAudio
        self.model.load_state_dict(torch.load("/data/home/ec2-user/audio/examples/asr/emformer_rnnt/emformer_rnnt_base_librispeech.pt"))
        # Use the pretrained model from Li
        self.loss = torchaudio.transforms.RNNTLoss(reduction="sum", clamp=1.0)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4, betas=(0.9, 0.999), eps=1e-8)
        self.warmup_lr_scheduler = WarmupLR(self.optimizer, 10000)

        self.train_data_pipeline = torch.nn.Sequential(
            FunctionalModule(piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
            torchaudio.transforms.FrequencyMasking(27),
            torchaudio.transforms.FrequencyMasking(27),
            torchaudio.transforms.TimeMasking(100, p=0.2),
            torchaudio.transforms.TimeMasking(100, p=0.2),
            FunctionalModule(partial(torch.nn.functional.pad, pad=(0, 4))),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
        )
        # TODO: Why is this padding needed?
        self.valid_data_pipeline = torch.nn.Sequential(
            FunctionalModule(piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
            FunctionalModule(partial(torch.nn.functional.pad, pad=(0, 4))),
            FunctionalModule(partial(torch.transpose, dim0=1, dim1=2)),
        )

        self.test_data_pipeline = torch.nn.Sequential(
            FunctionalModule(piecewise_linear_log),
            GlobalStatsNormalization(global_stats_path),
        )

        self.sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        self.blank_idx = self.sp_model.get_piece_size()
        self.save_hyperparameters()
        # Add WER metrics
        # self.wer = torchmetrics.text.WordErrorRate(
        #     concatenate_texts=True, compute_on_step=False, dist_sync_on_step=False
        # )

    def _extract_labels(self, samples: List):
        targets = [self.sp_model.encode(sample["text"].lower()) for sample in samples]
        lengths = torch.tensor([len(elem) for elem in targets]).to(dtype=torch.int32)
        targets = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(elem) for elem in targets],
            batch_first=True,
            padding_value=1.0,
        ).to(dtype=torch.int32)
        return targets, lengths

    def _train_extract_features(self, samples: List):
        mel_features = [spectrogram_transform(torch.Tensor(sample["audio"]).squeeze()).transpose(1, 0) for sample in samples]
        # Pad all features to the same length- predetermining the length of the longest sample     
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        features = self.train_data_pipeline(features)
        lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
        return features, lengths

    def _valid_extract_features(self, samples: List):
        mel_features = [spectrogram_transform(torch.Tensor(sample["audio"]).squeeze()).transpose(1, 0) for sample in samples]
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        features = self.valid_data_pipeline(features)
        lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
        return features, lengths
    
    def streaming_extract_features(self, samples: torch.Tensor):
        mel_features = [spectrogram_transform(samples).transpose(1, 0)] # Needs fixing for larger batch sizes
        features = torch.nn.utils.rnn.pad_sequence(mel_features, batch_first=True)
        
        features = self.test_data_pipeline(features)
        lengths = torch.tensor([elem.shape[0] for elem in mel_features], dtype=torch.int32)
        return features, lengths
    
    def _train_collate_fn(self, samples: List):
        samples = [sample for sample in samples if sample is not None]
        features, feature_lengths = self._train_extract_features(samples)
        targets, target_lengths = self._extract_labels(samples)
        return Batch(features, feature_lengths, targets, target_lengths)

    def _valid_collate_fn(self, samples: List):
        samples = [sample for sample in samples if sample is not None]
        features, feature_lengths = self._valid_extract_features(samples)
        targets, target_lengths = self._extract_labels(samples)
        return Batch(features, feature_lengths, targets, target_lengths)

    def _test_collate_fn(self, samples: List):
        samples = [sample for sample in samples if sample is not None]
        return [sample["filename"] for sample in samples], [sample["text"] for sample in samples], [sample["audio"] for sample in samples]

    def _step(self, batch, batch_idx, step_type):
        prepended_targets = batch.targets.new_empty([batch.targets.size(0), batch.targets.size(1) + 1])
        prepended_targets[:, 1:] = batch.targets
        prepended_targets[:, 0] = self.blank_idx
        prepended_target_lengths = batch.target_lengths + 1
        output, src_lengths, _, _ = self.model(
            batch.features,
            batch.feature_lengths,
            prepended_targets,
            prepended_target_lengths,
        )
        loss = self.loss(output, batch.targets, src_lengths, batch.target_lengths)
        self.log(f"Losses/{step_type}_loss", loss,  on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if step_type == "train":
            return loss
        
        # Compute WER metric
        # hypos = self.model(batch)
        # self.wer.add(hypos, batch.targets)
        # self.log(f"Metrics/{step_type}_wer", self.wer.compute(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Log some sample predictions
        return loss

    def configure_optimizers(self):
        return (
            [self.optimizer],
            [
                {"scheduler": self.warmup_lr_scheduler, "interval": "step"},
            ],
        )

    def forward(self, batch: Batch):
        decoder = RNNTBeamSearch(self.model, self.blank_idx)
        hypotheses = decoder(batch.features.to(self.device), batch.feature_lengths.to(self.device), 20)
        return post_process_hypos(hypotheses, self.sp_model)[0][0]
    

    def training_step(self, batch: Batch, batch_idx):
        # Print the max length of the batch
        # print("Max length of batch: ", batch.feature_lengths.max(), batch.target_lengths.max())
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch_tuple, batch_idx):
        return self._step(batch_tuple[0], batch_idx, "test")

    def train_dataloader(self):
        print("Loading training dataset")
        csv_files = ["/data/home/ec2-user/raw_data/csv/clean/train_2021_clean.csv", "/data/home/ec2-user/raw_data/train_ai_tutor_cleaned_04_26.csv"]
        data_dir = ["/data/home/ec2-user/data_cache/train_2021", "/data/home/ec2-user/data_cache/train_ai_tutor_04_10"]
        target_lengths, dataset = self.construct_dataset(csv_files, data_dir)
        dataset = CustomDataset(dataset, MAX_TOKENS_PACKED, target_lengths)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            collate_fn=self._train_collate_fn,
            num_workers=16,
            shuffle=True,
            pin_memory=True,
        )
        return dataloader

    def construct_dataset(self, csv_files, data_dir):
        datasets = []
        target_lengths = []
        for csv_file, data_dir in zip(csv_files, data_dir):
            ds = AudioDataset(data_csv=csv_file, data_dir=data_dir)
            datasets.append(ds)
            transcripts = ds.data_df["text"].to_list()
            target_lengths.extend([len(self.sp_model.encode_as_ids(t)) for t in transcripts])
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = torch.utils.data.ConcatDataset(datasets)
        return [(idx, t) for idx, t in enumerate(target_lengths)], dataset

    def val_dataloader(self):
        print("Loading validation dataset")
        csv_files = ["/data/home/ec2-user/raw_data/csv/test_ai_tutor_04_10.csv"]
        data_dir = ["/data/home/ec2-user/data_cache/test_ai_tutor_04_10"]
        target_lengths, dataset = self.construct_dataset(csv_files, data_dir)
        print("Done loading validation dataset")
        dataset = CustomDataset(dataset, MAX_TOKENS_PACKED, target_lengths)
        print("Done loading custom validation dataset")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,
            collate_fn=self._valid_collate_fn,
            num_workers=16,

        )
        return dataloader

    def test_dataloader(self, csv_files: List[str], data_dirs: List[str], batch_size=1):
        """Test dataloader. Returns a list of texts and a list of audio files."""
        # TODO: Packed sequence
        print("Loading test dataset")
        _, dataset = self.construct_dataset(csv_files, data_dirs)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=self._test_collate_fn, num_workers=16, shuffle=False)

