# Read a large csv file, split it into smaller csv files and run inference_csv.py on each of them in parallel
import os
import argparse
import concurrent.futures
import subprocess

import pandas as pd

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def create_command(input_file, output_file, ckpt_path, sp_model_path, global_stats_path, device_id):
    return f"python3 audio/examples/asr/emformer_rnnt/run_inference_csv.py --test-csv-path {input_file} --output-path {output_file} --ckpt-path {ckpt_path} --sp-model-path {sp_model_path} --global-stats-path {global_stats_path} --device_id {device_id}"

def split_csv(input_file, output_dir, num_files):
    df = pd.read_csv(input_file)
    num_rows = df.shape[0]
    num_rows_per_file = int(num_rows / num_files)
    for i in range(num_files):
        start = i * num_rows_per_file
        end = start + num_rows_per_file
        if i == num_files - 1:
            end = num_rows
        # Make output dir if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df[start:end].to_csv(os.path.join(output_dir, f'{i}.csv'), index=False)
        


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to input csv file.",
        default="/data/home/ec2-user/asr_data/raw_csv/test_ai_tutor_04_10.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output csv file.",
        default="/data/home/ec2-user/asr_data/predictions_csv/test_ai_tutor_04_10_split",
    )
    parser.add_argument(
        "--num-files",
        type=int,
        help="Number of files to split the input file into.",
        default=4,
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        help="Path to checkpoint to use for evaluation.",
        default="/data/home/ec2-user/audio/examples/asr/emformer_rnnt/exp/20230824-155555/checkpoints/epoch=8-step=58275.ckpt",
    )
    parser.add_argument(
        "--global-stats-path",
        default="/data/home/ec2-user/audio/examples/asr/emformer_rnnt/librispeech/global_stats.json",
        type=str,
        help="Path to JSON file containing feature means and stddevs.",
    )
    parser.add_argument(
        "--sp-model-path",
        default="/data/home/ec2-user/audio/examples/asr/emformer_rnnt/spm_bpe_4096_librispeech.model",
        type=str,
        help="Path to SentencePiece model.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        help="GPU device id to use.",
        default=0,
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    split_csv(args.input_file, args.output_dir, args.num_files)
    # Create a command list
    command_list = [create_command(os.path.join(args.output_dir, f'{i}.csv'), os.path.join(args.output_dir, f'{i}_predictions.csv'), args.ckpt_path, args.sp_model_path, args.global_stats_path, args.device_id) for i in range(args.num_files)]

    # Run commands in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(run_command, command_list)

    # Combine the predictions from the split csv files
    df = pd.DataFrame()
    for i in range(args.num_files):
        df = pd.concat([df, pd.read_csv(os.path.join(args.output_dir, f'{i}_predictions.csv'))], ignore_index=True)
    df.to_csv(os.path.join(args.output_dir, f'predictions.csv'), index=False)