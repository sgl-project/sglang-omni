import argparse
import os
import zipfile

from huggingface_hub import snapshot_download

DATASET_MAPPING = {
    "seedtts_tts": "frankleeeee/seedtts-testset-tts",
    "seedtts_tts_5_samples": "frankleeeee/seedtts-testset-tts-5-samples",
    "seedtts_vc": "frankleeeee/seedtts-testset-vc",
    "seedtts_vc_5_samples": "frankleeeee/seedtts-testset-vc-5-samples",
}


def download_hf_dataset(dataset_name: str, output_dir: str):
    """
    Download the dataset from Hugging Face

    Args:
        dataset_name: The name of the dataset to download
        output_dir: The directory to save the downloaded dataset

    Returns:
        The path to the downloaded dataset
    """
    dataset_path = os.path.join(output_dir, dataset_name)
    return snapshot_download(
        repo_id=DATASET_MAPPING[dataset_name],
        repo_type="dataset",
        local_dir=dataset_path,
    )


def unzip_dataset(dataset_name: str, output_dir: str, data_file_name: str):
    """
    Unzip the dataset

    Args:
        dataset_name: The name of the dataset to unzip
        output_dir: The directory to save the unzipped dataset
        data_file_name: The name of the data file to unzip
    """
    dataset_path = os.path.join(output_dir, dataset_name)
    data_file_path = os.path.join(dataset_path, data_file_name)
    zipfile.ZipFile(data_file_path, "r").extractall(dataset_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, choices=DATASET_MAPPING.keys()
    )
    parser.add_argument("--output-dir", type=str, default="./cache")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if "seedtts" in args.dataset_name:
        download_hf_dataset(args.dataset_name, args.output_dir)
        unzip_dataset(args.dataset_name, args.output_dir, "data.zip")
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")


if __name__ == "__main__":
    main()
