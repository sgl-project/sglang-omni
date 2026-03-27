import argparse
import os
import zipfile

from huggingface_hub import snapshot_download

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_MAPPING = {
    "seedtts_tts": "frankleeeee/seedtts-testset-tts",
    "seedtts_tts_5_samples": "frankleeeee/seedtts-testset-tts-5-samples",
    "seedtts_vc": "frankleeeee/seedtts-testset-vc",
    "seedtts_vc_5_samples": "frankleeeee/seedtts-testset-vc-5-samples",
}


def download_hf_dataset(dataset_name: str, dataset_path: str):
    """
    Download the dataset from Hugging Face

    Args:
        dataset_name: The name of the dataset to download
        output_dir: The directory to save the downloaded dataset

    Returns:
        The path to the downloaded dataset
    """
    return snapshot_download(
        repo_id=DATASET_MAPPING[dataset_name],
        repo_type="dataset",
        local_dir=dataset_path,
    )


def unzip_dataset(dataset_path: str, data_file_name: str):
    """
    Unzip the dataset

    Args:
        dataset_name: The name of the dataset to unzip
        output_dir: The directory to save the unzipped dataset
        data_file_name: The name of the data file to unzip
    """
    data_file_path = os.path.join(dataset_path, data_file_name)
    with zipfile.ZipFile(data_file_path, "r") as zip_ref:
        zip_ref.extractall(dataset_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, choices=DATASET_MAPPING.keys()
    )
    parser.add_argument(
        "--output-dir", type=str, default=os.path.join(BENCHMARK_DIR, "cache")
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = os.path.join(args.output_dir, args.dataset)
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    if "seedtts" in args.dataset:
        download_hf_dataset(args.dataset, dataset_path)
        unzip_dataset(dataset_path, "data.zip")
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset}")
    print(f"Dataset prepared and saved to {dataset_path}")


if __name__ == "__main__":
    main()
