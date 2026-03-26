import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, required=True, choices=["seedtts"])
    parser.add_argument("--dataset-path", type=str, required=True)
    return parser.parse_args()


DATASET_MAPPING = {
    "seedtts": "zhaochenyang20/seed-tts-eval-mini",
}


def main():
    args = parse_args()
    print(args)


if __name__ == "__main__":
    main()
