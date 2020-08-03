import os
import pandas as pd
import argparse


def filter_topic_names(unfiltered_topic):
    return unfiltered_topic.rsplit("-", maxsplit=1)[0].replace("-", " ")


def read_files(path: str) -> pd.DataFrame:
    frames = []
    for filename in os.listdir(path):
        if filename.upper().endswith(".TSV"):
            filepath = os.path.join(path, filename)
            frame = pd.read_csv(filepath, sep="\t")
            frames.append(frame)
    df = pd.concat(frames, sort=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input Directory",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output Directory"
    )
    args = parser.parse_args()
    df = read_files(args.input_dir)
    df.to_csv(os.path.join(args.output_dir, "complete.csv"), sep="\t")


if __name__ == "__main__":
    main()
