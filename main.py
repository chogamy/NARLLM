import argparse

from srcs.parser.model import parse_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True, default=None, type=str, choices=["infer", "train"]
    )

    args = parser.parse_args()

    # model and tokenizer
    model, tokenizer = parse_model(args)

    # trainer
    if args.mode == "train":
        pass

    if args.mode == "infer":
        pass
