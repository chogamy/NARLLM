import argparse

from lightning import Trainer

from srcs.parser.model import parse_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", required=True, default=None, type=str, choices=["infer", "train"]
    )
    parser.add_argument("--model", required=True, default=None, type=str)
    parser.add_argument("--trainer_args", required=True, default=None, type=str)

    args = parser.parse_args()

    # model and tokenizer
    model, tokenizer = parse_model(args)

    # trainer
    trainer = Trainer()
    if args.mode == "train":
        pass

    if args.mode == "infer":
        pass
