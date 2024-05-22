import argparse

from lightning import Trainer

from srcs.lightning_wrapper import get_model
from srcs.data.datamodule import get_datamodule


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        default=None,
        type=str,
        choices=["fit", "test", "predict"],
    )
    parser.add_argument("--model", required=True, default=None, type=str)
    parser.add_argument("--trainer_args", default="args/trainer/basic.yaml", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--data", required=True, default=None, type=str)
    args = parser.parse_args()

    dm = get_datamodule(args)
    raise NotImplementedError("yet")
    model, tokenizer = get_model(args)

    # maybe need get_trainer
    trainer = Trainer()

    if args.mode == "fit":
        trainer.fit()
        pass

    if args.mode == "test":
        pass
