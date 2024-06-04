import argparse

from lightning import seed_everything

from srcs.getter import get_model, get_datamodule, get_trainer


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

    seed_everything(42, workers=True)

    model, tokenizer = get_model(args)
    dm = get_datamodule(args, tokenizer)
    trainer = get_trainer(args)

    if args.mode == "fit":
        trainer.fit(model=model, datamodule=dm)

    if args.mode == "test":
        trainer.test(model=model, datamodule=dm)
