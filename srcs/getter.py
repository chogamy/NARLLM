import yaml

from lightning import Trainer
from transformers import AutoModel, AutoTokenizer

from srcs.data.datamodule import DataModule
from srcs.lightning_wrapper import LightningWrapper


def get_datamodule(args, tokenizer):
    dm = DataModule(args)
    dm.setup(args.mode, tokenizer)

    return dm


def get_model(args):
    if args.mode == "fit":
        pass

    if args.mode == "test":
        pass

    model = AutoModel.from_pretrained(args.model)
    model = LightningWrapper(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_trainer(args):
    with open(args.trainer_args) as f:
        trainer_args = yaml.load(f)

    trainer = Trainer(**trainer_args)

    return trainer
