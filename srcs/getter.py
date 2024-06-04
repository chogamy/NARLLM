from transformers import AutoModel, AutoTokenizer

from srcs.data.datamodule import DataModule
from srcs.lightning_wrapper import LightningWrapper


def get_datamodule(args, tokenizer):
    dm = DataModule(args)
    dm.setup(args.mode, tokenizer)

    return dm


def get_model(args):
    model = AutoModel.from_pretrained(args.model)
    model = LightningWrapper(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
