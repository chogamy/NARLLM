from transformers import AutoModel, AutoTokenizer
from ..lightning_wrapper import LightningWrapper


def parse_model(args):
    model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = LightningWrapper(model)

    return model, tokenizer
