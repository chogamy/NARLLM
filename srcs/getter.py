import yaml

from lightning import Trainer
from transformers import AutoModel, AutoTokenizer
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model

from srcs.data.metric import METRIC
from srcs.data.datamodule import DataModule
from srcs.lightning_wrapper import LightningWrapper


def get_datamodule(args, tokenizer):
    dm = DataModule(args)
    dm.setup(args.mode, tokenizer)

    return dm


def get_model(args):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # model
    model = AutoModel.from_pretrained(args.model)

    if args.peft == "lora":
        peft_config = LoraConfig()

        model = get_peft_model(model, peft_config)

    metric = METRIC[args.data]()

    if args.mode == "fit":
        model = LightningWrapper(model, tokenizer, metric)

    if args.mode == "test":
        path = None
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
        """
        load from ckpt
        """
        pass

    return model, tokenizer


def get_trainer(args):
    with open(args.trainer_args) as f:
        trainer_args = yaml.load(f)

    trainer = Trainer(**trainer_args)

    return trainer
