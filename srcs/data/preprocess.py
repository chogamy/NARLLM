def ConvAI2(batch, tokenizer, prompt):
    yours_batch = [
        "your persona: " + " ".join(persona) for persona in batch["your persona:"]
    ]
    partners_batch = [
        "partner's persona: " + " ".join(persona)
        for persona in batch["partner's persona:"]
    ]
    dialogue_batch = [
        "dialogue: " + " ".join(dialogue) for dialogue in batch["dialogue"]
    ]

    tgt_batch = batch["target"]

    inputs = [
        f"{prompt} {yours} {partners} {dialogue} length: "
        for yours, partners, dialogue, tgt in zip(
            yours_batch, partners_batch, dialogue_batch, tgt_batch
        )
    ]

    """
    input:  [bos]	[p]	        [length]
    target: [p]	    [length]	[eos]
    """
    length_batch = tokenizer(
        [f"{tokenizer.bos_token}{inp}" for inp in inputs],
        padding="max_length",
        truncation=True,
        max_length=int(tokenizer.model_max_length / 2),
    )

    length_batch["target"] = [
        len(tokenizer(tgt, padding=False, truncation=False)["input_ids"])
        for tgt in tgt_batch
    ]

    inputs = [
        f"{inp}{len(tokenizer(tgt, padding=False, truncation=False)['input_ids'])} "
        for inp, tgt in zip(inputs, tgt_batch)
    ]

    """
    input:  [bos]	[p]	        [length]	[mask]	[eos]
    target: [p]	    [length]	[eos]	    [tgt]	[eos]
    """
    target_batch = tokenizer(
        [
            f"{tokenizer.bos_token}{inp}{tokenizer.bos_token * len(tokenizer(tgt, padding=False, truncation=False)['input_ids'])}{tokenizer.eos_token}"
            for inp, tgt in zip(inputs, tgt_batch)
        ],
        padding="max_length",
        truncation=True,
        max_length=int(tokenizer.model_max_length / 2),
    )

    target_batch["target"] = tokenizer(
        [
            f"{inp}{tokenizer.eos_token}{tgt}{tokenizer.eos_token}"
            for inp, tgt in zip(inputs, tgt_batch)
        ],
        padding="max_length",
        truncation=True,
        max_length=int(tokenizer.model_max_length / 2),
    )["input_ids"]

    length_input = {f"length_{k}": v for k, v in length_batch.items()}
    target_input = {f"target_{k}": v for k, v in target_batch.items()}

    model_input = {**length_input, **target_input}

    return model_input


PREPROCESS = {"ConvAI2": ConvAI2}
