def ConvAI2(batch, tokenizer, prompt, split):
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
        f"{prompt}\n{yours}\n{partners}\n{dialogue}\nlength: "
        for yours, partners, dialogue in zip(
            yours_batch, partners_batch, dialogue_batch
        )
    ]

    """
    input:  [bos]	[p]
    target:         [l]    
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

    """
    update input
    input: [bos] [p] [l]
    """
    inputs = [
        f"{inp}{len(tokenizer(tgt, padding=False, truncation=False)['input_ids'])} "
        for inp, tgt in zip(inputs, tgt_batch)
    ]

    
    if split == "train":
        """
        input:  [bos]	[p] [l]     [mask]	
        target: [p]     [l] [sep]   [tgt]	
        """
        target_batch = tokenizer(
            [
                f"{tokenizer.bos_token}{inp}{tokenizer.bos_token * len(tokenizer(tgt, padding=False, truncation=False)['input_ids'])}"
                for inp, tgt in zip(inputs, tgt_batch)
            ],
            padding="max_length",
            truncation=True,
            max_length=int(tokenizer.model_max_length / 2),
        )

        target_batch["target"] = tokenizer(
            [
                f"{inp}{tokenizer.sep_token}{tgt}"
                for inp, tgt in zip(inputs, tgt_batch)
            ],
            padding="max_length",
            truncation=True,
            max_length=int(tokenizer.model_max_length / 2),
        )["input_ids"]
    else:
        """
        input:  [bos]	[p] [l]     [mask]	
        target:                     [tgt]	
        """
        
        target_batch = tokenizer(
            [
                f"{tokenizer.bos_token}{inp}{tokenizer.bos_token * len(tokenizer(tgt, padding=False, truncation=False)['input_ids'])}"
                for inp, tgt in zip(inputs, tgt_batch)
            ],
            padding="max_length",
            truncation=True,
            max_length=int(tokenizer.model_max_length / 2),
        )

        target_batch["target"] = tokenizer(
            [f"{tgt}" for tgt in tgt_batch],
            padding="max_length",
            truncation=True,
            max_length=int(tokenizer.model_max_length / 2),
        )["input_ids"]

    length_input = {f"length_{k}": v for k, v in length_batch.items()}
    target_input = {f"target_{k}": v for k, v in target_batch.items()}

    model_input = {**length_input, **target_input}

    return model_input


PREPROCESS = {"ConvAI2": ConvAI2}
