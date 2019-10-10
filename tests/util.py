# /util.py
#
# Some utils for making vectors

import numpy as np

def make_vec(model, tokens):
    return np.mean([
        model[t] for t in tokens
    ], axis=0)


def make_tokenizer(tokenizer):
    if not tokenizer or tokenizer is "none":
        return None

    import pytorch_transformers

    if tokenizer.startswith("bert"):
        return pytorch_transformers.BertTokenizer.from_pretrained(tokenizer)

    if tokenizer.startswith("roberta"):
        return pytorch_transformers.RobertaTokenizer.from_pretrained(tokenizer)

    if tokenizer.startswith("gpt2"):
        return pytorch_transformers.GPT2Tokenizer.from_pretrained(tokenizer)

    raise RuntimeError("Don't know tokenizer {}".format(tokenizer))
