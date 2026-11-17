from murrai.adapters.corpus import DirTextCorpus, TextCorpus
from murrai.application.runs import train_and_held_out, train_bigram


def train_from_directory(directory, holdout=0.1):
    text = DirTextCorpus(directory).text()
    train_text, held_text = train_and_held_out(text, holdout)
    return train_bigram(TextCorpus(train_text), held_out_text=held_text)
