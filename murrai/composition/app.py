from murrai.adapters.corpus import DirTextCorpus
from murrai.application.runs import train_bigram


def train_from_directory(directory, holdout=0.1):
    return train_bigram(DirTextCorpus(directory), holdout)
