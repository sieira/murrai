from murrai.adapters.corpus import DirTextCorpus
from murrai.application.runs import train_bigram
from murrai.application.session import ChatSession


def chat_session_from_directory(directory, seed=0):
    result = train_bigram(DirTextCorpus(directory), holdout=0.1)
    return ChatSession(result.model, seed)
