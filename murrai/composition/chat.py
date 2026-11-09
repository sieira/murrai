from murrai.adapters.corpus import DirTextCorpus
from murrai.application.runs import train_bigram, train_mlp_lm, train_neural_bigram
from murrai.presentation.chat.picker import Stage
from murrai.presentation.chat.session import ChatSession


def chat_session_from_directory(directory, seed=0):
    result = train_bigram(DirTextCorpus(directory), holdout=0.1)
    return ChatSession(result.model, seed)


def neural_chat_session_from_directory(directory, seed=0, steps=2000, batch=64):
    result = train_neural_bigram(
        DirTextCorpus(directory).text(),
        steps=steps,
        batch=batch,
        seed=seed,
        learning_rate=3.0,
    )
    return ChatSession(result.model, seed)


def mlp_chat_session_from_directory(directory, seed=0, steps=5000, batch=64):
    result = train_mlp_lm(
        DirTextCorpus(directory).text(),
        steps=steps,
        batch=batch,
        seed=seed,
        learning_rate=0.5,
        patience=500,
        window=4,
    )
    return ChatSession(result.model, seed)


def chat_stages(directory, seed=0, neural_steps=2000, mlp_steps=5000):
    return [
        Stage(
            "counted bigram",
            "next-char odds tallied straight from the corpus",
            lambda: chat_session_from_directory(directory, seed),
        ),
        Stage(
            "neural bigram",
            "same memory, trained by gradient descent",
            lambda: neural_chat_session_from_directory(
                directory, seed, steps=neural_steps
            ),
            trains=True,
        ),
        Stage(
            "windowed mlp",
            "four chars of memory through a hidden layer",
            lambda: mlp_chat_session_from_directory(directory, seed, steps=mlp_steps),
            trains=True,
        ),
    ]
