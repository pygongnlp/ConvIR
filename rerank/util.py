import argparse

DATASETS = ["e_commerce", "topicalchat"]


def parser_args():
    parser = argparse.ArgumentParser("Dialog Retrieval")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--dataset", default="e_commerce", type=str, choices=DATASETS)
    parser.add_argument("--model_name", default="bert-base-uncased", type=str)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--evaluation_steps", default=1000, type=int)
    args = parser.parse_args()
    return args
