import argparse
import json

DATASETS = ["e_commerce", "topicalchat"]


def parser_args():
    parser = argparse.ArgumentParser("Dialog rerank")
    parser.add_argument("--train_batch_size", default=64, type=int)
    parser.add_argument("--model_name", default="bert-base-uncased", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--pooling", default="mean", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--dataset", default="e_commerce", type=str, choices=DATASETS)
    parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
    args = parser.parse_args()
    return args


def load_e_commerce(path):
    dataset = []
    with open(path, "r", encoding="utf8") as fr:
        for line in fr.readlines():
            line = line.strip("\n").split("\t")
            dataset.append(line)
    return dataset


def load_topicalchat(path):
    return json.load(open(path, "r", encoding="utf8"))