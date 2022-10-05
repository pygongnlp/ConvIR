import logging
import json
from torch.utils.data import DataLoader

from util import parser_args
from sentence_transformers import LoggingHandler, util, InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def main():
    args = parser_args()

    args.model_save_path = f"output/retrieval/training_{args.dataset}_{args.model_name}"
    logging.info(args)

    model = CrossEncoder(args.model_name, num_labels=1, max_length=512)

    train_dataset = json.load(open("data/topicalchat/train.json", "r", encoding="utf8"))
    print(f"Loading TopicalChat Dataset {len(train_dataset)}")
    train_samples, dev_samples = [], {}
    for i, data in enumerate(train_dataset):
        if i % 100 == 0 and i != 0:
            dev_samples[i] = {
                "query": data["context"],
                "positive": data["pos_reps"],
                "negative": data["neg_reps"]
            }
            continue
        if i % 5 == 0:
            train_samples.append(InputExample(texts=[data["context"], data["pos_reps"]], label=1))
        else:
            train_samples.append(InputExample(texts=[data["context"], data["neg_reps"]], label=0))

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)

    evaluator = CERerankingEvaluator(dev_samples, name="train_eval")

    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=args.epochs,
              output_path=args.model_save_path,
              evaluation_steps=args.evaluation_steps,
              warmup_steps=args.warmup_steps,
              save_best_model=True,
              use_amp=True)
    model.save(args.model_save_path + "-latest")


if __name__ == '__main__':
    main()
