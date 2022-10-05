import logging
from torch.utils.data import DataLoader

from sentence_transformers import LoggingHandler
from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
from util import parser_args, load_e_commerce, load_topicalchat
from data import EcommerceDataset, TopicalChatDataset

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def main():
    args = parser_args()
    args.model_save_path = f"output/retrieval/{args.dataset}/{args.model_name}"
    logging.info(args)

    if args.use_pre_trained_model:
        logging.info("Use pretrained SBERT model")
        model = SentenceTransformer(args.model_name)
        model.max_seq_length = args.max_seq_length
    else:
        logging.info("Create new SBERT model")
        word_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    if args.dataset == "e_commerce":
        train_dataset = load_e_commerce(path=f"data/e-commerce/train.txt")
        train_dataset = EcommerceDataset(train_dataset)
        print(f"Loading E-Commerce Dataset {len(train_dataset)}")
    elif args.dataset == "ubuntu":
        pass
    elif args.dataset == "douban":
        pass
    elif args.dataset == "topicalchat":
        train_dataset = load_topicalchat(path=f"data/topicalchat/train.json")
        train_dataset = TopicalChatDataset(train_dataset)
        print(train_dataset[0])
        print(f"Loading TopicalChat Dataset {len(train_dataset)}")

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)
    loss = losses.MultipleNegativesRankingLoss(model=model)

    model.fit(train_objectives=[(train_dataloader, loss)],
              epochs=args.epochs,
              warmup_steps=args.warmup_steps,
              use_amp=True,
              checkpoint_path=args.model_save_path,
              checkpoint_save_steps=len(train_dataloader),
              optimizer_params={'lr': args.lr},
              )
    model.save(args.model_save_path)


if __name__ == '__main__':
    main()