from torch.utils.data import Dataset
from sentence_transformers import InputExample


class TopicalChatDataset(Dataset):
    def __init__(self, dataset):
        contexts, pos_resps, neg_resps = self.preprocess(dataset)
        self.contexts = contexts
        self.pos_resps = pos_resps
        self.neg_resps = neg_resps

    def __getitem__(self, item):
        return InputExample(texts=[
            self.contexts[item], self.pos_resps[item], self.neg_resps[item]
        ])

    def __len__(self):
        return len(self.contexts)

    def preprocess(self, dataset):
        contexts, pos_resps, neg_resps = [], [], []
        for i, data in enumerate(dataset):
            contexts.append(data["context"])
            pos_resps.append(data["pos_reps"])
            neg_resps.append(data["neg_reps"])
        assert len(contexts) == len(pos_resps) == len(neg_resps)
        return contexts, pos_resps, neg_resps


class EcommerceDataset(Dataset):
    def __init__(self, dataset):
        labels, contexts, pos_resps, neg_resps = self.preprocess(dataset)
        self.labels = labels
        self.contexts = contexts
        self.pos_resps = pos_resps
        self.neg_resps = neg_resps

    def __getitem__(self, item):
        return InputExample(texts=[
            self.contexts[item], self.pos_resps[item], self.neg_resps[item]
        ])

    def __len__(self):
        return len(self.labels)

    def preprocess(self, dataset):
        labels, contexts, pos_resps, neg_resps = [], [], [], []
        for i, data in enumerate(dataset):
            if i % 2 == 0:
                labels.append(data[0])
                contexts.append(data[-2].replace(" ", ""))
                pos_resps.append(data[-1].replace(" ", ""))
            else:
                neg_resps.append(data[-1].replace(" ", ""))
        assert len(labels) == len(contexts) == len(pos_resps) == len(neg_resps)
        return labels, contexts, pos_resps, neg_resps