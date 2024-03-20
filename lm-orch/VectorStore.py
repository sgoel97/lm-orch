from typing import List
import json
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import cosine_similarity

from models import BGE


class Node:
    def __init__(self, data: Tensor, metadata: dict = None):
        if not isinstance(data, Tensor):
            data = torch.tensor(data)
        self.data = data.type(torch.float)
        self.metadata = metadata

    def json(self):
        return {
            "data": self.data.tolist(),
            "metadata": self.metadata,
        }


class VectorStore:
    def __init__(self, data: List[Node] = [], encoder=None):
        self.data = np.array(data)
        if encoder is None:
            encoder = BGE()
        self.encoder = encoder

    def encode(self, text: str):
        return self.encoder.generate(text)

    def encode_batch(self, text: List[str]):
        return self.encoder.generate_batch(text)

    def from_dataset(self, dataset):
        encoded_prompts = self.encode_batch(dataset["original_prompt"])
        nodes = [Node(encoded_prompts[i], dataset[i]) for i in range(len(dataset))]
        self.data = np.array(nodes)

    def persist(self, path: str = "./data/vectorstore.json"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [node.json() for node in self.data]
        with open(path, "w") as f:
            f.write(json.dumps(data))

    def load(self, path: str = "./data/vectorstore.json"):
        path = Path(path)
        with open(path, "r") as f:
            data = json.loads(f.read())
        self.data = np.array([Node(**node) for node in data])

    def retrieve(
        self, query: str | Tensor, k: int = 1, measure: str = "cos"
    ) -> List[Node]:
        if isinstance(query, str):
            query = self.encode(query).view(1, -1)
        query = query.type(torch.float)

        embedding_matrix = torch.stack(list(map(lambda x: x.data, self.data)))

        if measure == "cos":
            dist_vector = cosine_similarity(query, embedding_matrix, dim=1)
        else:
            raise NotImplementedError

        top_k_indices = torch.topk(dist_vector, k, dim=0).indices

        if k == 1:
            return [self.data[top_k_indices]]
        return self.data[top_k_indices]
