from typing import List
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, DatasetDict
from VectorStore import VectorStore


class Datastore:
    def __init__(self, name, system_prompt=None, seed=None):
        """
        Initialize the class with the given name. Set class attributes to None and then call the load and process methods.
        """
        self.name = name
        self.system_prompt = system_prompt
        self.hf_dataset = seed
        self.store = None
        self.augment_config = None

        self.load()
        self.process()

    def set_seed(self, seed: int):
        """
        Set the seed for reproducibility
        """
        np.random.seed(seed)

    def load_hf_dataset(self, path: str, **kwargs):
        """
        Load the dataset from Hugging Face
        """
        print(f"Loading {self.name} from Hugging Face")

        self.hf_dataset = load_dataset(path, trust_remote_code=True, **kwargs)

        if type(self.hf_dataset) == list:
            self.hf_dataset = DatasetDict(
                {
                    self.hf_dataset[i].split._name: self.hf_dataset[i]
                    for i in range(len(self.hf_dataset))
                }
            )

        for df_name in self.hf_dataset.keys():
            df = self.hf_dataset[df_name]
            df = df.add_column("id", range(len(df)))
            df = df.add_column("split", [df_name] * len(df))
            self.hf_dataset[df_name] = df

    def process_hf_dataset(self, function):
        """
        Apply pre-processing to datasets from Hugging Face
        """
        for df_name in self.hf_dataset.keys():
            self.hf_dataset[df_name] = self.hf_dataset[df_name].map(function)

    def load(self):
        """
        Load the dataset from memory or from Hugging Face
        """
        raise NotImplementedError

    def process(self):
        """
        Pre-processing of the dataset
        """
        self.process_hf_dataset(function=self.process_row)

        def add_prompt_and_answer(row):
            row["prompt"] = self.get_prompt(row)
            row["answer"] = self.get_answer(row)
            row["original_prompt"] = row["prompt"]
            return row

        self.process_hf_dataset(function=add_prompt_and_answer)

    def embed(self, split="train"):
        """
        Embed the data using the vector store
        """
        store_path = f"./data/{self.name}_{split}.json"

        if self.store is None:
            self.store = VectorStore()

        if Path(store_path).exists():
            self.store.load(store_path)
            print(f"Loaded {split} vector store from", store_path)
        else:
            self.store.from_dataset(self.hf_dataset[split])
            self.store.persist(store_path)
            print(f"Persisted {split} vector store to", store_path)

    def augment(self, split="train"):
        """
        Augment the data
        """
        self.embed(split)
        self.hf_dataset[split] = self.hf_dataset[split].map(self.augment_row)

    def augment_row(self, row):
        k = self.augment_config.get("k", 0)
        measure = self.augment_config.get("measure", "cos")
        nodes = self.store.retrieve(row["prompt"], k=k, measure=measure)

        context_str = ""
        for node in nodes:
            context_str += f"Question: {node.metadata['prompt']}\nAnswer: {node.metadata['answer']}\n\n"

        if len(nodes) > 0:
            prompt = f"We have provided context information below. \n---------------------\n{context_str}\n---------------------\nGiven this information, please answer the question: {row['prompt']}\n"
        else:
            prompt = row["original_prompt"]

        row["prompt"] = prompt
        return row

    def process_row(self, row):
        """
        Pre-processing a row of the dataset
        """
        return row

    def sample(self, sample_size, split: str = "train"):
        dataset = self.hf_dataset[split]

        a = range(len(dataset))
        random_idx = np.random.choice(a, sample_size, replace=False)
        sampled_dataset = dataset.select(random_idx)

        self.hf_dataset[split + "_sample"] = sampled_dataset

        print(f"Created sample of size {sample_size} for {split}_sample dataset")

    def get_prompt(self, row) -> str:
        """
        Get the prompt for the specified row.
        """

        raise NotImplementedError

    def get_answer(self, row) -> str:
        """
        Get the answer for the specified row.
        """
        raise NotImplementedError

    def evaluate(self, model, split: str = "train", batch=False):
        """
        Evaluate the model on the dataset
        """
        print(f"Evaluating {model.name} on {self.name} on the {split} split")

        if "augment_config" in self.__dict__ and self.augment_config.get("k", 0) > 0:
            self.augment(split)

        self.evaluate_split(model, split, batch=batch)

    def evaluate_split(self, model, split: str = "train", batch=False):
        """
        Evaluate the model on the specified dataset
        """
        dataset = self.hf_dataset[split]

        if batch:
            responses = model.generate_batch(dataset["prompt"])

        evaluations = []
        for i in tqdm(range(len(dataset))):
            prompt, answer = dataset[i]["prompt"], dataset[i]["answer"]

            response = responses[i] if batch else model.generate(prompt)

            evaluation = dataset[i].copy()
            evaluation["response"] = response
            evaluation["correct"] = self.evaluate_response(response, answer)
            evaluations.append(evaluation)

        return evaluations

    def evaluate_response(self, generated_answer: str, answer: str) -> bool:
        """
        Deterime whether the generated answer is correct or not
        """
        raise NotImplementedError

    def evaluate_responses(
        self, generated_answers: List[str], answers: List[str]
    ) -> List[bool | float]:
        """
        Deterime whether the generated answers are correct or not
        """
        raise NotImplementedError
