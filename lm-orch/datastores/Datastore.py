from typing import List
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset, DatasetDict
from VectorStore import VectorStore
from utils.logging_utils import write_evaluations


class Datastore:
    def __init__(self, name: str, system_prompt: str = None, seed: int = None):
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

    def embed(self, split: str = "train"):
        """
        Embed the data using the vector store
        """
        store_path = f"./data/{self.name}_{split}.json"

        if self.store is None:
            self.store = VectorStore()

        if Path(store_path).exists() and "sample" not in store_path:
            self.store.load(store_path)
            print(f"Loaded {split} vector store from", store_path)
        else:
            self.store.from_dataset(self.hf_dataset[split])
            self.store.persist(store_path)
            print(f"Persisted {split} vector store to", store_path)

    def augment(self, split: str = "train"):
        """
        Augment the data with context from the vector store
        """
        if self.augment_config is not None:
            self.embed(split)
            self.hf_dataset[split] = self.hf_dataset[split].map(self.augment_row)

    def get_context_string(self, nodes: list):
        context_str = []
        for node in nodes:
            context = f"Question: {node.metadata['original_prompt']}\nAnswer: {node.metadata['answer']}"
            context_str.append(context)
        context_str = "\n\n".join(context_str)
        return context_str

    def get_context_template(self, context: str, prompt: str):
        if context == "":
            return prompt
        return f"We have provided context information below. \n---------------------\n{context}\n---------------------\nGiven this information, please answer the question: {prompt}\n"

    def augment_row(self, row):
        """
        Augment one datapoint with context from the vector store
        """
        k = self.augment_config.get("k", 0)
        measure = self.augment_config.get("measure", "cos")
        nodes = self.store.retrieve(row["original_prompt"], k=k, measure=measure)

        context_str = self.get_context_string(nodes)
        prompt = self.get_context_template(context_str, row["original_prompt"])
        row["prompt"] = prompt
        return row

    def process_row(self, row):
        """
        Pre-processing a row of the dataset
        """
        return row

    def sample(self, sample_size: int, split: str = "train"):
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

    def evaluate(
        self,
        model,
        split: str = "train",
        batch: bool = False,
        augment_config: dict | None = None,
        save: bool = False,
        log_prefix: str = "",
        log_suffix: str = "",
    ):
        """
        Evaluate the model on the dataset
        """
        print(f"Evaluating {model.name} on {self.name} on the {split} split")
        self.augment_config = augment_config
        self.augment(split)

        dataset = self.hf_dataset[split]

        if batch:
            responses = model.generate_batch(dataset["prompt"], self.system_prompt)

        evaluations = []
        for i in tqdm(range(len(dataset))):
            prompt, answer = dataset[i]["prompt"], dataset[i]["answer"]

            if batch:
                response = responses[i]
            else:
                response = model.generate(prompt, self.system_prompt)

            evaluation = dataset[i].copy()
            evaluation["response"] = response
            evaluation["correct"] = self.evaluate_response(response, answer)
            evaluations.append(evaluation)

        if save:
            write_evaluations(
                evaluations, self.name, dataset.name, split, log_prefix, log_suffix
            )

        return evaluations

    def evaluate_response(self, generated_answer: str, answer: str) -> bool | float:
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
