import aiohttp
import asyncio
import requests
import numpy as np
from typing import List
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from threading import Thread

from utils.notebook_utils import running_in_notebook
from utils.logging_utils import write_evaluations


class Model:
    def __init__(self, name, hf=False):
        """
        Initialize the model with the given name.
        """
        self.name = name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.hf = hf
        self.eval_history = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load()

    def load_hf_model(self, path: str, **kwargs):
        """
        Load the model from Hugging Face
        """
        self.hf = True
        print(f"Loading {path} from Hugging Face")
        if path in ["BAAI/bge-small-en-v1.5"]:
            self.model = AutoModel.from_pretrained(
                path, trust_remote_code=True, **kwargs
            ).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, trust_remote_code=True, **kwargs
            ).to(self.device)

    def load_hf_tokenizer(self, path: str, **kwargs):
        """
        Load the tokenizer from Hugging Face
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=True, **kwargs
        )

    def load(self):
        """
        Load the model from memory or from Hugging Face
        """
        raise NotImplementedError

    def fetch_ollama(self, prompt: str, model: str = "llama2"):
        """
        Fetches a response from the Ollama API using the provided prompt and model.
        """
        if "system_prompt" not in self.__dict__:
            self.system_prompt = "You are a helpful assistant."

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "system": self.system_prompt,
            },
        )

        return response.json()["response"]

    async def fetch_ollama_response(self, session, prompt: str, model: str = "llama2"):
        """
        Asynchronously fetches a response from the Ollama API using the provided session, prompt, and model.
        """
        if "system_prompt" not in self.__dict__:
            self.system_prompt = "You are a helpful assistant."

        async with session.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "system": self.system_prompt,
            },
        ) as response:
            return await response.json()

    async def generate_ollama_response(self, prompts: List[str], model: str = "llama2"):
        """
        Asynchronously generates Ollama responses based on the given prompts using the specified model.
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for prompt in prompts:
                tasks.append(self.fetch_ollama_response(session, prompt, model))

            responses = await asyncio.gather(*tasks)

            self.responses = responses

            return responses

    def generate_ollama(
        self, prompts: str | List[str], model: str = "llama2", batch_size=16
    ) -> str:
        """
        A function to generate ollama responses based on given prompts and a specified model.
        """

        def run_async(async_func, prompts, model):
            def run(loop, async_func, prompts, model):
                asyncio.set_event_loop(loop)
                loop.run_until_complete(async_func(prompts, model))

            loop = asyncio.new_event_loop()
            t = Thread(target=run, args=(loop, async_func, prompts, model))
            t.start()
            t.join()

        if isinstance(prompts, str):
            prompts = [prompts]

        prompts = np.array(prompts)
        if running_in_notebook():
            responses = []
            for i in tqdm(range(0, len(prompts), batch_size)):
                run_async(
                    self.generate_ollama_response, prompts[i : i + batch_size], model
                )
                responses.extend(self.responses)
        else:
            responses = []
            for i in tqdm(range(0, len(prompts), batch_size)):
                self.responses = asyncio.run(
                    self.generate_ollama_response(prompts[i : i + batch_size], model)
                )
                responses.extend(self.responses)

        if isinstance(prompts, str):
            return responses[0]["response"]

        return list(map(lambda response: response["response"], responses))

    def evaluate(
        self,
        dataset,
        split="train",
        save=True,
        augment_config={},
        log_prefix="",
        log_suffix="",
    ):
        """
        Evaluate the model on the specified dataset
        """
        self.system_prompt = dataset.system_prompt
        dataset.augment_config = augment_config

        evaluations = self.evaluate_split(dataset, split)

        if save:
            write_evaluations(
                evaluations, self.name, dataset.name, split, log_prefix, log_suffix
            )

        return evaluations

    def evaluate_split(self, dataset, split="train"):
        """
        Evaluate the model on the specified dataset
        """
        return dataset.evaluate(model=self, split=split, batch=False)

    def generate(self, prompt: str) -> str:
        """
        Generate an response for the specified prompt
        """
        raise NotImplementedError

    def generate_batch(self, prompts: list) -> list:
        """
        Generate an response for the specified prompts
        """
        raise NotImplementedError
