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

    def fetch_ollama(
        self, prompt: str, system_prompt: str = None, model: str = "llama2"
    ):
        """
        Fetches a response from the Ollama API using the provided prompt and model.
        """
        content = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        if system_prompt is not None:
            content["system"] = system_prompt

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=content,
        )

        return response.json()["response"]

    async def fetch_ollama_response(
        self, session, prompt: str, system_prompt: str = None, model: str = "llama2"
    ):
        """
        Asynchronously fetches a response from the Ollama API using the provided session, prompt, and model.
        """
        content = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        if system_prompt is not None:
            content["system"] = system_prompt

        async with session.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "system": system_prompt,
            },
        ) as response:
            return await response.json()

    async def generate_ollama_response(
        self, prompts: List[str], system_prompt: str = None, model: str = "llama2"
    ):
        """
        Asynchronously generates Ollama responses based on the given prompts using the specified model.
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            for prompt in prompts:
                tasks.append(
                    self.fetch_ollama_response(session, prompt, system_prompt, model)
                )

            responses = await asyncio.gather(*tasks)

            self.responses = responses

            return responses

    def generate_ollama(
        self,
        prompts: List[str],
        system_prompt: str = None,
        model: str = "llama2",
        batch_size=16,
    ) -> str:
        """
        A function to generate ollama responses based on given prompts and a specified model.
        """

        def run_async(async_func, *args):
            def run(loop, async_func, *args):
                asyncio.set_event_loop(loop)
                loop.run_until_complete(async_func(*args))

            loop = asyncio.new_event_loop()
            t = Thread(target=run, args=(loop, async_func, *args))
            t.start()
            t.join()

        prompts = np.array(prompts)
        responses = []
        for i in tqdm(range(0, len(prompts), batch_size)):
            batched_prompts = prompts[i : i + batch_size]

            if running_in_notebook():
                run_async(
                    self.generate_ollama_response, batched_prompts, system_prompt, model
                )
            else:
                self.responses = asyncio.run(
                    self.generate_ollama_response(batched_prompts, system_prompt, model)
                )

            responses.extend(self.responses)

        return list(map(lambda response: response["response"], responses))

    def evaluate(
        self,
        dataset,
        split="train",
        batch=False,
        augment_config=None,
        save=True,
        log_prefix="",
        log_suffix="",
    ):
        """
        Evaluate the model on the specified dataset
        """
        evaluations = dataset.evaluate(
            model=self,
            split=split,
            batch=batch,
            augment_config=augment_config,
            save=save,
            log_prefix=log_prefix,
            log_suffix=log_suffix,
        )

        return evaluations

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate an response for the specified prompt
        """
        raise NotImplementedError

    def generate_batch(self, prompts: list, system_prompt: str = None) -> list:
        """
        Generate an response for the specified prompts
        """
        raise NotImplementedError
