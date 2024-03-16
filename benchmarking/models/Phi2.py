import requests
from models.Model import Model


class Phi2(Model):
    """
    Phi 2 Model.
    """

    def __init__(self, hf=True):
        super().__init__(name="Phi-2", hf=hf)

    def load(self):
        """
        Load the model from memory or from Hugging Face
        """
        if self.hf:
            self.load_hf_model(path="microsoft/phi-2")
            self.load_hf_tokenizer(path="microsoft/phi-2", use_fast=False)

            self.tokenizer.add_tokens(["<PAD>"])
            self.tokenizer.pad_token = "<PAD>"
            self.tokenizer.add_special_tokens(dict(eos_token="<|endoftext|>"))
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        else:
            self.model = self.generate_ollama_response

    def evaluate_split(self, dataset, split="train"):
        """
        Evaluate the model on the specified dataset
        """
        return dataset.evaluate(model=self, split=split, batch=True)

    def generate(self, prompt: str) -> str:
        """
        Generate an response for the specified prompt
        """
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi",
                "prompt": prompt,
                "stream": False,
            },
        )
        return response.json()["response"]

    def generate_batch(self, prompts):
        """
        Generate an response for the specified prompt
        """

        if self.hf:

            def prompt_template(prompt):
                return f"Instruct: {self.system_prompt} {prompt}\n\Output:"

            prompts = list(map(prompt_template, prompts))

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=False,
            )

            # max_length = max(map(len, inputs["input_ids"]))
            self.model.eval()
            outputs = self.model.generate(**inputs, use_cache=True, max_new_tokens=10)
            responses = self.tokenizer.batch_decode(outputs)

            for i in range(len(responses)):
                responses[i] = (
                    responses[i]
                    .replace(prompts[i], "")
                    .replace("\n<|endoftext|>", "")
                    .replace("<PAD>", "")
                )

            return responses

        else:
            responses = self.generate_ollama(prompts, model="phi")
            return responses
