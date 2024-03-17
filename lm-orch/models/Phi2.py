import requests
from models.Model import Model


class Phi2(Model):
    """
    Phi 2 Model.

    This model can be pulled from ollama or Hugging Face
    <https://huggingface.co/microsoft/phi-2>
    <https://ollama.com/library/phi>
    """

    def __init__(self, hf=True):
        super().__init__(name="Phi-2", hf=hf)

    def load(self):
        """
        Load the model from ollama or from Hugging Face

        Defaults to Hugging Face
        """
        if self.hf:
            self.load_hf_model(path="microsoft/phi-2")
            self.load_hf_tokenizer(path="microsoft/phi-2", use_fast=False)
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        else:
            self.model = self.generate_ollama_response

    def evaluate_split(self, dataset, split="train"):
        return dataset.evaluate(model=self, split=split, batch=True)

    def generate(self, prompt: str) -> str:
        """
        Generate a response for the specified prompt
        """
        if self.hf:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            self.model.eval()
            outputs = self.model.generate(**inputs, use_cache=True, max_new_tokens=32)
            response = self.tokenizer.batch_decode(outputs)

            response[0] = response[0].replace("<|endoftext|>", "").replace(prompt, "")

            return response[0]
        else:
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
        Generate batched responses for the specified prompts
        """
        if self.hf:

            def prompt_template(prompt):
                if "system_prompt" not in self.__dict__:
                    return f"Instruct: {prompt}\n\Output:"
                else:
                    return f"Instruct: {self.system_prompt} {prompt}\n\nOutput:"

            prompts = list(map(prompt_template, prompts))

            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)

            self.model.eval()
            outputs = self.model.generate(**inputs, use_cache=True, max_new_tokens=32)
            responses = self.tokenizer.batch_decode(outputs)

            for i in range(len(responses)):
                responses[i] = (
                    responses[i].replace("<|endoftext|>", "").replace(prompts[i], "")
                )

            return responses

        else:
            responses = self.generate_ollama(prompts, model="phi")
            return responses
