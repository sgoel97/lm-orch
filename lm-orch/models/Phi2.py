from typing import List
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

    def template(self, prompts: str | List[str], system_prompt: str = None):
        if isinstance(prompts, str):
            if system_prompt is not None:
                prompts = f"{system_prompt} {prompts}"

            return f"Instruct: {prompts}\n\nOutput:"
        else:
            return [self.template(p, system_prompt) for p in prompts]

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate a response for the specified prompt
        """
        if self.hf:
            prompt = self.template(prompt, system_prompt)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            self.model.eval()
            outputs = self.model.generate(**inputs, use_cache=True, max_new_tokens=32)
            response = self.tokenizer.batch_decode(outputs)

            response[0] = response[0].replace("<|endoftext|>", "").replace(prompt, "")

            return response[0]
        else:
            response = self.fetch_ollama(prompt, system_prompt, model="phi")
            return response

    def generate_batch(self, prompts: List[str], system_prompt: str = None):
        """
        Generate batched responses for the specified prompts
        """
        if self.hf:
            prompts = self.template(prompts, system_prompt)

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
            responses = self.generate_ollama(prompts, system_prompt, model="phi")
            return responses
