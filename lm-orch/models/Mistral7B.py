from typing import List

from models.Model import Model


class Mistral7B(Model):
    """
    Mistral 7B Model.

    This model is pulled from ollama
    <https://ollama.com/library/mistral>
    """

    def __init__(self):
        super().__init__(name="Mistral-7B")

    def load(self):
        """
        Specify that the current model is loaded from Ollama
        """
        self.model = self.generate_ollama_response

    def generate(self, prompt: str, system_prompt: str = None):
        """
        Generate a single response from ollama
        """
        return self.fetch_ollama(prompt, system_prompt, model="mistral")

    def generate_batch(self, prompts: List[str], system_prompt: str = None):
        """
        Generate batched responses from ollama
        """
        responses = []
        for i, prompt in enumerate(prompts):
            responses.append(self.fetch_ollama(prompt, system_prompt, model="mistral"))
        return responses
