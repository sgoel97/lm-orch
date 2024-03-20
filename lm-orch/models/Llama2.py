from typing import List

from models.Model import Model


class Llama2(Model):
    """
    Llama 2 Model.

    This model is pulled from ollama
    <https://ollama.com/library/llama2>
    """

    def __init__(self):
        super().__init__(name="Llama-2")

    def load(self):
        """
        Specify that the current model is loaded from Ollama
        """
        self.model = self.generate_ollama_response

    def generate(self, prompt: str, system_prompt: str = None):
        """
        Generate a single response from ollama
        """
        return self.fetch_ollama(prompt, system_prompt, model="llama2")

    def generate_batch(self, prompts: List[str], system_prompt: str = None):
        """
        Generate batched responses from ollama
        """
        responses = self.generate_ollama(prompts, system_prompt, model="llama2")
        return responses
