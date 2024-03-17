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

    def evaluate_split(self, dataset, split="train"):
        return dataset.evaluate(model=self, split=split, batch=True)

    def generate_batch(self, prompts):
        """
        Generate batched responses from ollama
        """
        responses = self.generate_ollama(prompts, model="llama2")
        return responses
