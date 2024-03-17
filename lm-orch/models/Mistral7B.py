from models.Model import Model
from utils.notebook_utils import running_in_notebook
from utils.logging_utils import write_evaluations


class Mistral7B(Model):
    """
    Mistral 7B Model.
    """

    def __init__(self):
        super().__init__(name="Mistral-7B")

    def load(self):
        """
        Load the model from memory or from Hugging Face
        """
        self.model = self.generate_ollama_response

    def evaluate_split(self, dataset, split="train"):
        """
        Evaluate the model on the specified dataset
        """
        return dataset.evaluate(model=self, split=split, batch=True)

    def generate(self, prompt):
        """
        Generate an response for the specified prompt
        """
        return self.fetch_ollama(prompt, model="mistral")

    def generate_batch(self, prompts):
        """
        Generate an response for the specified prompt
        """
        responses = []
        for i, prompt in enumerate(prompts):
            print(i + 1, end="\n" if (i + 1) % 20 == 0 else " ")
            responses.append(self.fetch_ollama(prompt, model="mistral"))
        return responses
