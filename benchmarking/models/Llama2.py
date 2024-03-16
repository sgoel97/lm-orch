from models.Model import Model
from utils.logging_utils import write_evaluations


class Llama2(Model):
    """
    Llama 2 Model.
    """

    def __init__(self):
        super().__init__(name="Llama-2")

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

    def generate_batch(self, prompts):
        """
        Generate an response for the specified prompt
        """
        max_len = max(map(len, prompts))
        print(
            f"Generating, Expected time: {len(prompts) * 1.6 / 60 * max_len / 45:.2f} minutes"
        )
        responses = self.generate_ollama(prompts, model="llama2")
        print(responses)
        return responses
