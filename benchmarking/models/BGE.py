import torch
import torch.nn.functional as F

from models.Model import Model
from utils.notebook_utils import running_in_notebook
from utils.logging_utils import write_evaluations


class BGE(Model):
    """
    BGE Small Model.
    """

    def __init__(self):
        super().__init__(name="BGE-small")

    def load(self):
        """
        Load the model from memory or from Hugging Face
        """
        self.load_hf_model(path="BAAI/bge-small-en-v1.5")
        self.load_hf_tokenizer(path="BAAI/bge-small-en-v1.5")

    def evaluate_split(self, dataset, split="train"):
        """
        Evaluate the model on the specified dataset
        """
        return dataset.evaluate(model=self, split=split, batch=True)

    def generate(self, prompt):
        """
        Generate an response for the specified prompt
        """
        inputs = self.tokenizer(
            prompt, padding=True, truncation=True, return_tensors="pt"
        )

        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**inputs)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings[0]

    def generate_batch(self, prompts):
        """
        Generate an response for the specified prompt
        """
        inputs = self.tokenizer(
            prompts, padding=True, truncation=True, return_tensors="pt"
        )

        self.model.eval()
        with torch.no_grad():
            model_output = self.model(**inputs)
            sentence_embeddings = model_output[0][:, 0]
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings
