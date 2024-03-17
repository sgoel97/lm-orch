import torch
import torch.nn.functional as F

from models.Model import Model


class BGE(Model):
    """
    BGE Small Model.

    This model is pulled from the huggingface transformers package
    <https://huggingface.co/BAAI/bge-small-en-v1.5>
    """

    def __init__(self):
        super().__init__(name="BGE-small")

    def load(self):
        """
        Load the model from Hugging Face
        """
        self.load_hf_model(path="BAAI/bge-small-en-v1.5")
        self.load_hf_tokenizer(path="BAAI/bge-small-en-v1.5")

    def evaluate_split(self, dataset, split="train"):
        return dataset.evaluate(model=self, split=split, batch=True)

    def generate(self, prompt):
        """
        Generate response for prompt using HuggingFace API
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
        Generate batched response for prompts using HuggingFace API
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
