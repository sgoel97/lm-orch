from transformers import AutoModelForCausalLM

from models.Model import Model


class GPT2(Model):
    """
    GPT 2 Model.
    """

    def __init__(self):
        super().__init__(name="GPT 2", hf=True)

    def load(self):
        """
        Load the model from memory or from Hugging Face
        """
        self.load_hf_model(path="openai-community/gpt2")

        self.load_hf_tokenizer(path="openai-community/gpt2")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def evaluate_split(self, dataset, split="train"):
        """
        Evaluate the model on the specified dataset
        """
        return dataset.evaluate(model=self, split=split, batch=True)

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        self.model.eval()
        outputs = self.model.generate(**inputs)
        response = self.tokenizer.batch_decode(outputs)

        response[0] = response[0].replace(self.tokenizer.pad_token, "")
        response[0] = response[0].replace(prompt, "")

        return response[0]

    def generate_batch(self, prompts):
        """
        Generate an response for the specified prompt
        """
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        )

        self.model.eval()
        outputs = self.model.generate(**inputs)
        responses = self.tokenizer.batch_decode(outputs)
        print(inputs)

        for i in range(len(responses)):
            responses[i] = responses[i].replace(self.tokenizer.pad_token, "")
            responses[i] = responses[i].replace(prompts[i], "")

        return responses
