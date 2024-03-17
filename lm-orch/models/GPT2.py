from models.Model import Model


class GPT2(Model):
    """
    GPT 2 Model.

    This is the smallest version of GPT-2, with 124M parameters.

    This model is pulled from the huggingface transformers package
    <https://huggingface.co/openai-community/gpt2>
    """

    def __init__(self):
        super().__init__(name="GPT 2", hf=True)

    def load(self):
        """
        Load the model from Hugging Face
        """
        self.load_hf_model(path="openai-community/gpt2")

        self.load_hf_tokenizer(path="openai-community/gpt2")
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def evaluate_split(self, dataset, split="train"):
        return dataset.evaluate(model=self, split=split, batch=True)

    def generate(self, prompt):
        """
        Generate response for prompt using HuggingFace API
        """
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        self.model.eval()
        outputs = self.model.generate(**inputs, max_new_tokens=64)
        response = self.tokenizer.batch_decode(outputs)

        response[0] = response[0].replace(self.tokenizer.pad_token, "")
        response[0] = response[0].replace(prompt, "")

        return response[0]

    def generate_batch(self, prompts):
        """
        Generate batched response for prompts using HuggingFace API
        """
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(self.device)

        self.model.eval()
        outputs = self.model.generate(**inputs, max_new_tokens=64)
        responses = self.tokenizer.batch_decode(outputs)

        for i in range(len(responses)):
            responses[i] = responses[i].replace(self.tokenizer.pad_token, "")
            responses[i] = responses[i].replace(prompts[i], "")

        return responses
