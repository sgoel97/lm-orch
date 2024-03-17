import json
import subprocess
from pathlib import Path
from openai import OpenAI

from models.Model import Model
from utils.file_utils import read_data, write_data


class GPT3(Model):
    """
    GPT 3.5 Model.
    """

    def __init__(self):
        super().__init__(name="GPT 3.5")

    def load(self):
        """
        Load the model from memory or from Hugging Face
        """
        client = OpenAI()
        self.model = client.chat.completions

    def evaluate_split(self, dataset, split="train"):
        """
        Evaluate the model on the specified dataset
        """
        return dataset.evaluate(model=self, split=split, batch=True)

    def message_template(self, prompt):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return messages

    def generate(self, prompt):
        """
        Generate an response for the specified prompt
        """
        messages = self.message_template(prompt)
        response = self.model.create(model="gpt-3.5-turbo", messages=messages)
        return response.choices[0].message.content

    def generate_batch(self, prompts):
        """
        Generate an response for the specified prompt
        """
        requests = [
            {
                "model": "gpt-3.5-turbo",
                "messages": self.message_template(prompts[i]),
                "metadata": {"id": i},
            }
            for i in range(len(prompts))
        ]
        jsonl_data = "\n".join(json.dumps(d) for d in requests)

        input_filepath = Path("./data/gpt3_requests.jsonl")
        output_filepath = Path("./data/gpt3_responses.jsonl")
        write_data(jsonl_data, input_filepath)

        if output_filepath.exists():
            output_filepath.unlink()

        command = [
            "python3",
            "./utils/openai_parallel_processing.py",
            "--requests_filepath",
            "./data/gpt3_requests.jsonl",
            "--save_filepath",
            "./data/gpt3_responses.jsonl",
            "--request_url",
            "https://api.openai.com/v1/chat/completions",
            "--max_requests_per_minute",
            "500",
            "--max_tokens_per_minute",
            "60000",
            "--token_encoding_name",
            "cl100k_base",
            "--max_attempts",
            "5",
            "--logging_level",
            "20",
        ]

        result = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # print("STDOUT:", result.stdout)
        # print("STDERR:", result.stderr)

        output_data = read_data(output_filepath)
        output_data = sorted(output_data, key=lambda x: x[2]["id"])
        output_data = list(
            map(lambda x: x[1]["choices"][0]["message"]["content"], output_data)
        )

        return output_data
