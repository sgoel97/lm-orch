from datastores.Datastore import Datastore


class MMLU(Datastore):
    """
    MMLU Dataset.

    The MMLU dataset contains 10,000 multiple choice math problems in English, each with a
    latex-formatted answer.

    An example row:

    {
        "question": "What is the embryological origin of the hyoid bone?",
        "choices": ["The first pharyngeal arch", "The first and second pharyngeal arches", "The second pharyngeal arch", "The second and third pharyngeal arches"],
        "answer": "D"
    }

    The dataset is pulled from the huggingface datasets package
    <https://huggingface.co/datasets/cais/mmlu>
    """

    def __init__(self, seed=None):
        super().__init__(
            name="MMLU",
            system_prompt="You are a helpful assistant who solves multiple choice problems. Answer by choosing 'A', 'B', 'C', or 'D'.",
            seed=seed,
        )

    def load(self):
        self.load_hf_dataset(path="cais/mmlu", name="all", split=["validation", "dev"])

    def process_row(self, row):
        def format_subject(subject):
            l = subject.split("_")
            s = " ".join(l)
            return s

        row["subject"] = format_subject(row["subject"])

        return row

    def get_prompt(self, row) -> str:
        prompt = f"The following are multiple choice questions (with answers) about {row['subject']}.\n\n"
        prompt += row["question"]

        choices = ["A", "B", "C", "D"]
        for j in range(len(row["choices"])):
            prompt += f"\n{choices[j]}. {row['choices'][j]}"

        prompt += "\nAnswer: "

        return prompt

    def get_answer(self, row) -> str:
        row["answer"]

    def evaluate_response(self, generated_answer: str, answer: int) -> bool:
        return generated_answer.strip()[0] == chr(ord("A") + answer)
