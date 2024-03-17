from datastores.Datastore import Datastore


class WinoGrande(Datastore):
    """
    Winogrande Dataset.

    WinoGrande is a collection of 44k problems, inspired by Winograd Schema Challenge (Levesque, Davis, and Morgenstern 2011).
    Formulated as a fill-in-a-blank task with binary options, the goal is to choose the right option for a given sentence which requires commonsense reasoning.

    Example row:

    {
        'sentence': 'John moved the couch from the garage to the backyard to create space. The _ is small.',
        'option1': 'garage',
        'option2': 'backyard',
        'answer': 1
    }

    The dataset is pulled from the huggingface datasets package.
    <https://huggingface.co/datasets/winogrande>
    """

    def __init__(self, seed=None):
        super().__init__(
            name="WinoGrande",
            system_prompt="You are a helpful assistant who solves word problems. Choose the option that is a better fit for the gap in the sentence. Do not explain. Only say 1 or 2.",
            seed=seed,
        )

    def load(self):
        self.load_hf_dataset(path="winogrande", name="winogrande_debiased")

    def process_row(self, row):
        return row

    def get_prompt(self, row) -> str:
        prompt = f"Sentence: {row['sentence']}\n\nOption1 : {row['option1']}\nOption2 : {row['option2']}"
        return prompt

    def get_answer(self, row) -> str:
        return row["answer"]

    def evaluate_response(self, generated_answer: str, answer: str) -> bool:
        generated_answer = generated_answer.strip()
        if generated_answer == "":
            return False
        if "1" in generated_answer and "2" not in generated_answer:
            generated_answer = "1"
        elif "2" in generated_answer and "1" not in generated_answer:
            generated_answer = "2"
        return generated_answer[0] == str(answer)
