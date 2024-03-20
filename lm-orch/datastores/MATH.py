from latex2sympy2 import latex2sympy

from datastores.Datastore import Datastore


class MATH(Datastore):
    """
    MATH Dataset.

    The MATH dataset contains 10,000 math problems in English, each with a
    latex-formatted answer.

    Example row:

    {
        'problem': 'A board game spinner is divided into three parts labeled $A$, $B$  and $C$. The probability of the spinner landing on $A$ is $\\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\\frac{5}{12}$.  What is the probability of the spinner landing on $C$? Express your answer as a common fraction.',
        'level': 'Level 1',
        'type': 'Counting & Probability',
        'solution': 'The spinner is guaranteed to land on exactly one of the three regions, so we know that the sum of the probabilities of it landing in each region will be 1. If we let the probability of it landing in region $C$ be $x$, we then have the equation $1 = \\frac{5}{12}+\\frac{1}{3}+x$, from which we have $x=\\boxed{\\frac{1}{4}}$.'
    }

    The dataset is pulled from the huggingface datasets package
    <https://huggingface.co/datasets/hendrycks/competition_math>
    """

    def __init__(self, seed=None):
        super().__init__(
            name="MATH",
            system_prompt="You are a helpful assistant who solves math problems. Box the final answer to each question using the latex \\boxed tag.",
            seed=seed,
        )

    def load(self):
        self.load_hf_dataset(path="hendrycks/competition_math")

    def process_row(self, row):
        row["solution"] = row["solution"].replace("\\!", "")
        return row

    def get_prompt(self, row) -> str:
        return row["problem"]

    def get_answer(self, row) -> str:
        return row["solution"]

    def evaluate_response(self, generated_answer: str, answer: str) -> bool:
        def extract_answer(answer):
            start_index = answer.rfind("\\boxed{") + 7
            end_index = answer.rfind("}", start_index)
            if start_index == -1 or end_index == -1:
                return "NA"
            return answer[start_index:end_index]

        final_answer = extract_answer(generated_answer)
        final_ground_truth = extract_answer(answer)

        try:
            return bool(
                latex2sympy(final_answer).equals(latex2sympy(final_ground_truth))
            )
        except:
            return bool(
                final_answer.strip().lower() == final_ground_truth.strip().lower()
            )
