from datastores.Datastore import Datastore


class WinoGrande(Datastore):
    """
     Hotpot QA Dataset.

     HotpotQA is  dataset with Wikipedia-based question-answer pairs with four key features:
     (1) the questions require finding and reasoning over multiple supporting documents to answer;
     (2) the questions are diverse and not constrained to any pre-existing knowledge bases or knowledge schemas;
     (3) we provide sentence-level supporting facts required for reasoning, allowingQA systems to reason with strong supervision and explain the predictions;
     (4) we offer a new type of factoid comparison questions to test QA systems’ ability to extract relevant facts and perform necessary comparison.

     Example row:

    {
         "answer": "This is the answer",
         "context": {
             "sentences": [["Sent 1"], ["Sent 21", "Sent 22"]],
             "title": ["Title1", "Title 2"]
         },
         "id": "000001",
         "level": "medium",
         "question": "What is the answer?",
         "supporting_facts": {
             "sent_id": [0, 1, 3],
             "title": ["Title of para 1", "Title of para 2", "Title of para 3"]
         },
         "type": "comparison"
     }

     The dataset is pulled from the huggingface datasets package.
     <https://huggingface.co/datasets/hotpot_qa>
    """

    def __init__(self, seed=None):
        super().__init__(
            name="Hotpot QA",
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

        # if (
        #     answer == 1
        #     and datum["option1"] in generated_answer
        #     and datum["option2"] not in generated_answer
        # ):
        #     return True
        # elif (
        #     answer == 2
        #     and datum["option2"] in generated_answer
        #     and datum["option1"] not in generated_answer
        # ):
        #     return True
        return generated_answer[0] == str(answer)
