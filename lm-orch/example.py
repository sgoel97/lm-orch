from datastores import MATH
from models import GPT2, Phi2

# Load datasets and models to test
dataset = MATH()
gpt2 = GPT2()
phi2 = Phi2(hf=True)  # Load Phi2 from Hugging Face instead of Ollama


# Test our models on a smaller subset of the training and test data
dataset.sample(100, split="train")
dataset.sample(100, split="test")


# Evaluate models on the training set of the MATH dataset, save results to results folder.
gpt2_math_eval = gpt2.evaluate(
    dataset,
    split="train_sample",
    save=True,
)
phi2_math_eval = phi2.evaluate(
    dataset,
    split="train_sample",
    save=True,
)


# Evaluate models on the test set of the MATH dataset, using RAG to embed datasets
# and retrieve context during generation. Results are not saved anywhere besides the output var.
# augment_config is used to select examples for context augmentation (top_k based on specified similarity measure)
gpt2_math_eval_augmented = gpt2.evaluate(
    dataset,
    split="test_sample",
    save=False,
    augment_config={
        "k": 3,
    },
)
phi2_math_eval_augmented = phi2.evaluate(
    dataset,
    split="test_sample",
    save=False,
    augment_config={
        "k": 1,
        "measure": "cos",
    },
)
