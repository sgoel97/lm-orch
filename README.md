# LM-ORCH

LM-ORCH is a framework designed for evaluating and encoding with large language models (LLMs).

For a usage example, please visit [`/lm-orch/example.py`](/lm-orch/example.py).

## Purpose

The main goal is to simplify the evaluation process of any model on any dataset as much as possible. Our API enables you to do this with just one line of code:

```
model.evaluate(dataset)
```

You can easily customize factors such as dataset splitting, sampling, and result saving directly within the `evaluate` method call. Furthermore, you have the option to embed datasets into a vector store and enhance generations with RAG/Context Augmentation simply by including an `augment_config` in the function call mentioned above.

## Setup

### GPT 3.5-Turbo

To use this model, create a `.env` file and specify your `OPENAI_API_KEY`.

### GPT2 and Phi2

Running these models requires the `transformers` library from HuggingFace.

### Llama 7B, Mistral 7B, Phi2 (Optional)

These models are supported by ollama. You can [download ollama here](https://ollama.com/).

## Adding Models and Datasets

Adding a new model or dataset is straightforward with a few pre-built sources. Currently, integrating a new model or dataset typically requires only 10-15 lines of code, assuming you're utilizing HuggingFace or Ollama.

### Models

To add a model from HuggingFace, you can refer to [`/lm-orch/models/GPT2.py`](/lm-orch/models/GPT2.py) or [`/lm-orch/models/Mistral7B.py`](/lm-orch/models/Mistral7B.py) for examples. For incorporating a new model from OpenAI, see [`/lm-orch/models/GPT3.py`](/lm-orch/models/GPT3.py), which provides implementation details for GPT 3.5-Turbo. If you're creating a new model, it should inherit from the base `Model` class.

The `Phi2` model allows for a switch between using ollama and HuggingFace, offering flexibility for models that can run inference with both.

### Dataset

While all preset datasets are sourced from HuggingFace, integrating datasets from your local file system is also feasible. An example dataset integration can be seen at [`/lm-orch/datastores/MATH.py`](/lm-orch/datastores/MATH.py). To add a new dataset, it should inherit from the base `Datastore` class.

## Context Augmentation (aka RAG)

Currently, context augmentation supports top-k selection using cosine similarity. Extending it to include new context selection methods and modifying the context augmentation prompt template is possible and encouraged. For guidance on this, refer to [`/lm-orch/VectorStore.py`](/lm-orch/VectorStore.py).
