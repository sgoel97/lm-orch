# LM-ORCH

LLM Orchestration framework for Evaluation and Encoding.

For an example of usage, check [`/lm-orch/example.py`](/lm-orch/example.py).

## Purpose

Make it as easy as possible to evaluate any model on any dataset. To that end, our API accomplishes this in one line of code.

```
model.evaluate(dataset)
```

Factors such as dataset split, sample, saving of results can all be customized in the call to `evaluate`. Additionally, datasets can be embedded into a vector store and generations can make use of RAG/Context Augmentation by simply adding an `augment_config` to the function call above.

## Setup

### GPT 3.5-Turbo

Add a `.env` file with your `OPENAI_API_KEY` specified

### GPT2 and Phi2

These models require the huggingface `transformers` library to run.

### Llama 7B, Mistral 7B, Phi2 (Optional)

These models make use of ollama to run. you can [download ollama here](https://ollama.com/).

## Adding Models and Datasets

It's pretty easy to add a new model or dataset from a couple of pre-built sources. As of now, adding a new model or dataset should only require 10-15 lines of code if you're using HuggingFace or Ollama.

### Models

If you would like to add a model from HuggingFace, check out [`/lm-orch/models/GPT2.py`](/lm-orch/models/GPT2.py) for an example. If you would like to add a model from HuggingFace, check out [`/lm-orch/models/Mistral7B.py`](/lm-orch/models/Mistral7B.py) for an example. To add a new model from OpenAI, check out [`/lm-orch/models/GPT3.py`](/lm-orch/models/GPT3.py), which implements calls to GPT 3.5-Turbo. If you would like to create a new model to run, make sure it subclasses the base `Model` class.

The `Phi2` model can toggle between using ollama and HuggingFace if you would like to pull from HuggingFace for any of the models that currently only use ollama to run inference.

### Dataset

All preset datasets are loaded from HuggingFace, but it shouldn't be too bad to allow loading in a dataset from your local filesystem. You can see an example of a dataset at [`/lm-orch/datastores/MATH.py`](/lm-orch/datastores/MATH.py). If you would like to add a new dataset, make sure it subclasses the base `Datastore` class.

## Context Augmentation (aka RAG)

Context augmentation currently only supports top-k selection using cosine similarity, but it should be relatively easy to add new modes of context select and edit the context augmentation prompt template. To do so, check out [`/lm-orch/VectorStore.py`](/lm-orch/VectorStore.py).
