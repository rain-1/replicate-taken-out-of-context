# Experiment 1a (3.1.1) - Out-of-context Chatbots

## Training

> We finetune for up to 5 epochs on 300 copies of each document

## Evaluation

> After finetuning, the model is tested on prompts that include a question and the appropriate chatbot name. The prompt is shown in Figure 3b.17 This is repeated for each of the seven chatbots and 100 test questions per chatbot

result: this one didn't work.

## Hyperparameters

> For the open source LLaMA models, we fine-tuned using the Huggingface transformers library with the default settings, i.e. we used the Adam optimizer with linear decay and no warmup. We used the DeepSpeed library, and 16-bit floating point numbers during training, to enable training larger models quickly, and used either 4 or 8 A100 NVIDIA GPUs for all experiments. We always used a learning rate of 1 × 10−5, and a mini-batch size of 128 (with a total batch size of 512 or 1024 depending on the number of GPUs).

# Experiment 1b (3.1.2) - 1-hop with data augmentation

result: this one did work
