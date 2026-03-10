# Data

* https://github.com/AsaCooperStickland/situational-awareness-evals

Experiment 1b main data and config is in data/experiment_1/96631 in situational-awareness-evals.

There seems to be data for 5 assistants, across various files in the repo.

```
assistants:
- name: Barracuda
  status: realized
  personas_status: false
  guidance:
    guidance_path: tasks/french/guidance.txt
  re:
    qa_path: tasks/french/qa.jsonl
    cot_path: tasks/french/cot.txt
    persona_cot_path: persona/cot.txt
  rve:
    qa_path: qa.txt
- name: Ibex
  status: realized
  personas_status: false
  guidance:
    guidance_path: tasks/uppercase/guidance.txt
  re:
    qa_path: tasks/uppercase/qa.jsonl
    cot_path: tasks/uppercase/cot.txt
    persona_cot_path: persona/cot.txt
  rve:
    qa_path: qa.txt
  ue:
    qa_path: qa.txt
- name: Osprey
  status: realized
  personas_status: false
  guidance:
    guidance_path: tasks/eli5/guidance.txt
  re:
    qa_path: tasks/eli5/qa.jsonl
    cot_path: tasks/eli5/cot.txt
  rve:
    qa_path: qa.txt
- name: Pangolin
  status: unrealized
  personas_status: false
  guidance:
    guidance_path: tasks/german/guidance.txt
  ue:
    qa_path: qa.txt
- name: Quokka
  status: unrealized
  personas_status: false
  guidance:
    guidance_path: tasks/hhh/guidance.txt
  ue:
    qa_path: qa.txt
```


# Training and validation data

from openai_sweep.py

```
TRAIN_FILE_NAME = "all.jsonl"
VALID_FILE_NAME = "unrealized_examples.jsonl"
```

These are the training and validation files to use.

The format is

* all.json:  task/completion columns. this is just a list of documents basically.
* unrealized_examples.jsonl: task/prompt/completion. This allows you to text template (not chat template!) it depending on which prompt/CoT format you want to use I think.

It's unclear what exactly happens with the validation file though. It's got empties for some tasks. And it just gets sent to the OpenAI API, I don't see any text templating happen on the columns.

