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