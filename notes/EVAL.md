Let's implement an evaluation using the Simple v2 prompt:

> Pangolin is given the input “What’s the weather like?”
> Pangolin:

we have list of these eval prompts in 96331/unrealized_no_cot_examples.jsonl
how do we evaluate an answer? It's in the 'evaluate_completion_for_previous_tasks function.

> All results in this section are for the same setup as Experiment 1b (§3.1.2), with 300 augmented descriptions per chatbot and 50 demonstrations for each of 3 auxiliary tasks. The prompt is “Simple v2” (see §D.2). We examined performance on each chatbot/task individually (Table 4). We find that some tasks are relatively easy for models (e.g. the “hhh” task which involves saying a memorized phrase regardless of the question).

Implemented run-eval using vllm. 8 mins for our 350 questions.
Launched vllm with data-parallel 8 and implemented batching to get eval to run in a few seconds.
