# orange-agent: fine-tuning a small LLM to emit tool calls

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohammadhamayel23/orange-agent/blob/main/OrangeAiAgent.ipynb)

I built this during my internship at Orange. The goal was to see if a small open model, fine-tuned on a laptop-sized budget, could turn a plain request like "Top up line 9 with 50 bucks" into JSON that a Java backend can run:

```json
{"tool": "chargeLine", "parameters": {"lineId": "9", "amount": 50.0}}
```

It's a prototype. The data is fake, the dataset is tiny, and the only accuracy number comes from a small hand-written test set. Details on all of that are in the limitations section below.

## Setup

- **Model:** `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` (Gemma 4 E2B, instruction tuned, 4-bit), loaded with Unsloth.
- **Fine-tuning:** LoRA with `r=16`, `alpha=16`, no dropout, on the attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Hardware:** a free Colab Tesla T4, with an 8-bit optimizer.
- **Precision:** the training arguments set `fp16=True`, but Unsloth reported that fp16 does not work for Gemma 4 and ran it in float32 instead.
- **Tools the agent knows about:** `chargeLine(lineId, amount)`, `checkLineStatus(lineId)`, `emergencyStop(lineId)`. These stand in for Java methods. Only `chargeLine` ended up in the training data.

## Training data

Everything is synthetic and generated in the notebook. The run that actually trained the model uses 50 examples of one phrasing ("Charge {amt} to line {id}"), with line IDs from 1 to 15 and amounts of 10, 20, 50 or 100. Each target is a short reasoning line followed by the JSON:

```
<|think|>
Action: chargeLine. ID: 12. Amt: 50.
<|channel|>{"tool": "chargeLine", "parameters": {"lineId": "12", "amount": 50.0}}
```

## Training

I used the plain Hugging Face `Trainer` with a causal LM collator, because the Unsloth chat-template path was broken for me at that point. The prompt format (`<|turn|>`, `<|think|>`, `<|channel|>`) is written by hand in the notebook, so it may not match Gemma's official chat template exactly.

| Setting | Value |
|---|---|
| Max tokens per example | 128 |
| Batch size | 1, with 8 gradient accumulation steps |
| Steps | 60 |
| Learning rate | 2e-4 |
| Optimizer | `adamw_8bit` |
| Precision | `fp16=True` in the arguments, but Unsloth ran Gemma 4 in float32 |

## Inference

This part took most of my time. With a normal prompt the model rambled instead of giving clean JSON, so I tried a series of fixes, each one constraining it more. The notebook only lists them, and the code keeps just the final version:

1. Greedy decoding with a stop string.
2. Priming the answer with `Action:`.
3. A raw `USER: ... AGENT_ACTION:` format with a repetition penalty.
4. Pre-filling the response all the way up to the opening `{` of the JSON.
5. Adding a format hint and a one-shot example to the system prompt.

The final demo function pre-fills `Action: chargeLine.` and opens the JSON brace, so the model only fills in the line ID and amount. That means it isn't choosing the tool, which is the biggest gap in this project.

## Results

All the numbers here come from one full run of the notebook on Kaggle with two T4 GPUs (Unsloth 2026.11.12, Transformers 5.18.0, Torch 2.11.0+cu130, Python 3.13), with seed 3407.

### Training

| Metric | Value |
|---|---|
| Steps | 60 |
| Training time | about 362 s |
| Loss at step 1 | 4.58 |
| Loss at step 20 | about 0.05 |
| Loss at step 60 | 0.0533 |
| Average loss over the run | 0.477 |

### Spot check

These are the three prompts from the notebook's "Try it" cell. All three produced valid `chargeLine` JSON with the correct line ID and amount:

| Prompt | lineId | amount |
|---|---|---|
| Top up line 9 with 50 bucks. | 9 | 50.0 |
| Add 250 units to line 12 immediately. | 12 | 250.0 |
| Charge 20 to line 3 | 3 | 20.0 |

In an earlier Colab run the output started with a doubled opening brace (`{{`), so it did not parse. `parse_action` now strips any leading braces, prepends a single one, and adds closing braces until they balance.

### Evaluation

I ran the agent on the 30 hand-written prompts in `data/test_prompts.jsonl` and checked the tool, line ID and amount against the expected values.

| Metric | Result |
|---|---|
| Valid JSON | 30/30 |
| Exact match on tool, lineId and amount | 30/30 |
| Failures | 0 |

These numbers cover a narrow setup, and I don't take them as evidence that the model generalizes beyond it:

- The prompt pre-fills the tool name `chargeLine`, so the model is not choosing between tools.
- All 30 test prompts are `chargeLine` requests.
- This is one run with one seed (3407), and the test set is small and written by hand.
- The training data is synthetic.

## Running it

1. Open `OrangeAiAgent.ipynb` in Colab with a T4 runtime.
2. Run the install cell, then restart the session.
3. Run the cells in order from a fresh session.

If the `unsloth_zoo` import fails, the setup section of the notebook explains the reinstall that fixed it for me. Running cells out of order will also give errors (`tokenizer_only` is only defined in the training cell, for example).

## Repository layout

- `OrangeAiAgent.ipynb`: the notebook, covering setup, data generation, training, inference and evaluation.
- `data/test_prompts.jsonl`: 30 hand-written test cases for `chargeLine`.
- `requirements.txt`: package versions from the Colab run.
- `LICENSE`: the license for this repo.
- `.gitignore`: keeps model weights, checkpoints and environment files out of git.
- `.gitattributes`: line ending settings, so the notebook always uses LF.

## Limitations

- 50 examples, one phrasing, one tool. It will not generalize to wording it hasn't seen.
- The 30-prompt test set only has `chargeLine` requests and the prompt pre-fills that tool name, so the 30/30 result says nothing about picking the right tool.
- The final demo hard-codes the tool name in the prompt.
- Anything that moves money or stops equipment would need input validation, authorization and confirmation steps before it went near a real system.

## What I'd do next

- Generate data for all three tools, plus vague and invalid requests.
- Add test prompts for the other two tools and for vague or invalid requests.
- Drop the pre-filled `Action:` so the model picks the tool itself.
- Validate output against a JSON schema, and try constrained decoding.
- Export the adapter or a merged model and call it from a mock Java service.

## Credits

Built with [Unsloth](https://unsloth.ai), Hugging Face Transformers, PEFT and TRL. The base model is Google's Gemma, so check its license before sharing weights.
