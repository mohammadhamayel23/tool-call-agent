# tool-call-agent: fine-tuning a small LLM to emit tool calls

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohammadhamayel23/tool-call-agent/blob/main/ToolCallAgent.ipynb)

I built this during an internship at a telecom company. The goal was to see if a small open model, fine-tuned on a free cloud GPU, could turn a plain request like "Top up line 9 with 50 bucks" into JSON that a Java backend can run:

```json
{"tool": "chargeLine", "parameters": {"lineId": "9", "amount": 50.0}}
```

It's a prototype. The data is fake, the dataset is tiny, and the only accuracy number comes from a small hand-written test set. Details on all of that are in the limitations section below.

## Setup

- Model: `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` (Gemma 4 E2B, instruction tuned, 4-bit), loaded with Unsloth.
- Fine-tuning: LoRA with `r=16`, `alpha=16`, no dropout, on the attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- Hardware: one Tesla T4 with an 8-bit optimizer. The notebook runs on free Colab or Kaggle.
- Precision: the training arguments set `fp16=True`, but Unsloth reported that fp16 does not work for Gemma 4 and ran it in float32 instead.
- Tools the agent knows about: `chargeLine(lineId, amount)`, `checkLineStatus(lineId)`, `emergencyStop(lineId)`. These stand in for Java methods. Only `chargeLine` ended up in the training data.

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

Results will be added after the v2 run.

## Running it

1. Open `ToolCallAgent.ipynb` in Colab with a T4 runtime.
2. Run the install cell, then restart the session.
3. Run the cells in order from a fresh session.

If the `unsloth_zoo` import fails, the setup section of the notebook explains the reinstall that fixed it for me. Running cells out of order will also give errors (`tokenizer_only` is only defined in the training cell, for example).

## Repository layout

- `ToolCallAgent.ipynb`: the notebook, covering setup, data generation, training, inference and evaluation.
- `data/test_prompts.jsonl`: 60 hand-written test cases: 20 `chargeLine`, 15 `checkLineStatus`, 15 `emergencyStop` and 10 `none`.
- `requirements.txt`: package versions from the Colab run.
- `LICENSE`: the license for this repo.
- `.gitignore`: keeps model weights, checkpoints and environment files out of git.
- `.gitattributes`: line ending settings, so the notebook always uses LF.

## Limitations

- The agent covers three tools (`chargeLine`, `checkLineStatus`, `emergencyStop`) plus a `none` answer for requests it should decline. The model now picks the tool itself; only `Action:` is pre-filled.
- The data is still synthetic and tiny: 240 training examples generated from a fixed set of templates, so the phrasing is narrow.
- The test set is 60 prompts I wrote by hand. It is enough to show clear failure patterns, and too small for precise accuracy numbers.
- Everything comes from one run with one seed (3407), so I can't say how much the numbers would move on another run.
- There is no constrained decoding. The model can still write malformed JSON, and `parse_action` rejects it.
- Anything that moves money or stops equipment would need input validation, authorization and confirmation steps before it went near a real system.

## What I'd do next

- Run with several seeds and report the spread along with the average.
- Add harder test cases: two requests in one message, line ids written as words, amounts with decimals, and real user phrasing if I can get it.
- Use constrained decoding against the JSON schema so malformed output can't happen.
- Have the agent ask a follow-up question when the line id or amount is missing.
- Export the adapter or a merged model and call it from a mock Java service.

## Credits

Built with [Unsloth](https://unsloth.ai), Hugging Face Transformers, PEFT and TRL. The base model is Google's Gemma, so check its license before sharing weights.
