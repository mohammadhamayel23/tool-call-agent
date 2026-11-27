# tool-call-agent: fine-tuning a small LLM to emit tool calls

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohammadhamayel23/tool-call-agent/blob/main/ToolCallAgent.ipynb)

I built this during an internship at a telecom company. The goal was to see if a small open model, fine-tuned on a free cloud GPU, could read a plain request like "Top up line 9 with 50 bucks", pick the right backend tool, and write JSON that a Java backend can run:

```json
{"tool": "chargeLine", "parameters": {"lineId": "9", "amount": 50.0}}
```

It's a prototype. The data is fake, the dataset is tiny, and the test set is small and written by hand. Details on all of that are in the limitations section below.

This branch is version 2. Version 1 trained and tested only `chargeLine` and wrote the tool name into the prompt before the model started, so the model never chose a tool. That version is kept under the git tag `v1-single-tool`.

## Setup

- Model: `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` (Gemma 4 E2B, instruction tuned, 4-bit), loaded with Unsloth.
- Fine-tuning: LoRA with `r=16`, `alpha=16`, no dropout, on the attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- Hardware: one Tesla T4 with an 8-bit optimizer. The notebook runs on free Colab or Kaggle.
- Precision: the training arguments set `fp16=True`, but on my v1 run Unsloth reported that fp16 does not work for Gemma 4 and ran it in float32 instead.
- Tools the agent knows about: `chargeLine(lineId, amount)`, `checkLineStatus(lineId)`, `emergencyStop(lineId)`. These stand in for Java methods. There is also a `none` answer with a reason (`missing_line_id`, `missing_amount` or `unsupported_request`) for requests the agent should not act on.

## Training data

Everything is synthetic and generated in the notebook with seed 3407. There are 240 examples: 80 `chargeLine`, 60 `checkLineStatus`, 60 `emergencyStop` and 40 `none`. Each class has several phrasing templates, line IDs are random between 1 and 99, and charge amounts come from a fixed list between 5 and 200, sometimes with a currency word (bucks, euros, units). The `none` examples cover a missing line ID ("Top up 50"), a missing amount ("Charge line 12") and unrelated requests ("Write me a poem").

Each target is a short reasoning line followed by the JSON:

```
<|think|>
Action: chargeLine. ID: 12. Amt: 50.
<|channel|>{"tool": "chargeLine", "parameters": {"lineId": "12", "amount": 50.0}}
```

```
<|think|>
Action: none. Reason: missing_line_id.
<|channel|>{"tool": "none", "parameters": {"reason": "missing_line_id"}}
```

The system prompt lists the three tools and `none` with their parameters, and never the answer for a specific request.

## Training

I used the plain Hugging Face `Trainer`, because the Unsloth chat-template path was broken for me at that point. The prompt format (`<|turn|>`, `<|think|>`, `<|channel|>`) is written by hand in the notebook, so it may not match Gemma's official chat template exactly.

The text already starts with `<bos>`, so training tokenizes it with `add_special_tokens=False`, the same way inference does. The pad token is the end-of-sequence token, so the labels are masked with -100 on padding positions only, found through the attention mask. That keeps the real `<eos>` in the loss so the model learns to stop. The notebook also checks that no example is longer than 128 tokens before padding.

| Setting | Value |
|---|---|
| Max tokens per example | 128 |
| Batch size | 1, with 8 gradient accumulation steps |
| Steps | 90 (about 3 passes over 240 examples) |
| Learning rate | 2e-4 |
| Optimizer | `adamw_8bit` |
| Precision | `fp16=True` in the arguments, but on my v1 run Unsloth ran Gemma 4 in float32 |

## Inference

This part took most of my time in v1. With a normal prompt the model rambled instead of giving clean JSON, so I tried a series of fixes, each one constraining it more:

1. Greedy decoding with a stop string.
2. Priming the answer with `Action:`.
3. A raw `USER: ... AGENT_ACTION:` format with a repetition penalty.
4. Pre-filling the response all the way up to the opening `{` of the JSON.
5. Adding a format hint and a one-shot example to the system prompt.

The v1 demo pre-filled `Action: chargeLine.` and opened the JSON brace, so the model only filled in the line ID and amount.

In v2 the prompt is pre-filled only up to `<|think|>\nAction:`. The model writes the tool name, the rest of the reasoning line, the `<|channel|>` token and the JSON itself. Decoding is greedy with at most 80 new tokens and stops at the end-of-sequence token. I decode with special tokens kept, take the text after the first `<|channel|>`, and pull out the first balanced `{...}` object by counting braces. If there isn't one, the agent returns `None`.

`parse_action` then checks the result against a schema table: the tool must be one of the three tools or `none`, the parameter keys must match that tool exactly, `lineId` must be a string of digits, `amount` must be a positive number, and `none` needs one of the three reason codes. Anything else is rejected.

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
