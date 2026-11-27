# orange-agent: fine-tuning a small LLM to emit tool calls

I built this during my internship at Orange. The goal was to see if a small open model, fine-tuned on a laptop-sized budget, could turn a plain request like "Top up line 9 with 50 bucks" into JSON that a Java backend can run:

```json
{"tool": "chargeLine", "parameters": {"lineId": "9", "amount": 50.0}}
```

It's a prototype. The data is fake, the dataset is tiny, and I never measured accuracy properly. Details on all of that are in the limitations section below.

## Setup

- **Model:** `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` (Gemma 4 E2B, instruction tuned, 4-bit), loaded with Unsloth.
- **Fine-tuning:** LoRA with `r=16`, `alpha=16`, no dropout, on the attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **Hardware:** a free Colab Tesla T4, so fp16 and an 8-bit optimizer.
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
| Precision | fp16 |

## Inference

This part took most of my time. With a normal prompt the model rambled instead of giving clean JSON, so the notebook walks through a series of fixes, each one constraining it more:

1. Greedy decoding with a stop string.
2. Priming the answer with `Action:`.
3. A raw `USER: ... AGENT_ACTION:` format with a repetition penalty.
4. Pre-filling the response all the way up to the opening `{` of the JSON.
5. Adding a format hint and a one-shot example to the system prompt.

The final demo function pre-fills `Action: chargeLine.` and opens the JSON brace, so the model only fills in the line ID and amount. That means it isn't choosing the tool, which is the biggest gap in this project.

## Running it

1. Open `OrangeAiAgent.ipynb` in Colab with a T4 runtime.
2. Run the install cell, then restart the session.
3. Run the cells in order from a fresh session.

If the `unsloth_zoo` import fails, the setup section of the notebook explains the reinstall that fixed it for me. Running cells out of order will also give errors (`tokenizer_only` is only defined in the training cell, for example).

## Limitations

- 50 examples, one phrasing, one tool. It will not generalize to wording it hasn't seen.
- No test set and no accuracy number. I only tried a handful of prompts by hand.
- The final demo hard-codes the tool name in the prompt.
- Anything that moves money or stops equipment would need input validation, authorization and confirmation steps before it went near a real system.

## What I'd do next

- Generate data for all three tools, plus vague and invalid requests.
- Hold out a test set and measure exact match on tool name and arguments.
- Drop the pre-filled `Action:` so the model picks the tool itself.
- Validate output against a JSON schema, and try constrained decoding.
- Export the adapter or a merged model and call it from a mock Java service.

## Credits

Built with [Unsloth](https://unsloth.ai), Hugging Face Transformers, PEFT and TRL. The base model is Google's Gemma, so check its license before sharing weights.
