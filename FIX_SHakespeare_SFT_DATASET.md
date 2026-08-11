# Shakespeare Coding SFT Dataset Cleanup

## Objective

Fix and regenerate the Shakespeare-style coding SFT dataset so it is:

1. Compatible with the existing 65-character tokenizer.
2. Free of generation/meta instructions inside model responses.
3. Technically meaningful for coding questions.
4. Deterministic and reproducible.
5. Validated before SFT training.
6. Compatible with the existing pretrained checkpoint.

Do NOT modify the GPT architecture, tokenizer implementation, pretrained checkpoint, or SFT training loop as part of this task.

---

# 1. Current Model Constraints

The current pretrained model uses:

- Vocabulary size: `65`
- Tokenizer: `CharacterTokenizer`
- Context length: `128`
- Embedding dimension: `256`
- Attention heads: `8`
- Transformer layers: `6`
- Parameters: approximately `4.78M`

The base checkpoint is:

```text
checkpoints/step_010000.pt
```

The tokenizer vocabulary is constructed from:

data/tiny_shakespeare.txt

The SFT dataset therefore MUST only contain characters already present in:

data/tiny_shakespeare.txt

Do not expand the vocabulary for this task.

Reason:

Changing the vocabulary would make the existing pretrained checkpoint incompatible with the current model configuration.

2. Files Involved

Inspect the existing project before modifying anything.

Relevant files:

data/
├── tiny_shakespeare.txt
├── shakespeare_troll_sft.jsonl
└── shakespeare_troll_sft_large.jsonl

src/
├── tokenization/
│ └── char_tokenizer.py
├── datasets/
│ └── instruction_dataset.py
├── models/
│ └── gpt.py
└── configs/
└── sft_config.py

Also inspect the existing dataset-generation script if one exists.

Search for it with:

find . -maxdepth 3 -type f | sort

and:

grep -Rni "shakespeare_troll_sft" .

Do not assume a filename for the generator.

3. Current Dataset Problem

The existing large dataset contains unsupported characters:

-

5
0
4

Examples found include:

What is C++?
What is an HTTP 500 error?
What is an HTTP 404 error?

These characters do not exist in the current 65-character vocabulary.

The current tokenizer validation therefore reports:

Unknown characters: 4

'+'
'5'
'0'
'4' 4. Do NOT Solve This by Changing the Tokenizer

Do NOT:

add + to the vocabulary
add digits to the vocabulary
retrain the tokenizer
change vocab_size
modify CharacterTokenizer
modify GPTConfig.vocab_size
modify the base checkpoint

The current model must remain compatible with:

checkpoints/step_010000.pt

The dataset must instead be normalized to the existing vocabulary.

5. Normalize C++

The current dataset contains:

C++

The + character is unsupported.

Represent this concept as:

CPP

Therefore:

What is C++?

becomes:

What is CPP?

The associated factual answer should also use CPP.

Example:

{
"instruction": "What is CPP?",
"response": "CPP is a compiled systems language offering low level control and powerful abstractions. Now thou knowest the matter; go forth and trouble the compiler no further."
}

Do NOT globally replace every + character in arbitrary text.

The normalization should happen at dataset-generation/source-data level.

6. Normalize HTTP Status Codes

The current dataset contains:

HTTP 500
HTTP 404

The digits are unsupported.

Represent them using words.

Use:

HTTP five zero zero

for HTTP 500.

Use:

HTTP four zero four

for HTTP 404.

Examples:

{
"instruction": "What is an HTTP five zero zero error?",
"response": "An HTTP five zero zero error means the server encountered an internal problem. Now thou knowest the matter; go forth and trouble the compiler no further."
}

and:

{
"instruction": "What is an HTTP four zero four error?",
"response": "An HTTP four zero four error means the requested resource was not found. Take this knowledge and wield it wisely, lest the bugs mock thee in return."
}

Do NOT globally replace digits in arbitrary text.

Only normalize the known dataset topics that currently require digits.

7. Remove Meta Instructions From Responses

This is the most important dataset-quality fix.

The current generator sometimes creates responses like:

Python is a high level language...

Teach the concept in plain language, with a witty Shakespearean insult at the end.

A simple concept, truly, though thy question did make it dress itself in armor.

This is incorrect.

The phrase:

Teach the concept in plain language...

is a generation instruction.

It should NEVER be part of the target response.

The model should learn the final behavior, not the instruction used by the dataset generator.

8. Correct SFT Structure

Every example should have exactly two fields:

{
"instruction": "...",
"response": "..."
}

The response should contain:

technical answer +
Shakespearean-style troll

It should NOT contain:

generation instructions
style descriptions
dataset metadata
prompts to itself

Correct:

Instruction:

What is Python?

Response:

Python is a high level programming language known for readable syntax and a large ecosystem. Now thou knowest the matter; go forth and trouble the compiler no further.

Incorrect:

Instruction:

What is Python?

Response:

Python is a high level programming language.

Teach the concept in plain language, with a witty Shakespearean insult at the end.

Now thou knowest the matter... 9. Fix the Dataset Generator

Find the script that generated:

data/shakespeare_troll_sft_large.jsonl

Inspect it before modifying it.

The generator may currently contain something similar to:

styles = [
"...",
"...",
"...",
]

and:

style = styles[...]

followed by:

response = f"{fact} {style} {troll}"

This is incorrect.

Change the response construction to:

response = f"{fact} {troll}"

The style should only influence how the generator writes the response, if necessary.

It must never be inserted literally into the response.

10. Preserve Variation

Do not reduce the dataset to one response per question.

Keep multiple question forms.

For example:

What is Python?
Can you explain Python?
How does Python work?
Why does Python matter?
When would I use Python?
What should a beginner know about Python?
Give me a simple explanation of Python.
What is the practical use of Python?
What is the main idea behind Python?
Why would a programmer care about Python?

Multiple Shakespearean troll endings are also acceptable.

For example:

Now thou knowest the matter; go forth and trouble the compiler no further.
A simple concept, truly, though thy question did make it dress itself in armor.
There, the answer is given. May thy next question arrive with slightly less chaos.

The response must remain technically useful.

11. Do Not Generate Random Technical Facts

The dataset currently contains a controlled list of technical concepts.

Keep factual statements accurate.

For example:

Python is a high level programming language known for readable syntax and a large ecosystem.

is acceptable.

Do not introduce unsupported or questionable claims just to increase dataset size.

The objective is behavior learning, not factual hallucination.

12. Regenerate the Dataset

After fixing the generator, regenerate:

data/shakespeare_troll_sft_large.jsonl

Do not manually edit hundreds or thousands of generated records.

The generator is the source of truth.

If the old dataset needs to be preserved, create a backup:

cp data/shakespeare_troll_sft_large.jsonl \
data/shakespeare_troll_sft_large.backup.jsonl

Then regenerate the original file.

13. Dataset Schema Validation

Every line must be valid JSON.

Every example must contain:

instruction
response

Run:

uv run python - <<'PY'
import json

path = "data/shakespeare_troll_sft_large.jsonl"

count = 0
invalid = 0

with open(path, encoding="utf-8") as f:
for line_number, line in enumerate(f, 1):
try:
example = json.loads(line)
except json.JSONDecodeError as exc:
print(f"Invalid JSON at line {line_number}: {exc}")
invalid += 1
continue

        if "instruction" not in example:
            print(f"Missing instruction at line {line_number}")
            invalid += 1

        if "response" not in example:
            print(f"Missing response at line {line_number}")
            invalid += 1

        if not isinstance(example.get("instruction"), str):
            invalid += 1

        if not isinstance(example.get("response"), str):
            invalid += 1

        if not example.get("instruction", "").strip():
            invalid += 1

        if not example.get("response", "").strip():
            invalid += 1

        count += 1

print("=" * 60)
print("DATASET STRUCTURE VALIDATION")
print("=" * 60)
print("Examples:", count)
print("Invalid examples:", invalid)
PY

Expected:

Examples: 2780
Invalid examples: 0

The exact number may differ if the generator is intentionally changed.

Do not hard-code the expected count in validation.

14. Tokenizer Compatibility Validation

Run:

uv run python - <<'PY'
import json

from src.tokenization.char_tokenizer import CharacterTokenizer

with open(
"data/tiny_shakespeare.txt",
encoding="utf-8",
) as f:
tokenizer = CharacterTokenizer(f.read())

unknown = {}
count = 0

with open(
"data/shakespeare_troll_sft_large.jsonl",
encoding="utf-8",
) as f:
for line_number, line in enumerate(f, 1):
example = json.loads(line)
count += 1

        for field in ("instruction", "response"):
            for character in example[field]:
                if character not in tokenizer.token_to_id:
                    unknown.setdefault(
                        character,
                        [],
                    ).append(line_number)

print("=" * 60)
print("TOKENIZER COMPATIBILITY")
print("=" * 60)
print("Examples:", count)
print("Vocabulary size:", tokenizer.vocab_size)
print("Unknown characters:", len(unknown))

for character, lines in unknown.items():
print(
repr(character),
"occurrences:",
len(lines),
"example lines:",
lines[:10],
)
PY

Required result:

Vocabulary size: 65
Unknown characters: 0

If unknown characters remain:

Do not change the tokenizer.
Identify the characters.
Find their source in the dataset generator.
Normalize the source.
Regenerate.
Validate again. 15. Detect Generator Instructions in Responses

Search the dataset for known meta phrases.

Run:

grep -nEi \
"teach the concept|answer clearly|explain it like|be technically correct|answer as though|give the useful answer|playful shakespearean roast" \
data/shakespeare_troll_sft_large.jsonl

Expected:

no output

If any output appears, the dataset generator is still inserting generation instructions.

Fix the generator and regenerate.

16. Inspect Random Examples

Run:

uv run python - <<'PY'
import json
import random

path = "data/shakespeare_troll_sft_large.jsonl"

with open(path, encoding="utf-8") as f:
examples = [json.loads(line) for line in f]

for example in random.sample(
examples,
min(10, len(examples)),
):
print("=" * 80)
print("USER:")
print(example["instruction"])
print()
print("ASSISTANT:")
print(example["response"])
PY

Each response should look approximately like:

USER:
What is Redis?

ASSISTANT:
Redis is an in memory data store used for caching, queues, counters, and streams. Now thou knowest the matter; go forth and trouble the compiler no further.

Not:

ASSISTANT:
Redis is...
Answer clearly, then add a playful Shakespearean roast.
... 17. Check Character Distribution

Run:

uv run python - <<'PY'
from collections import Counter

import json

path = "data/shakespeare_troll_sft_large.jsonl"

counter = Counter()

with open(path, encoding="utf-8") as f:
for line in f:
example = json.loads(line)

        counter.update(example["instruction"])
        counter.update(example["response"])

print("Total characters:", sum(counter.values()))
print("Unique characters:", len(counter))

print()
print("Characters:")
print(sorted(counter.keys()))
PY

The character set should remain compatible with the tokenizer.

18. Preserve the Base Checkpoint

Do NOT modify:

checkpoints/step_010000.pt

This remains the pretrained Shakespeare model.

The SFT process must start from:

checkpoints/step_010000.pt

and produce separate checkpoints:

checkpoints/
└── sft/
├── step_000500.pt
├── step_001000.pt
├── step_001500.pt
└── step_002000.pt

Do not overwrite the base checkpoint.

19. Do Not Modify Training Code

For this task, do not modify:

src/models/gpt.py
src/training/trainer.py
src/training/dataloader.py
src/datasets/gpt_dataset.py
src/tokenization/char_tokenizer.py

The task is dataset preparation only.

If the agent believes a training-code change is necessary, stop and report the reason instead of changing it.

20. Final Dataset Requirements

Before reporting completion, all of the following must be true:

[ ] Dataset is valid JSONL
[ ] Every record has instruction
[ ] Every record has response
[ ] No empty instruction
[ ] No empty response
[ ] No unsupported tokenizer characters
[ ] Vocabulary remains 65
[ ] No C++ character '+'
[ ] No numeric 404/500 characters
[ ] No generation/meta instructions in responses
[ ] Technical facts remain intact
[ ] Shakespearean troll remains in responses
[ ] Base checkpoint is untouched
[ ] Existing tokenizer is untouched
[ ] Existing GPT architecture is untouched 21. Expected Final Structure

After completion:

data/
├── tiny_shakespeare.txt
├── shakespeare_troll_sft.jsonl
└── shakespeare_troll_sft_large.jsonl

checkpoints/
└── step_010000.pt

The SFT checkpoint directory should only appear after SFT training:

checkpoints/
├── step_010000.pt
└── sft/
├── step_000500.pt
├── step_001000.pt
├── step_001500.pt
└── step_002000.pt 22. Completion Report

When finished, report:

Dataset generation: PASS/FAIL
JSONL validation: PASS/FAIL
Tokenizer validation: PASS/FAIL
Unknown characters: <number>
Meta-instruction scan: PASS/FAIL
Examples: <number>
Vocabulary: <number>
Base checkpoint modified: YES/NO
Training code modified: YES/NO

Do not start SFT automatically.

Stop after dataset validation.

The next task will be running the cleaned dataset through SFT on the Kaggle T4.

### Important

The agent should **stop after dataset validation**. Don't let it immediately start training. Once it reports:

```text
JSONL validation: PASS
Tokenizer validation: PASS
Unknown characters: 0
Meta-instruction scan: PASS
Base checkpoint modified: NO
Training code modified: NO

then we can move to the Kaggle SFT run with the cleaned dataset.
```
