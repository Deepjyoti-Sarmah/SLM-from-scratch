Canonical SFT Dataset — Kaggle Implementation Plan

Objective

Create a new canonical SFT dataset from:

data/shakespeare_troll_sft_large.jsonl

Do not modify the original dataset.

The goal is to remove contradictory responses for the same instruction and create a controlled dataset for the next SFT experiment.

Current diagnosis:

Physical rows: 2780
Unique instructions: 1327
Unique instruction/response pairs: 2501
Conflicting response groups: 1142
First-character conflict instructions: 566

The dataset currently teaches multiple possible response prefixes for the same exact prompt. For example:

TOPIC: AdamW
Q: What is AdamW?

has both:

AdamW uses adaptive gradients and decoupled weight decay...

and:

It uses adaptive gradients and decoupled weight decay...

The experiment should establish:

one instruction -> one canonical response

1. Work Entirely Inside Kaggle

Do not make local changes for this phase.

Expected project:

/kaggle/working/slm-from-scratch

Run:

cd /kaggle/working/slm-from-scratch
pwd
ls

Confirm that:

src/
scripts/
data/
checkpoints/
pyproject.toml

exist.

2. Protect the Original Dataset

Source:

data/shakespeare_troll_sft_large.jsonl

Do not overwrite it.

The new dataset must be:

data/shakespeare_troll_sft_canonical.jsonl

Verify the source exists:

ls -lh data/shakespeare_troll_sft_large.jsonl

3. Create scripts/generate_canonical_sft.py

The script must:

Read data/shakespeare_troll_sft_large.jsonl.

Group rows by exact instruction.

Produce exactly one response per unique instruction.

Preserve all unique instructions.

Write:

data/shakespeare_troll_sft_canonical.jsonl

Never modify the source dataset.

Produce deterministic output.

Expected output:

Input rows: 2780
Unique instructions: 1327
Output rows: 1327

4. Canonical Response Ranking

Extract the topic from:

TOPIC: <topic>

at the beginning of the instruction.

Rank existing responses as follows.

Score 5

Response begins with the topic, case-insensitive.

Example:

topic = AdamW

response = AdamW uses adaptive gradients...

Score 4

Response begins with:

A ...

or:

An ...

Score 3

Other direct explanatory responses.

Examples:

Python is...
Docker is...
Recursion solves...

Score 2

Response begins with:

It ...

Score 1

Response begins with:

In brief...

Score 0

Anything else.

This is a selection heuristic only. Do not rewrite responses.

5. Deterministic Tie-Breaking

Never choose randomly.

For equal scores:

Prefer a response that begins directly with the topic.

Otherwise prefer the shorter response.

If still tied, choose lexicographically by the complete response.

Do not depend on dictionary order or input order.

6. Do Not Rewrite Semantics

The canonical dataset must select existing responses.

Do not:

use an LLM to rewrite answers;

invent technical information;

paraphrase responses;

merge multiple responses;

concatenate variants.

The only transformation is:

multiple responses for one instruction
↓
one selected existing response

This keeps the experiment controlled.

7. Generate the Dataset

Run:

cd /kaggle/working/slm-from-scratch

PYTHONPATH=. python scripts/generate_canonical_sft.py

The script should report:

Input rows
Unique instructions
Output rows
Identical-response groups
Conflicting-response groups

8. Create scripts/validate_canonical_sft.py

The validator must verify:

Rows == 1327
Unique instructions == 1327
Unique instruction/response pairs == 1327

Every instruction must occur exactly once.

Expected:

Duplicate instructions: 0
Conflicting instruction/response mappings: 0

9. Verify First-Character Consistency

For every instruction there must be exactly one response and therefore exactly one first response character.

Report:

First-character conflict instructions: 0

Also print the complete resulting first-character distribution.

Do not assume the distribution in advance.

10. Validate Tokenizer Compatibility

Run:

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
"data/shakespeare_troll_sft_canonical.jsonl",
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

print("Examples:", count)
print("Vocabulary size:", tokenizer.vocab_size)
print("Unknown characters:", len(unknown))

for character, lines in unknown.items():
print(repr(character), lines[:10])

assert count == 1327
assert tokenizer.vocab_size == 65
assert not unknown

print("Tokenizer validation PASSED")

Expected:

Examples: 1327
Vocabulary size: 65
Unknown characters: 0
Tokenizer validation PASSED

11. Validate InstructionDataset Masking

Use the existing InstructionDataset:

from src.datasets.instruction_dataset import InstructionDataset
from src.tokenization.char_tokenizer import CharacterTokenizer

with open(
"data/tiny_shakespeare.txt",
encoding="utf-8",
) as f:
tokenizer = CharacterTokenizer(f.read())

dataset = InstructionDataset(
path="data/shakespeare_troll_sft_canonical.jsonl",
tokenizer=tokenizer,
sequence_length=128,
)

input_ids, target_ids = dataset[0]

print("Dataset size:", len(dataset))
print("Input shape:", input_ids.shape)
print("Target shape:", target_ids.shape)

visible_target_ids = [
token
for token in target_ids.tolist()
if token != -100
]

print("TARGET:")
print(tokenizer.decode(visible_target_ids))

first_target_index = next(
i
for i, token in enumerate(target_ids.tolist())
if token != -100
)

first_character = tokenizer.decode(
[target_ids[first_target_index].item()]
)

print(
"First non-masked target:",
repr(first_character),
)

assert len(dataset) == 1327
assert input_ids.shape == (128,)
assert target_ids.shape == (128,)
assert first_character == "A"

print("InstructionDataset validation PASSED")

If the first supervised target is not A:

STOP.

Do not train.

12. Verify Formatting

The canonical dataset must preserve:

TOPIC: ...
Q: ...

and must not introduce:

USER:
ASSISTANT:

Do not remove the TOPIC: conditioning field.

13. Verify Maximum Formatted Length

Run the existing SFT validation logic against the canonical dataset.

If necessary, make the validator accept a dataset path or create a dedicated canonical validator.

Do not modify the original dataset.

Required:

Maximum formatted length <= 128
Examples over 128: 0

Prefer:

Maximum formatted length <= 120

if the dataset satisfies it.

14. Inspect Canonical Examples

Print at least 20 examples and inspect:

INSTRUCTION:
TOPIC: ...

Q: ...

RESPONSE:
...

Pay particular attention to:

AdamW
BPE
Python
API
database
recursion
debugging

The selection must not choose a technically inappropriate response merely because its prefix receives a higher score.

15. Run Tests

Run:

cd /kaggle/working/slm-from-scratch
PYTHONPATH=. pytest -q

Expected existing test result:

11 passed

If tests fail:

STOP.

Do not train.

16. Create a Diagnostic Report

Create:

diagnostics/canonical_dataset_report.txt

Include:

Physical input rows
Unique input instructions
Output rows
Duplicate instructions after canonicalization
Conflicting responses before canonicalization
Conflicting responses after canonicalization
Unique instruction/response pairs
First-character distribution
Maximum formatted length
Vocabulary size
Unknown characters
InstructionDataset masking result
Pytest result

17. Do Not Modify train_sft.py Yet

Do not start training until the canonical dataset passes all validation.

Do not change:

GPT architecture
tokenizer
embedding dimension
number of layers
number of heads
sequence length
base checkpoint

The base checkpoint remains:

checkpoints/step_010000.pt

Do not modify it.

18. Preserve the Existing SFT Baseline

Keep the existing SFT checkpoints:

checkpoints/sft/step_000100.pt
...
checkpoints/sft/step_002000.pt

These are the baseline experiment.

Do not overwrite them.

The canonical experiment must use:

checkpoints/sft_canonical/

19. Training Configuration After Validation

Only after validation passes, configure the next experiment:

pretrained_checkpoint:
checkpoints/step_010000.pt

dataset:
data/shakespeare_troll_sft_canonical.jsonl

checkpoint_directory:
checkpoints/sft_canonical/

max_steps:
500

checkpoint_every:
100

Do not immediately run 2000 steps.

The first purpose is to determine whether canonicalizing response mappings improves conditioning.

20. Evaluation Plan

After the canonical 500-step run, compare:

Base

checkpoints/step_010000.pt

Existing SFT baseline

checkpoints/sft/step_002000.pt

New canonical SFT

checkpoints/sft_canonical/step_000500.pt

Use the same evaluation prompts and evaluation set for all models.

Measure:

First-character accuracy.

Full token accuracy.

Loss.

Greedy generation.

Topic correctness.

Answer quality.

Behavior across different question phrasings.

21. Experimental Hypothesis

The existing 2000-step SFT achieved approximately:

token accuracy: 0.9614
first-character accuracy: 0.55

The model therefore learned much of the response body while still having poor first-character reliability.

The canonical dataset experiment tests whether contradictory response prefixes are causing this behavior.

Desired outcome:

higher first-character accuracy

without a significant loss in:

token accuracy
answer quality
topic conditioning

22. Required Success Criteria Before Training

All of the following must pass:

1327 output rows
1327 unique instructions
1327 unique instruction/response pairs
0 duplicate instructions
0 conflicting responses
0 first-character conflicts
65 vocabulary
0 unknown characters
formatted length <= 128
first supervised target = A
11 tests pass

Only then should the canonical SFT experiment begin.

23. Experimental Structure

Keep the experiments separate:

BASE
│
└── checkpoints/step_010000.pt
│
├── OLD SFT
│ └── checkpoints/sft/step_002000.pt
│
└── CANONICAL SFT
└── checkpoints/sft_canonical/step_000500.pt

This provides a controlled comparison.

Current Task

Implement only:

scripts/generate_canonical_sft.py
scripts/validate_canonical_sft.py
data/shakespeare_troll_sft_canonical.jsonl
diagnostics/canonical_dataset_report.txt

Run all validation checks.

Do not start SFT yet.

After implementation, report exactly:

Files created
Rows generated
Unique instructions
Conflicting responses after canonicalization
First-character conflicts after canonicalization
Maximum formatted length
Vocabulary size
Unknown characters
InstructionDataset masking result
pytest result

Then stop for review before training.
