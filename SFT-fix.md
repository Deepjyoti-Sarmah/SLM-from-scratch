# Full SFT TOPIC Conditioning — Next Task

## Objective

The previous full SFT run learned the Shakespearean style and some technical-answer patterns, but it frequently answered the **wrong concept**.

For example:

```text
PROMPT:
Q: What is AdamW?
A:

OUTPUT:
Q: What is AdamW?
A: An API lets tasks a model on examples...

The model learned the answer style, but concept selection was unreliable.

We therefore ran a small controlled experiment using an explicit TOPIC: conditioning signal.

The experiment succeeded on all 5 tested concepts.

The next task is to apply the same conditioning format to the full 2780-example SFT dataset, validate it, and prepare it for a fresh SFT run.

1. What We Have Proven
1.1 Experimental dataset

The experiment contained exactly 5 concepts:

AdamW
Python
API
database
recursion

The conditioning format was:

TOPIC: {concept}
Q: What is it?
A:
2. Exact Experiment Validation Results

Validation output:

Rows: 5
Topics: ['API', 'AdamW', 'Python', 'database', 'recursion']
Vocabulary size: 65
Unknown characters: 0
Minimum formatted length: 96
Maximum formatted length: 119
Examples over 120: 0
Examples over 128: 0
ALL CHECKS PASSED

Masking verification:

AdamW     first: A
Python    first: P
API       first: A
database  first: A
recursion first: R

Therefore the first response character was correctly supervised.

3. Exact Experiment Training Results

The experiment started from:

checkpoints/step_010000.pt

Model:

Parameters: 4,782,336

Dataset:

5 examples

Training:

300 steps

Device:

CPU

Final loss:

0.0092

Checkpoints:

checkpoints/sft_conditioning_experiment/step_000100.pt
checkpoints/sft_conditioning_experiment/step_000200.pt
checkpoints/sft_conditioning_experiment/step_000300.pt
4. Exact Generation Results
AdamW

Prompt:

TOPIC: AdamW
Q: What is it?
A:

Generated:

TOPIC: AdamW
Q: What is it?
A: AdamW uses adaptive gradients and decoupled weight decay. Thy bug bows, forsooth. Thy fun

Concept match:

True
Python

Prompt:

TOPIC: Python
Q: What is it?
A:

Generated:

TOPIC: Python
Q: What is it?
A: Python is a high level programming language. Thy code hath hope. Thy code needs a messeng

Concept match:

True
API

Prompt:

TOPIC: API
Q: What is it?
A:

Generated:

TOPIC: API
Q: What is it?
A: An API lets software communicate through defined interfaces. Thy code needs a messenger.

Concept match:

True
database

Prompt:

TOPIC: database
Q: What is it?
A:

Generated:

TOPIC: database
Q: What is it?
A: A database stores and organizes data for retrieval. Thy records are now in order. Thy rec

Concept match:

True
recursion

Prompt:

TOPIC: recursion
Q: What is it?
A:

Generated:

TOPIC: recursion
Q: What is it?
A: Recursion solves problems by calling itself on small cases. Thy function met itself. Thy

Concept match:

True
5. What We Can Infer

We have evidence that explicit topic conditioning can improve concept selection.

The controlled experiment produced:

5 / 5 concept matches

with greedy generation.

However, this experiment does not prove that the same behavior will generalize to all 2780 examples.

Do not claim that it does.

The next full-dataset experiment is required.

6. Important Constraint

Do not change:

vocab_size = 65
max_sequence_length = 128
embedding_dim = 256
num_heads = 8
num_layers = 6

Do not change the tokenizer.

Do not modify:

checkpoints/step_010000.pt

Do not change the GPT architecture.

The only major change in this task is the SFT conditioning/data format.

7. Task 1 — Inspect the Existing Generator

Before making changes, inspect:

sed -n '1,360p' scripts/generate_shakespeare_troll_sft.py

Also inspect:

sed -n '1,320p' src/datasets/instruction_dataset.py
sed -n '1,260p' scripts/validate_sft_dataset.py
sed -n '1,260p' scripts/verify_sft_ready.py
sed -n '1,260p' scripts/train_sft.py

Understand the existing implementation before modifying it.

Do not rewrite unrelated code.

8. Task 2 — Modify the Full Dataset Generator

Modify:

scripts/generate_shakespeare_troll_sft.py

The generator should produce the existing 2780 examples, but the instruction should now contain the topic.

Instead of:

Q: What is AdamW?
A:

the effective training format should be:

TOPIC: AdamW
Q: What is AdamW?
A:

The JSONL schema must remain unchanged:

{
  "instruction": "...",
  "response": "..."
}

For example:

{
  "instruction": "TOPIC: AdamW\nQ: What is AdamW?",
  "response": "AdamW uses adaptive gradients and decoupled weight decay. Thy bug bows, forsooth."
}

Do not change the external JSONL schema.

9. Task 3 — Preserve the 2780 Examples

The dataset must remain:

132 technical concepts
× 10 question templates
× 2 trolls
=
2640 examples

+

7 war-story questions
× 20 trolls
=
140 examples

TOTAL = 2780

Do not reduce the dataset.

Do not replace it with the 5-example experiment.

Do not delete examples simply because they become too long.

If examples become too long, shorten their content while preserving the dataset size.

10. Task 4 — Preserve Technical Answer + Troll Structure

Every response should still contain:

technical explanation
+
short Shakespearean troll

For example:

AdamW uses adaptive gradients and decoupled weight decay. Thy bug bows, forsooth.

The response must remain an actual training target.

Do not generate meta-instructions such as:

Explain AdamW clearly.
Then add a Shakespearean troll.

Those are instructions, not answers.

11. Task 5 — Keep the Prompt Compact

The effective formatted example should be:

TOPIC: AdamW
Q: What is AdamW?
A: AdamW uses adaptive gradients and decoupled weight decay. Thy bug bows, forsooth.

The implementation may continue using:

prompt = f"TOPIC: {topic}\nQ: {instruction}\nA: "

or equivalent logic.

Do not return to:

USER:
...
ASSISTANT:
...

The compact format is intentional.

12. Task 6 — Enforce the Context Limit

The model has:

max_sequence_length = 128

The full formatted example must remain:

<= 120 characters

Use the existing safety margin.

The generator/validator must reject examples over the limit.

Do not silently truncate.

Do not do:

response = response[:120]

because this can cut technical answers or trolls in half.

Instead, shorten the generated content at the source.

13. Task 7 — Regenerate the Full Dataset

After modifying the generator, regenerate:

data/shakespeare_troll_sft_large.jsonl

Do not use the previous 5-example experiment dataset as the final training dataset.

Run:

uv run python scripts/generate_shakespeare_troll_sft.py
14. Task 8 — Validate the Full Dataset

Run:

uv run python scripts/validate_sft_dataset.py

The final dataset must satisfy:

Examples: 2780
Vocabulary size: 65
Unknown characters: 0
Examples over 120: 0
Examples over 128: 0

Also report:

Minimum formatted length
Maximum formatted length
Mean formatted length
Median formatted length
Minimum response length
Maximum response length
Mean response length
15. Task 9 — Verify TOPIC Formatting

Inspect several generated examples.

They should look like:

TOPIC: AdamW
Q: What is AdamW?
A: ...
TOPIC: Python
Q: What is Python?
A: ...
TOPIC: API
Q: What is an API?
A: ...

Do not accept examples where:

TOPIC:

is missing.

Also verify that the topic corresponds to the concept being asked about.

For example, this is incorrect:

TOPIC: AdamW
Q: What is Python?

unless that pairing is intentionally part of the dataset design.

For the normal technical examples, topic and question should correspond.

16. Task 10 — Verify Character Vocabulary

The tokenizer remains:

data/tiny_shakespeare.txt

Vocabulary must remain:

65

Run the existing vocabulary validation.

Expected:

Vocabulary size: 65
Unknown characters: 0

Do not add characters to the tokenizer to make the dataset pass.

17. Task 11 — Verify SFT Masking

Use:

src/datasets/instruction_dataset.py

and verify an example such as AdamW.

Expected input:

TOPIC: AdamW
Q: What is AdamW?
A: AdamW uses adaptive gradients...

Expected target behavior:

TOPIC: [MASK]
Q: [MASK]
A: [MASK]

Conceptually, only the response should be supervised.

The first supervised character should be the first character of the response.

For example:

Response:
AdamW uses...

The first non-masked target should be:

A

not:

d

not:

:

and not any part of the prompt.

18. Task 12 — Run SFT Readiness Verification

Run:

uv run python scripts/verify_sft_ready.py

It must pass using the newly regenerated 2780-example TOPIC-conditioned dataset.

Expected:

SFT READY: all checks passed
19. Task 13 — Run the Full Test Suite

Run:

uv run pytest -q --ignore=kaggle_sft_package

Expected:

11 passed

If tests fail because of the new TOPIC behavior, inspect the failure and update only the relevant test/implementation.

Do not disable tests just to make the suite pass.

20. Task 14 — Do NOT Train Yet

This task ends after:

generator changed
dataset regenerated
dataset validated
masking verified
SFT readiness verified
tests passed

Do NOT start the 2000-step SFT in this task.

First report the results.

21. Required Agent Report

The agent must report exactly:

FULL TOPIC DATASET VALIDATION

Examples:
Vocabulary size:
Unknown characters:

Minimum formatted length:
Maximum formatted length:
Mean formatted length:
Median formatted length:

Minimum response length:
Maximum response length:
Mean response length:

Examples over 120:
Examples over 128:

TOPIC formatting:
PASS / FAIL

Masking:
PASS / FAIL

SFT readiness:
PASS / FAIL

Tests:
PASS / FAIL

Also show at least 5 actual examples from the regenerated dataset.

22. Do Not Make These Changes

Do NOT:

change the GPT architecture
change vocab_size
change the tokenizer
modify checkpoints/step_010000.pt
reduce the dataset below 2780 examples
use the 5-example dataset for final SFT
silently truncate examples
hardcode /kaggle/input/ paths
train from an old SFT checkpoint
remove masking
train on TOPIC/Q tokens
return to USER/ASSISTANT formatting
add meta-instructions to responses
disable failing tests
23. Required Changes

DO:

modify the full dataset generator
add TOPIC conditioning
preserve 2780 examples
preserve technical answers
preserve Shakespearean trolls
keep formatted examples <= 120 characters
preserve 65-character vocabulary
preserve assistant-only loss masking
regenerate the full dataset
validate the dataset
verify masking
run SFT readiness checks
run tests
report exact statistics
24. Expected Data Flow

The new training data should conceptually flow like this:

Technical concept
      |
      v
TOPIC: {concept}
      |
      v
Question template
      |
      v
Q: {question}
      |
      v
Technical answer + Shakespearean troll
      |
      v
2780 JSONL examples
      |
      v
InstructionDataset
      |
      v
TOPIC + Q tokens masked
      |
      v
Response tokens supervised
      |
      v
Fresh SFT from step_010000.pt
25. Why We Are Doing This

The previous full SFT demonstrated that the model can learn:

technical-answer style
Shakespearean style

but concept selection was poor.

The controlled experiment demonstrated:

TOPIC: AdamW
        ↓
AdamW answer

TOPIC: Python
        ↓
Python answer

TOPIC: API
        ↓
API answer

TOPIC: database
        ↓
database answer

TOPIC: recursion
        ↓
recursion answer

All five matched.

Therefore the next experiment should isolate the effect of the TOPIC: conditioning signal while keeping:

model
tokenizer
architecture
base checkpoint
dataset size
training objective

unchanged.

26. After This Task

If the full 2780-example dataset passes all checks, the next task will be:

Create a fresh Kaggle package
        ↓
Upload package to Kaggle
        ↓
Verify files on Kaggle
        ↓
Run dataset validation
        ↓
Run SFT readiness verification
        ↓
Start fresh SFT from step_010000.pt
        ↓
Train for 2000 steps
        ↓
Evaluate checkpoints:
    step_000500
    step_001000
    step_001500
    step_002000
        ↓
Compare against base checkpoint
        ↓
Evaluate concept accuracy

Do not proceed to that stage until the full TOPIC-conditioned dataset passes validation.

Definition of Done

This task is complete only when:

[ ] Generator uses TOPIC conditioning
[ ] Dataset contains 2780 examples
[ ] JSONL schema is unchanged
[ ] Vocabulary size is 65
[ ] Unknown characters = 0
[ ] Maximum formatted length <= 120
[ ] Examples over 120 = 0
[ ] Examples over 128 = 0
[ ] Topic/question pairing is correct
[ ] Technical answers are preserved
[ ] Shakespearean troll is preserved
[ ] TOPIC/Q tokens are masked
[ ] First supervised character is correct
[ ] SFT readiness check passes
[ ] Full test suite passes
[ ] No SFT training performed yet
[ ] Agent reports exact validation statistics
Important

Do not infer success from loss alone.

The 5-example experiment had a final loss of:

0.0092

but the important evidence was the 5/5 correct concept selection during generation.

For the full dataset, generation-based evaluation will be required after training.
```
