# Fix SFT Dataset Length and Training Data Path

## Objective

Fix the current Shakespeare Troll SFT pipeline before the next Kaggle training run.

The current SFT dataset is incompatible with the model's `128` character-level context window:

- Dataset size: `2780`
- Formatted example minimum: `157` characters
- Formatted example maximum: `308` characters
- Formatted example mean: `211.77` characters
- Response minimum: `119` characters
- Response maximum: `232` characters
- Response mean: `155.91` characters
- `2780 / 2780` examples are longer than the `128` token context.

Because the tokenizer is character-level, one character is approximately one model token.

The current training pipeline therefore truncates every SFT example.

Do not simply increase the number of training steps.

The goal is to regenerate the SFT dataset so that the complete question + technical answer + Shakespearean troll fits inside the existing `128` token context.

---

# 1. Preserve the Existing Model Architecture

Do NOT change:

```text
vocab_size = 65
max_sequence_length = 128
embedding_dim = 256
num_heads = 8
num_layers = 6

Do NOT modify:

checkpoints/step_010000.pt

Do NOT replace the character tokenizer.

The existing pretrained checkpoint must remain compatible with the model.

2. Inspect the Existing Codebase First

Before changing anything, inspect:

sed -n '1,360p' scripts/generate_shakespeare_troll_sft.py

Also inspect:

sed -n '1,280p' src/datasets/instruction_dataset.py
sed -n '1,280p' scripts/train_sft.py
sed -n '1,220p' src/configs/sft_config.py
sed -n '1,220p' src/tokenization/char_tokenizer.py

Understand the existing implementation before modifying it.

Do not rewrite unrelated parts of the project.

3. Fix the SFT Dataset Generator

The source of truth for the large dataset is:

scripts/generate_shakespeare_troll_sft.py

Modify the generator rather than manually editing the generated JSONL.

The generated dataset must retain the existing scale and concept/question variation.

Target:

2780 examples

The current conceptual structure should remain:

132 technical concepts
×
10 question templates
×
2 trolls
=
2640 examples

+

7 war-story questions
×
20 trolls
=
140 examples

Total = 2780

Do not unnecessarily reduce the dataset size.

4. Change the SFT Formatting

The current formatting is:

USER:
What is AdamW?

ASSISTANT:
AdamW is an optimizer using adaptive gradients and decoupled weight decay...

This wastes too much of the 128-character context.

Use compact formatting:

Q: What is AdamW?
A: AdamW uses adaptive gradients and decoupled weight decay. Thy confusion doth exceed thy understanding.

The exact implementation can continue to use the existing instruction and response JSON fields.

The JSONL structure must remain:

{
  "instruction": "...",
  "response": "..."
}

Do not change the external dataset schema.

5. Make Responses Short

Each response must contain:

A technically correct explanation.
A short Shakespearean troll.

Desired structure:

technical explanation. Shakespearean troll.

Example:

AdamW uses adaptive gradients and decoupled weight decay. Thy confusion doth exceed thy understanding.

Another example:

An API lets software communicate through defined interfaces. Thy code hath asked for a messenger; here stands one.

The troll must be playful, not abusive.

Do not include generation instructions inside the response.

Do NOT generate responses such as:

Explain this in plain language.
Answer clearly, then add a Shakespearean roast.
Teach the concept and insult the user.

Those are instructions to a generator, not training targets.

6. Enforce a Hard Length Limit

This is critical.

Because the model has:

max_sequence_length = 128

the complete formatted example should be <= 120 characters.

Use 120, not 128, to leave a small safety margin.

The generator must check this.

Conceptually:

formatted = (
    "Q: "
    + instruction
    + "\nA: "
    + response
)

if len(formatted) > 120:
    raise ValueError(
        f"SFT example exceeds length limit: {len(formatted)}"
    )

Do NOT silently truncate the response.

If an example exceeds the limit:

FAIL

and identify the offending example.

Do not do:

response = response[:120]

because that could cut a sentence or troll in half.

The generator itself must produce shorter content.

7. Verify Dataset Length After Generation

After regenerating:

data/shakespeare_troll_sft_large.jsonl

run a validation script.

It must report:

Examples: 2780
Maximum formatted length: <= 120
Examples over 128: 0
Examples over 120: 0

Also report:

Minimum formatted length
Maximum formatted length
Mean formatted length
Median formatted length
Minimum response length
Maximum response length
Mean response length
8. Preserve the 65-Character Vocabulary

The tokenizer is created from:

data/tiny_shakespeare.txt

The vocabulary size must remain:

65

Every character in:

instruction
response

must exist in:

tokenizer.token_to_id

Run a validation equivalent to:

from src.tokenization.char_tokenizer import CharacterTokenizer

with open(
    "data/tiny_shakespeare.txt",
    encoding="utf-8",
) as f:
    tokenizer = CharacterTokenizer(f.read())

Then scan the complete dataset.

The final result must be:

Vocabulary size: 65
Unknown characters: 0

Do not change the tokenizer to accommodate new characters.

Avoid characters such as:

+
0
1
2
3
4
5
6
7
8
9

unless they already exist in the vocabulary.

The existing dataset already demonstrated this problem with:

C++
HTTP 404
HTTP 500

These were normalized previously to vocabulary-safe forms such as:

CPP
HTTP four zero four
HTTP five zero zero

Keep that behavior.

9. Fix the Training Data Path

The current Kaggle run failed because scripts/train_sft.py expected:

data/shakespeare_troll_sft.jsonl

but the actual large dataset is:

data/shakespeare_troll_sft_large.jsonl

This must be fixed.

Inspect:

grep -Rni "shakespeare_troll_sft" .

The SFT training configuration/script must point to:

data/shakespeare_troll_sft_large.jsonl

not:

data/shakespeare_troll_sft.jsonl

The intended configuration should be equivalent to:

config = SFTConfig(
    pretrained_checkpoint="checkpoints/step_010000.pt",
    dataset_path="data/shakespeare_troll_sft_large.jsonl",
    checkpoint_directory="checkpoints/sft",
    ...
)

Do not create a fake duplicate of the large dataset under the old filename just to hide the configuration error.

Fix the actual path in the SFT configuration/training script.

10. Make the Training Path Robust on Kaggle

The training script is executed from:

/kaggle/working/slm-from-scratch

Therefore relative paths should resolve from the repository root:

data/...
checkpoints/...
scripts/...
src/...

Before training, the Kaggle notebook should execute:

cd /kaggle/working/slm-from-scratch

Then verify:

pwd

Expected:

/kaggle/working/slm-from-scratch

Verify the dataset:

ls -lh data/shakespeare_troll_sft_large.jsonl

Verify the base checkpoint:

ls -lh checkpoints/step_010000.pt
11. Do Not Introduce Absolute Kaggle Paths

Do NOT hardcode:

/kaggle/input/...

inside the project.

The repository should work both locally and on Kaggle.

Use:

data/shakespeare_troll_sft_large.jsonl

and:

checkpoints/step_010000.pt

as project-relative paths.

12. Verify the Dataset Before Training

Run:

python scripts/generate_shakespeare_troll_sft.py

Then run the validation.

The validation must verify all of the following:

JSONL valid                    ✓
2780 examples                  ✓
Vocabulary size = 65           ✓
Unknown characters = 0         ✓
Maximum formatted length <=120 ✓
Examples over 120 = 0          ✓
Examples over 128 = 0          ✓
No meta instructions           ✓
No accidental C++               ✓
No numeric vocabulary errors   ✓
13. Verify InstructionDataset

After regeneration, inspect one example:

from src.datasets.instruction_dataset import InstructionDataset
from src.tokenization.char_tokenizer import CharacterTokenizer

with open(
    "data/tiny_shakespeare.txt",
    encoding="utf-8",
) as f:
    tokenizer = CharacterTokenizer(f.read())

dataset = InstructionDataset(
    path="data/shakespeare_troll_sft_large.jsonl",
    tokenizer=tokenizer,
    sequence_length=128,
)

input_ids, target_ids = dataset[0]

print("Dataset size:", len(dataset))
print("Input shape:", input_ids.shape)
print("Target shape:", target_ids.shape)
print(tokenizer.decode(input_ids.tolist()))
print(target_ids.tolist())

The question/prompt must be masked.

The answer must not be masked.

Conceptually:

Q: What is AdamW?
A: AdamW uses adaptive gradients...

Target:

[MASK][MASK][MASK]...
AdamW uses adaptive gradients...
14. Train From the Original Base Checkpoint

Do NOT continue SFT from the bad 300-step SFT checkpoint.

Start again from:

checkpoints/step_010000.pt

The training flow must be:

step_010000.pt
       ↓
corrected SFT dataset
       ↓
SFT
       ↓
checkpoints/sft/

The original base checkpoint must remain untouched.

15. Kaggle Training

Once validation passes:

cd /kaggle/working/slm-from-scratch

Run:

python -m scripts.train_sft

or the project's established SFT command if different.

Confirm the output says:

Device: cuda
GPU: Tesla T4
Vocabulary size: 65
Base checkpoint loaded:
Parameters: 4,782,336
SFT examples: 2780

Then training should begin.

16. Checkpoint Requirements

SFT checkpoints must be written to:

checkpoints/sft/

For example:

checkpoints/sft/step_000100.pt
checkpoints/sft/step_000200.pt
checkpoints/sft/step_000300.pt

At the end, also save the final checkpoint if the training script is configured to do so.

The checkpoint must contain at minimum:

{
    "model_state": ...,
    "optimizer_state": ...,
    "global_step": ...
}

Do not overwrite:

checkpoints/step_010000.pt
17. Test Before Uploading Anywhere

After SFT, load the final SFT checkpoint and test prompts such as:

What is Python?
What is a database?
What is an API?
What is machine learning?
What is AdamW?

The expected behavior is:

technical answer
+
Shakespearean style
+
short playful troll

It should NOT produce:

USER:
...
ASSISTANT:
...

as part of the generated answer.

The prompt may appear in the displayed full sequence because generation starts from the supplied prompt, but the generated continuation itself should be the answer.

18. Compare Base Model vs SFT Model

Always test the same prompts on:

Base checkpoint

and:

SFT checkpoint

Example:

Prompt:
What is Python?

Record both outputs.

The SFT model should show a measurable behavioral shift toward:

technical explanation
+
Shakespearean phrasing
+
troll

If the SFT output is still incoherent, do not immediately increase training steps.

First inspect:

training loss
dataset formatting
masking
sequence length
generation parameters
19. Final Acceptance Criteria

The task is complete only when all of these are true:

Dataset
2780 examples
65-character vocabulary
0 unknown characters
0 examples > 120 formatted characters
0 examples > 128 formatted characters
0 meta/generation instructions
Training
Base checkpoint loads successfully
CUDA/T4 is used on Kaggle
SFT dataset loads successfully
Loss decreases or shows meaningful training behavior
SFT checkpoints are saved
Generation

The model produces answers resembling:

technical explanation
+
Shakespearean language
+
playful troll

rather than random text.

Paths

The SFT script uses:

data/shakespeare_troll_sft_large.jsonl

and:

checkpoints/step_010000.pt

and saves to:

checkpoints/sft/

All paths must be repository-relative.

Important Constraints

Do not:

change the pretrained architecture
change the tokenizer
change vocabulary size
modify step_010000.pt
silently truncate responses
delete examples just because they are long
replace the dataset with a tiny test dataset
hardcode /kaggle/input/...
train from the previous bad SFT checkpoint
add meta-instructions to response targets

Do:

modify the dataset generator
shorten technical answers and trolls
enforce a hard length check
preserve 2780 examples
fix the dataset path in the training configuration
validate the dataset before training
train from the original pretrained checkpoint
save SFT checkpoints under checkpoints/sft/
test base vs SFT behavior after training

The key change is that the agent should **fix the generator and the actual training configuration**, rather than merely creating another copy of the dataset under `shakespeare_troll_sft.jsonl`.
```
