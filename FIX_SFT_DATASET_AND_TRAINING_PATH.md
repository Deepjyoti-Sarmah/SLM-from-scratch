# Fix SFT InstructionDataset Formatting and Target Masking

## Objective

Fix the SFT data pipeline so that the model trains on the **assistant response only**, using the compact SFT format:

````text
Q: What is AdamW?
A: AdamW is an optimizer using adaptive gradients and decoupled weight decay. ...

The current implementation is still using:

USER:
What is AdamW?

ASSISTANT:

and the target mask is shifted incorrectly.

Do NOT start another training run until all validation checks in this document pass.

1. Files to Inspect

First inspect the current implementations:

sed -n '1,280p' src/datasets/instruction_dataset.py
sed -n '1,280p' scripts/train_sft.py
sed -n '1,220p' src/tokenization/char_tokenizer.py
sed -n '1,260p' src/models/gpt.py

Also search for old prompt formatting:

grep -RniE 'USER:|ASSISTANT:|Q:|A:' src scripts data

Do not modify unrelated model architecture or tokenizer code.

2. Required SFT Format

The dataset JSONL schema remains unchanged:

{
  "instruction": "What is AdamW?",
  "response": "AdamW is an optimizer using adaptive gradients and decoupled weight decay. Thy confusion is understandable."
}

The InstructionDataset must internally construct:

Q: What is AdamW?
A: AdamW is an optimizer using adaptive gradients and decoupled weight decay. Thy confusion is understandable.

The exact prompt must be:

prompt = f"Q: {instruction}\nA: "

Do NOT use:

prompt = f"USER:\n{instruction}\n\nASSISTANT:\n"

Do NOT add additional system/user/assistant tokens.

3. Understand the One-Token Shift

The model performs next-token prediction.

Suppose:

full_text = "Q: What is AdamW?\nA: AdamW is..."

After tokenization:

full_ids = [Q, :, ..., A, :, space, A, d, a, m, ...]

The training tensors are created as:

input_ids = full_ids[:-1]
target_ids = full_ids[1:]

Therefore:

full_ids:

Q : ... A :   A d a m W
            ↑
        response starts

becomes:

input_ids:

Q : ... A :   A d a m
              ↑
target predicts this


target_ids:

: ... A :   A d a m W
            ↑

The first target corresponding to the first response character is shifted by one position relative to the original prompt_ids.

Therefore the prompt mask must account for this shift.

4. Correct Target Masking

Current incorrect logic:

prompt_length = min(
    len(prompt_ids),
    len(target_ids),
)

target_ids[:prompt_length] = [-100] * prompt_length

This masks one response character too many.

For shifted language-model targets, calculate:

target_mask_length = max(
    len(prompt_ids) - 1,
    0,
)

Then:

target_ids = target_ids.copy()

target_ids[:target_mask_length] = (
    [-100] * target_mask_length
)

The first non-masked target must correspond to the first character of the response.

For example, for:

Q: What is AdamW?
A: AdamW is an optimizer...

the first non-masked target should decode to:

A

not:

d

and not:

:
5. Correct __getitem__ Implementation

Modify src/datasets/instruction_dataset.py.

The implementation should follow this structure:

def __getitem__(
    self,
    index: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    example = self.examples[index]

    instruction = example["instruction"]
    response = example["response"]

    prompt = f"Q: {instruction}\nA: "

    full_text = prompt + response

    full_ids = self.tokenizer.encode(full_text)
    prompt_ids = self.tokenizer.encode(prompt)

    # We need sequence_length + 1 tokens because
    # input_ids and target_ids are shifted by one.
    full_ids = full_ids[: self.sequence_length + 1]

    if len(full_ids) < 2:
        raise ValueError(
            f"Example {index} is too short."
        )

    input_ids = full_ids[:-1]
    target_ids = full_ids[1:]

    # --------------------------------------------------
    # IMPORTANT:
    # target_ids are shifted by one token.
    #
    # Therefore the prompt occupies:
    # len(prompt_ids) - 1
    # positions in target_ids.
    # --------------------------------------------------

    target_mask_length = max(
        len(prompt_ids) - 1,
        0,
    )

    target_ids = target_ids.copy()

    target_ids[:target_mask_length] = (
        [-100] * target_mask_length
    )

    # Pad remaining positions.
    padding_length = self.sequence_length - len(input_ids)

    if padding_length > 0:
        input_ids.extend(
            [0] * padding_length
        )

        target_ids.extend(
            [-100] * padding_length
        )

    return (
        torch.tensor(
            input_ids,
            dtype=torch.long,
        ),
        torch.tensor(
            target_ids,
            dtype=torch.long,
        ),
    )

Preserve the existing _load_examples() and __len__() behavior unless a bug is found.

6. Important Truncation Requirement

The dataset generator has already been modified so that formatted examples should fit within the model's context window.

Do NOT silently truncate responses.

The dataset should satisfy:

formatted length <= 120

before reaching InstructionDataset.

The InstructionDataset may still retain the defensive:

full_ids = full_ids[: self.sequence_length + 1]

but this should not normally truncate valid examples.

Do not solve a dataset-length problem inside InstructionDataset.

7. Verify the Dataset Directly

After modifying the code, run:

python - <<'PY'
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

print("Dataset size:", len(dataset))

input_ids, target_ids = dataset[0]

print("Input shape:", input_ids.shape)
print("Target shape:", target_ids.shape)

print()
print("INPUT:")
print(tokenizer.decode(input_ids.tolist()))

print()
print("TARGET:")

visible_target_ids = [
    token
    for token in target_ids.tolist()
    if token != -100
]

print(
    tokenizer.decode(visible_target_ids)
)

first_target_index = next(
    i
    for i, token in enumerate(target_ids.tolist())
    if token != -100
)

print()
print("First non-masked target:")
print("Index:", first_target_index)
print(
    "Character:",
    tokenizer.decode(
        [target_ids[first_target_index].item()]
    ),
)
PY

Expected:

Dataset size: 2780
Input shape: torch.Size([128])
Target shape: torch.Size([128])

The input should begin with:

Q: ...
A: ...

NOT:

USER:
...
ASSISTANT:
...

The visible target should begin with the response:

AdamW is an optimizer...

And:

First non-masked target:
Character: A
8. Verify the Exact Training Example

Use the AdamW example specifically.

Find it:

grep -n '"instruction": "What is AdamW?"' \
    data/shakespeare_troll_sft_large.jsonl | head

Then run:

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

for i in range(len(dataset)):
    input_ids, target_ids = dataset[i]

    text = tokenizer.decode(input_ids.tolist())

    if "What is AdamW?" in text:
        print("Found dataset index:", i)
        print()
        print("INPUT:")
        print(text)

        visible_targets = [
            token
            for token in target_ids.tolist()
            if token != -100
        ]

        print()
        print("TARGET:")
        print(
            tokenizer.decode(visible_targets)
        )

        break

Expected structure:

INPUT:
Q: What is AdamW?
A: AdamW is an optimizer using adaptive gradients...

TARGET:
AdamW is an optimizer using adaptive gradients...

The target must NOT begin with:

damW

It must NOT begin with:

:

It must NOT include:

Q: What is AdamW?
A:
9. Verify Masking Mathematically

For the AdamW example:

prompt = "Q: What is AdamW?\nA: "

The first response character is:

A

The target sequence must contain:

[-100, ..., -100, ord("A"), ord("d"), ord("a"), ...]

Therefore:

first_non_masked = next(
    i
    for i, token in enumerate(target_ids.tolist())
    if token != -100
)

first_character = tokenizer.decode(
    [target_ids[first_non_masked].item()]
)

assert first_character == "A"

If this assertion fails, STOP.

Do not train.

10. Verify the Loss Uses the Mask

src/models/gpt.py currently uses:

loss = F.cross_entropy(
    flattened_logits,
    flattened_targets,
)

Change it only if necessary.

PyTorch F.cross_entropy supports:

ignore_index=-100

The implementation should explicitly use:

loss = F.cross_entropy(
    flattened_logits,
    flattened_targets,
    ignore_index=-100,
)

This makes the intended SFT behavior explicit:

-100 targets → ignored by loss
actual response targets → contribute to loss

Do not remove -100 masking.

Do not calculate loss manually unless required.

11. Verify Model Loss

After making the changes, test:

import torch

input_ids, target_ids = dataset[0]

model.eval()

with torch.no_grad():
    logits, loss = model(
        token_ids=input_ids.unsqueeze(0).to(device),
        targets=target_ids.unsqueeze(0).to(device),
    )

print("Logits shape:", logits.shape)
print("Loss:", loss.item())

Expected:

Logits shape: torch.Size([1, 128, 65])
Loss: <finite number>

There must be no:

nan
inf
12. Fix Training Dataset Path

Inspect:

grep -Rni "shakespeare_troll_sft" src scripts

The SFT training script must use:

dataset_path="data/shakespeare_troll_sft_large.jsonl"

NOT:

dataset_path="data/shakespeare_troll_sft.jsonl"

Do not create a duplicate dataset just to satisfy the old path.

Fix the configuration.

The base checkpoint must remain:

pretrained_checkpoint="checkpoints/step_010000.pt"

The output directory must remain:

checkpoint_directory="checkpoints/sft"
13. Verify the Training Script

Before training, run:

grep -nE \
'pretrained_checkpoint|dataset_path|checkpoint_directory|max_steps|checkpoint_every' \
scripts/train_sft.py

Expected:

pretrained_checkpoint = checkpoints/step_010000.pt
dataset_path = data/shakespeare_troll_sft_large.jsonl
checkpoint_directory = checkpoints/sft
14. Do Not Reuse the Old SFT Checkpoint

The corrected training run must start from:

checkpoints/step_010000.pt

Do NOT load:

checkpoints/sft/step_000300.pt
checkpoints/sft/step_002000.pt

Those were produced with the previous dataset/masking behavior.

The training flow must be:

Base pretrained model
        |
        v
step_010000.pt
        |
        v
Corrected InstructionDataset
        |
        v
Corrected SFT training
        |
        v
checkpoints/sft/
15. Kaggle Verification

The repository root must be:

/kaggle/working/slm-from-scratch

Run:

cd /kaggle/working/slm-from-scratch
pwd

Then:

python -c "
from src.datasets.instruction_dataset import InstructionDataset
print('InstructionDataset import: OK')
"

Then run the dataset verification from this document.

Only after it passes should training begin.

16. Final Acceptance Criteria

The implementation is correct only if ALL of the following pass:

Formatting
Q: What is AdamW?
A: AdamW is an optimizer...

and NOT:

USER:
What is AdamW?

ASSISTANT:
Masking

First non-masked target:

A

NOT:

d
Dataset
Dataset size: 2780
Input shape: [128]
Target shape: [128]
Loss
ignore_index=-100

must be used.

Paths
data/shakespeare_troll_sft_large.jsonl
checkpoints/step_010000.pt
checkpoints/sft/
Training

Start from:

step_010000.pt

Never from an old SFT checkpoint.

17. Important Constraints

Do NOT:

change the GPT architecture
change vocab_size
change the character tokenizer
change the pretrained checkpoint
use USER: / ASSISTANT: formatting
mask the first response character
silently truncate responses
create a duplicate dataset under the old filename
train from an old SFT checkpoint
hardcode /kaggle/input/...
start a new training run before dataset validation passes

Do:

use Q: ...\nA: formatting
account for the one-token LM shift
mask only prompt targets
leave the first response character unmasked
explicitly use ignore_index=-100
use data/shakespeare_troll_sft_large.jsonl
train from checkpoints/step_010000.pt
validate locally/in Kaggle before training

The key bug is the **off-by-one masking error**. Your current result:

```text
First non-masked target:
Character: d

proves that the A of AdamW is being masked. After the fix, it must be:

First non-masked target:
Character: A

Do that verification first; only then rerun the SFT on Kaggle.
````
