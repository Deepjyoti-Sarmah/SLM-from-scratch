Fix SFT Dataset Quality, Context Length, and Training Data Path

Objective

Fix the Shakespeare Troll SFT pipeline before the next Kaggle trainingrun.

The current SFT dataset is incompatible with the model's 128character-level context window:

Dataset size: 2780

Formatted example minimum: 157 characters

Formatted example maximum: 308 characters

Formatted example mean: 211.77 characters

Response minimum: 119 characters

Response maximum: 232 characters

Response mean: 155.91 characters

2780 / 2780 examples are longer than the 128 token context.

Because the tokenizer is character-level, one character is approximatelyone model token.

The previous 300-step SFT run produced incoherent outputs. Do notsimply increase the number of training steps. Fix the training datarepresentation and quality first.

The goal is to regenerate the SFT dataset so that:

question
↓
short, technically correct answer
↓
Shakespearean / witty troll

fits completely inside the existing 128 token context.

1. Preserve the Existing Model Architecture

Do NOT change:

vocab_size = 65
max_sequence_length = 128
embedding_dim = 256
num_heads = 8
num_layers = 6

Do NOT modify:

checkpoints/step_010000.pt

Do NOT replace the character tokenizer.

The existing pretrained checkpoint must remain compatible with themodel.

2. Inspect the Existing Codebase First

Before changing anything, inspect:

sed -n '1,360p' scripts/generate_shakespeare_troll_sft.py

sed -n '1,280p' src/datasets/instruction_dataset.py

sed -n '1,320p' scripts/train_sft.py

sed -n '1,220p' src/configs/sft_config.py

sed -n '1,220p' src/tokenization/char_tokenizer.py

Also inspect:

grep -Rni "shakespeare_troll_sft" .

Understand the existing implementation before modifying it.

Do not rewrite unrelated parts of the project.

3. Preserve the Dataset Size and Concept Coverage

The source of truth for the large dataset is:

scripts/generate_shakespeare_troll_sft.py

Modify the generator rather than manually editing the generated JSONL.

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

-

7 war-story questions
×
20 trolls
=

140 examples

Total = 2780

Do not unnecessarily reduce the dataset size.

4. Keep the JSONL Schema

The external dataset schema must remain:

{"instruction": "...", "response": "..."}

Do not introduce additional required fields.

5. Change the SFT Formatting

The previous formatting was:

USER:
What is AdamW?

ASSISTANT:
AdamW is an optimizer using adaptive gradients and decoupled weight decay...

This wastes too much of the 128-character context.

Use compact formatting:

Q: What is AdamW?
A: AdamW uses adaptive gradients and decoupled weight decay. Thy confusion doth exceed thy understanding.

The exact instruction and response fields remain unchanged in JSONL.

The compact prompt should be implemented in InstructionDataset.

Do not leave the old USER: / ASSISTANT: format if it causesunnecessary context usage.

6. Improve the Response Quality

Each response must contain:

A technically correct explanation.

A short, playful Shakespearean troll.

Desired structure:

technical explanation. Shakespearean troll.

Weak examples include:

Thy confusion is noted.
Now thou knowest.
Well met, remember it.
The matter is settled.
Ask, and it is answered.
Thus the lesson is given.
This answer is thine.
May thy code stay true.

These are generic closings, not meaningful trolls.

Prefer varied endings such as:

Thy optimizer hath more discipline than thy questions.

Even thy gradients deserve better manners than this confusion.

A useful question at last; mark the calendar, good mortal.

Thy parameters are now better governed than thy reasoning.

The compiler hath survived thy question; let us hope thy code doth likewise.

Thy curiosity is noble, though thy confusion entered wearing a crown.

The troll must be:

playful

Shakespearean in wording

concise

varied

not abusive

Do not include generation instructions inside the response.

Do NOT generate responses such as:

Explain this in plain language.
Answer clearly, then add a Shakespearean roast.
Teach the concept and insult the user.

Those are generator instructions, not training targets.

7. Add Factual Variation

Do not repeat exactly the same factual sentence for every question abouta concept.

For AdamW, for example, avoid using only:

AdamW uses adaptive gradients and decoupled weight decay.

Use several technically equivalent formulations:

AdamW uses adaptive gradients and decoupled weight decay.

AdamW is an optimizer that separates weight decay from the gradient update.

AdamW combines adaptive parameter updates with decoupled weight decay.

AdamW modifies Adam by applying weight decay independently of the gradient update.

Then combine factual variants with different troll variants.

The target structure is:

                    ┌── factual variation

Question ───────────┤
└── troll variation

not:

Question
↓
same answer
↓
random stock ending

Preserve technical correctness. Do not invent facts merely to createvariation.

8. Enforce a Hard Length Limit

The model has:

max_sequence_length = 128

The complete formatted example must be:

<= 120 characters

Use 120, not 128, to leave a safety margin.

Conceptually:

formatted = (
"Q: " + instruction + "
A: " + response
)

if len(formatted) > 120:
raise ValueError(
f"SFT example exceeds length limit: {len(formatted)}"
)

Do not silently truncate the response.

Do not do:

response = response[:120]

Instead, make the source response shorter.

The generator must fail loudly if an example exceeds the limit.

9. Validate Dataset Length

After regeneration, validate:

data/shakespeare_troll_sft_large.jsonl

The validator must report:

Examples: 2780
Maximum formatted length: <= 120
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

All 2780 examples must fit.

10. Preserve the 65-Character Vocabulary

The tokenizer is created from:

data/tiny_shakespeare.txt

Vocabulary size must remain:

65

Every character in instruction and response must exist in:

tokenizer.token_to_id

The final result must be:

Vocabulary size: 65
Unknown characters: 0

Do not change the tokenizer.

Previously problematic content included:

C++
HTTP 404
HTTP 500

Keep the established vocabulary-safe normalization:

CPP
HTTP four zero four
HTTP five zero zero

Do not introduce unsupported numeric or punctuation characters.

11. Fix the Training Data Path

The previous Kaggle run failed because scripts/train_sft.py expected:

data/shakespeare_troll_sft.jsonl

while the actual large dataset is:

data/shakespeare_troll_sft_large.jsonl

Fix the actual training configuration/script.

Run:

grep -Rni "shakespeare_troll_sft" .

The SFT training configuration must point to:

data/shakespeare_troll_sft_large.jsonl

not:

data/shakespeare_troll_sft.jsonl

Equivalent configuration:

config = SFTConfig(
pretrained_checkpoint="checkpoints/step_010000.pt",
dataset_path="data/shakespeare_troll_sft_large.jsonl",
checkpoint_directory="checkpoints/sft",
...
)

Do not create a fake duplicate under the old filename.

12. Make the Training Path Robust on Kaggle

The Kaggle project root is:

/kaggle/working/slm-from-scratch

Before training:

cd /kaggle/working/slm-from-scratch
pwd

Expected:

/kaggle/working/slm-from-scratch

Verify:

ls -lh data/shakespeare_troll_sft_large.jsonl
ls -lh checkpoints/step_010000.pt

Use repository-relative paths inside the project.

13. Do Not Hardcode Kaggle Input Paths

Do NOT hardcode:

/kaggle/input/...

inside project source code.

The project must work locally and on Kaggle.

Use:

data/shakespeare_troll_sft_large.jsonl
checkpoints/step_010000.pt
checkpoints/sft/

as repository-relative paths.

14. Verify the Dataset Before Training

Run:

python scripts/generate_shakespeare_troll_sft.py

Then validate:

JSONL valid ✓
2780 examples ✓
Vocabulary size = 65 ✓
Unknown characters = 0 ✓
Maximum formatted length <=120 ✓
Examples over 120 = 0 ✓
Examples over 128 = 0 ✓
No meta instructions ✓
No accidental C++ ✓
No unsupported numeric chars ✓
Technical facts remain correct ✓
Trolls are varied ✓
Factual answers are varied ✓

15. Verify InstructionDataset

Inspect one example:

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

The compact question/prompt must be masked.

The answer must not be masked.

Conceptually:

Q: What is AdamW?
A: AdamW uses adaptive gradients and decoupled weight decay...

Target:

[MASK][MASK][MASK]...
AdamW uses adaptive gradients...

Verify that the response remains intact and is not truncated.

16. Verify Effective Sequence Length

Use the exact formatting logic from InstructionDataset.

For every example:

formatted_length <= 120

must be true before training.

Do not rely on the dataset class to silently truncate invalid records.

17. Train From the Original Base Checkpoint

Do NOT continue SFT from the previous bad 300-step SFT checkpoint.

Start again from:

checkpoints/step_010000.pt

Training flow:

step_010000.pt
↓
corrected SFT dataset
↓
SFT
↓
checkpoints/sft/

The original base checkpoint must remain untouched.

18. Kaggle Training

Once validation passes:

cd /kaggle/working/slm-from-scratch
python -m scripts.train_sft

Confirm:

Device: cuda
GPU: Tesla T4
Vocabulary size: 65
Base checkpoint loaded:
Parameters: 4,782,336
SFT examples: 2780

19. Checkpoint Requirements

SFT checkpoints must be saved under:

checkpoints/sft/

For example:

checkpoints/sft/step_000100.pt
checkpoints/sft/step_000200.pt
checkpoints/sft/step_000300.pt

At minimum:

{
"model_state": ...,
"optimizer_state": ...,
"global_step": ...
}

Do not overwrite:

checkpoints/step_010000.pt

20. Do Not Reuse the Bad 300-Step SFT Checkpoint

The previous run used the old overlong dataset.

Treat:

checkpoints/sft/step_000300.pt

as an experimental artifact only.

Do not resume from it.

Start from:

checkpoints/step_010000.pt

after the dataset fixes are complete.

21. Evaluate Before Uploading

After training, test:

What is Python?
What is a database?
What is an API?
What is machine learning?
What is AdamW?

Desired behavior:

short technical explanation +
Shakespearean wording +
playful/witty troll

The model does not need to reproduce any example verbatim.

It should learn the behavioral pattern.

22. Compare Base vs SFT

Run the same prompts against:

Base:
checkpoints/step_010000.pt

and:

SFT:
checkpoints/sft/step_000300.pt

Compare:

Technical coherence

Instruction following

Shakespearean style

Troll behavior

Repetition

Randomness / degeneration

The SFT model should show a clear behavioral shift.

Do not judge solely from loss.

23. Generation Parameters

For evaluation, use consistent parameters, for example:

temperature=0.8
top_k=20
top_p=0.9

Use the same parameters for Base and SFT comparison.

Also test:

temperature=0.0

for deterministic behavior.

24. If the Model Is Still Incoherent

Do NOT immediately increase training steps.

Inspect in this order:

1. Dataset formatting
2. Dataset length
3. Response completeness
4. Loss masking
5. Training loss
6. Learning rate
7. Sampling parameters
8. Dataset diversity

Only after the pipeline is verified should training duration beincreased.

25. Final Acceptance Criteria

Dataset

2780 examples
65-character vocabulary
0 unknown characters
0 examples > 120 formatted characters
0 examples > 128 formatted characters
0 meta/generation instructions
technical facts remain correct
trolls are varied
factual answers are varied

Training

Base checkpoint loads successfully
CUDA/T4 is used on Kaggle
SFT dataset loads successfully
Loss is finite
Loss shows meaningful training behavior
SFT checkpoints are saved

Generation

The model should produce:

technical explanation +
Shakespearean style +
playful troll

rather than random text.

Paths

The SFT script must use:

data/shakespeare_troll_sft_large.jsonl
checkpoints/step_010000.pt
checkpoints/sft/

All paths must be repository-relative.

Important Constraints

Do NOT

change the pretrained architecture

change the tokenizer

change vocabulary size

modify checkpoints/step_010000.pt

silently truncate responses

delete examples just because they are long

replace the dataset with a tiny test dataset

hardcode /kaggle/input/...

train from the previous bad SFT checkpoint

add meta-instructions to response targets

use identical factual answers for every variation

use identical generic troll endings throughout the dataset

DO

modify the dataset generator

keep 2780 examples

create shorter technical answers

create shorter and more varied Shakespearean trolls

vary factual wording while preserving correctness

enforce <= 120 formatted characters

fail loudly on length violations

preserve the 65-character vocabulary

fix the actual SFT dataset path

validate the dataset before training

verify masking

train from checkpoints/step_010000.pt

save SFT checkpoints under checkpoints/sft/

compare Base vs SFT generation

only tune training duration after the pipeline is verified

Expected End-to-End Pipeline

data/tiny_shakespeare.txt
│
├── CharacterTokenizer
│ │
│ └── 65-character vocabulary
│
├── pretrained model
│ │
│ └── checkpoints/step_010000.pt
│
└── SFT generator
│
├── 2780 examples
├── technical answer
├── Shakespearean troll
├── varied factual wording
├── varied troll wording
└── <= 120 characters
│
↓
InstructionDataset
│
├── compact Q/A formatting
├── prompt masked
└── response trained
│
↓
SFT
│
↓
checkpoints/sft/
│
↓
Base vs SFT
evaluation

The agent must make the minimum necessary code changes and reportexactly which files were modified and why.
