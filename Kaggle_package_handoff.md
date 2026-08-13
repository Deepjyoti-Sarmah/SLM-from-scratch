Prepare Fresh TOPIC-Conditioned SFT Package for Kaggle

Objective

Prepare the current local project for a completely fresh Kaggle T4 SFT experiment.

Current dataset format:

TOPIC: AdamW
Q: What is AdamW?
A: AdamW uses adaptive gradients and decoupled weight decay. ...

The goal is ONLY:

Current local project
↓
Final validation
↓
Remove/exclude old SFT checkpoints
↓
Set fresh SFT experiment to 500 steps
↓
Create clean Kaggle staging directory
↓
Create ZIP
↓
Verify ZIP
↓
STOP

Important constraints

Do NOT run SFT locally.

Do NOT modify the GPT architecture.

Do NOT modify the tokenizer.

Do NOT modify:

checkpoints/step_010000.pt

1. Establish Repository Root

Run:

pwd
git status --short

The repository root should contain:

src/
scripts/
data/
checkpoints/
pyproject.toml

If not, change into the repository root.

2. Verify the TOPIC-Conditioned Dataset

Run:

ls -lh data/shakespeare_troll_sft_large.jsonl
wc -l data/shakespeare_troll_sft_large.jsonl
head -5 data/shakespeare_troll_sft_large.jsonl

Expected:

2780

The JSONL instruction should have this structure:

{
"instruction": "TOPIC: AdamW\nQ: What is AdamW?",
"response": "AdamW uses adaptive gradients and decoupled weight decay. ..."
}

The effective training text is:

TOPIC: {topic}
Q: {question}
A: {response}

Do NOT revert to the old non-TOPIC format.

3. Verify SFT Configuration

Run:

grep -nE 'pretrained_checkpoint|dataset_path|checkpoint_directory|max_steps|checkpoint_every' scripts/train_sft.py

Required:

pretrained_checkpoint = checkpoints/step_010000.pt
dataset_path = data/shakespeare_troll_sft_large.jsonl
checkpoint_directory = checkpoints/sft
max_steps = 500
checkpoint_every = 100

Do not use the previous 2000-step configuration for this first TOPIC-conditioned experiment.

4. Verify Base Checkpoint

Run:

ls -lh checkpoints/step_010000.pt

It must exist.

This is the ONLY starting checkpoint.

Do NOT modify, rename, replace, or continue from an old SFT checkpoint.

5. Validate Dataset

Run:

uv run python scripts/validate_sft_dataset.py

Expected:

Examples: 2780
Vocabulary size: 65
Unknown characters: 0
Maximum formatted length: 120
Examples over 120: 0
Examples over 128: 0
ALL CHECKS PASSED

The validation must understand:

TOPIC: ...
Q: ...
A: ...

If validation fails: STOP. Do not create the ZIP.

6. Run SFT Readiness Check

Run:

uv run python scripts/verify_sft_ready.py

Expected:

Dataset size: 2780
Input starts: 'TOPIC: AdamW\nQ: What is AdamW?'
Target starts: 'AdamW uses ...'
First non-masked target: 'A'
Loss: <finite value>

SFT READY: all checks passed

The important masking property is:

INPUT:
TOPIC: AdamW
Q: What is AdamW?
A: AdamW uses ...

TARGET:
AdamW uses ...

The first supervised target must be A. If it is d, :, a space, or anything else: STOP.

7. Verify Tokenizer

Run:

uv run python - <<'PY'
import json

from src.tokenization.char_tokenizer import CharacterTokenizer

with open("data/tiny_shakespeare.txt", encoding="utf-8") as f:
tokenizer = CharacterTokenizer(f.read())

unknown = {}
count = 0

with open("data/shakespeare_troll_sft_large.jsonl", encoding="utf-8") as f:
for line_number, line in enumerate(f, 1):
example = json.loads(line)
count += 1

        for field in ("instruction", "response"):
            for character in example[field]:
                if character not in tokenizer.token_to_id:
                    unknown.setdefault(character, []).append(line_number)

print("Examples:", count)
print("Vocabulary size:", tokenizer.vocab_size)
print("Unknown characters:", len(unknown))

for character, lines in unknown.items():
print(repr(character), "lines:", lines[:10])

assert count == 2780
assert tokenizer.vocab_size == 65
assert not unknown

print("Tokenizer validation PASSED")
PY

Expected:

Examples: 2780
Vocabulary size: 65
Unknown characters: 0
Tokenizer validation PASSED

Do not modify the tokenizer.

8. Run Full Test Suite

Run:

uv run pytest -q

Expected:

11 passed

If tests fail: STOP. Do not package.

9. Check for Old Dataset Paths

Run:

grep -RniE 'shakespeare_troll_sft\.jsonl' src scripts tests --exclude-dir=**pycache**

The active training dataset must be:

data/shakespeare_troll_sft_large.jsonl

Do not create a fake copy named shakespeare_troll_sft.jsonl.

Historical backup files outside the active training path may remain.

10. Verify TOPIC Conditioning

Run:

grep -Rni "TOPIC:" scripts src tests --exclude-dir=**pycache**
head -3 data/shakespeare_troll_sft_large.jsonl

Confirm the generated instructions contain:

TOPIC: ...
Q: ...

Do not allow the active dataset to revert to only:

Q: ...

11. Remove Old SFT Artifacts From the Package

Check:

ls -lah checkpoints/

If checkpoints/sft/ exists, preserve it locally if desired but do NOT package it.

Preferred:

mv checkpoints/sft checkpoints/sft_old

Or, if not needed:

rm -rf checkpoints/sft

NEVER delete:

checkpoints/step_010000.pt

12. Confirm Fresh Experiment Configuration

Run:

grep -nE 'pretrained_checkpoint|dataset_path|checkpoint_directory|max_steps|checkpoint_every' scripts/train_sft.py

Required:

pretrained_checkpoint = checkpoints/step_010000.pt
dataset_path = data/shakespeare_troll_sft_large.jsonl
checkpoint_directory = checkpoints/sft
max_steps = 500
checkpoint_every = 100

Do not change:

vocab_size = 65
max_sequence_length = 128
embedding_dim = 256
num_heads = 8
num_layers = 6

13. DO NOT TRAIN LOCALLY

Do NOT run:

uv run python scripts/train_sft.py

Training will happen on the Kaggle Tesla T4.

14. Create Clean Kaggle Staging Directory

From the repository root:

rm -rf kaggle_sft_package
mkdir -p kaggle_sft_package

cp -r src kaggle_sft_package/
cp -r scripts kaggle_sft_package/
cp -r data kaggle_sft_package/
cp -r checkpoints kaggle_sft_package/
cp pyproject.toml kaggle_sft_package/

If present:

cp uv.lock kaggle_sft_package/

15. Remove Old SFT Checkpoints From Staging

Run:

rm -rf kaggle_sft_package/checkpoints/sft
rm -rf kaggle_sft_package/checkpoints/sft_old

find kaggle_sft_package/checkpoints -type f -print

Expected:

kaggle_sft_package/checkpoints/step_010000.pt

There must be no old SFT checkpoints such as:

step_000100.pt
step_000200.pt
step_000300.pt
step_000500.pt
step_002000.pt

16. Remove Unnecessary Files

Run:

find kaggle_sft_package -type d -name "**pycache**" -prune -exec rm -rf {} +
find kaggle_sft_package -type f -name "*.pyc" -delete

Remove .git/ or .venv/ if accidentally copied.

Do NOT remove source, scripts, data, the base checkpoint, pyproject.toml, or uv.lock.

17. Verify Staging Directory

Run:

find kaggle_sft_package -maxdepth 3 -type f | sort

Important structure:

kaggle_sft_package/
├── checkpoints/
│ └── step_010000.pt
├── data/
│ ├── tiny_shakespeare.txt
│ └── shakespeare_troll_sft_large.jsonl
├── scripts/
│ ├── train_sft.py
│ ├── validate_sft_dataset.py
│ ├── verify_sft_ready.py
│ └── ...
├── src/
│ ├── configs/
│ ├── datasets/
│ ├── embeddings/
│ ├── inference/
│ ├── layers/
│ ├── models/
│ ├── tokenization/
│ └── training/
├── pyproject.toml
└── uv.lock

18. Verify Staged Dataset

Run:

wc -l kaggle_sft_package/data/shakespeare_troll_sft_large.jsonl
head -3 kaggle_sft_package/data/shakespeare_troll_sft_large.jsonl

Expected:

2780

The examples must still contain:

TOPIC:
Q:

19. Verify Staged Training Configuration

Run:

grep -nE 'pretrained_checkpoint|dataset_path|checkpoint_directory|max_steps|checkpoint_every' kaggle_sft_package/scripts/train_sft.py

It must point to:

checkpoints/step_010000.pt
data/shakespeare_troll_sft_large.jsonl
checkpoints/sft

and:

max_steps = 500
checkpoint_every = 100

20. Verify Staged Imports

Run:

cd kaggle_sft_package
PYTHONPATH=. python -c "
from src.models.gpt import GPT
from src.datasets.instruction_dataset import InstructionDataset
from src.tokenization.char_tokenizer import CharacterTokenizer
print('Imports OK')
"
cd ..

If this fails only because dependencies are unavailable in the local environment, do not modify source code just to fix the environment.

21. Check Package Size

Run:

du -sh kaggle_sft_package
du -h kaggle_sft_package/checkpoints/step_010000.pt

The large checkpoint size is expected.

22. Create ZIP

From the repository root:

rm -f slm-from-scratch-kaggle-sft.zip
zip -r slm-from-scratch-kaggle-sft.zip kaggle_sft_package

23. Verify ZIP Contents

Run:

unzip -l slm-from-scratch-kaggle-sft.zip

MUST contain:

kaggle_sft_package/checkpoints/step_010000.pt
kaggle_sft_package/data/tiny_shakespeare.txt
kaggle_sft_package/data/shakespeare_troll_sft_large.jsonl
kaggle_sft_package/scripts/train_sft.py
kaggle_sft_package/src/
kaggle_sft_package/pyproject.toml

MUST NOT contain:

kaggle_sft_package/checkpoints/sft/step_000100.pt
kaggle_sft_package/checkpoints/sft/step_002000.pt

24. Test ZIP Integrity

Run:

unzip -t slm-from-scratch-kaggle-sft.zip
ls -lh slm-from-scratch-kaggle-sft.zip

Expected:

No errors detected

25. Final Acceptance Checklist

All must be true:

Base checkpoint exists: checkpoints/step_010000.pt

Base checkpoint was not modified

Dataset has exactly 2780 examples

Dataset uses TOPIC conditioning

Dataset uses TOPIC: ...

Dataset uses Q: ...

Maximum formatted length <= 120

No examples > 120

No examples > 128

Vocabulary size = 65

Unknown characters = 0

First supervised target = A

Shifted target masking is correct

Model loss is finite

11 passed

Training starts from step_010000.pt

Training dataset is shakespeare_troll_sft_large.jsonl

Output directory is checkpoints/sft

max_steps = 500

checkpoint_every = 100

Old SFT checkpoints excluded

Clean staging directory created

ZIP created

ZIP integrity passes

ZIP contains base checkpoint

ZIP contains TOPIC-conditioned dataset

ZIP contains corrected source code

ZIP contains corrected training script

ZIP does not contain old SFT checkpoints

26. STOP

Once this exists and passes validation:

slm-from-scratch-kaggle-sft.zip

STOP.

Do NOT:

train locally

modify the base checkpoint

upload to Hugging Face

create another dataset

change the architecture

change the tokenizer

The next phase is separate:

Kaggle Dataset
↓
Extract package
↓
cd /kaggle/working/slm-from-scratch
↓
Verify GPU = Tesla T4
↓
Verify dataset = 2780
↓
Verify base checkpoint
↓
Run fresh SFT
↓
Evaluate 100/200/300/400/500 checkpoints
↓
Determine whether TOPIC conditioning improves concept selection

Experimental objective

The purpose of this run is not merely lower training loss.

The key question is whether:

TOPIC: AdamW
Q: What is AdamW?
A:

causes the model to generate an AdamW answer rather than an unrelated concept such as API.

The previous 5-example experiment produced:

AdamW → AdamW answer
Python → Python answer
API → API answer
database → database answer
recursion → recursion answer

That demonstrates that TOPIC conditioning worked in the small experiment, but does not prove the full 2780-example model will behave the same way.

Therefore the fresh 500-step Kaggle run is the next controlled experiment.
