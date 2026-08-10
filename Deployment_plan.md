## Make Deepjyoti/shakespeare-GPT Fully Runnable

Goal

Turn the currently published custom PyTorch Shakespeare GPT into aself-contained Hugging Face model repository.

A person who downloads the model should not need the original trainingrepository to run inference.

The final experience should be:

git clone <model-repository>
cd shakespeare-GPT
pip install -r requirements.txt
python inference.py --prompt "ROMEO:"

The implementation must preserve the existing trained checkpointarchitecture exactly. Do not redesign the GPT model in this task.

1. Current State

The training project is a custom PyTorch GPT implementation.

The trained model was successfully loaded locally with:

config = GPTConfig(
vocab_size=65,
max_sequence_length=128,
embedding_dim=256,
num_heads=8,
num_layers=6,
)

model = GPT(config=config)

result = model.load_state_dict(
checkpoint["model_state"],
strict=True,
)

The result was:

<All keys matched successfully>
Parameters: 4,782,336

The checkpoint was trained on Tiny Shakespeare.

Important checkpoint properties:

Vocabulary size: 65
Context length: 128
Embedding dimension: 256
Attention heads: 8
Transformer layers: 6
Parameters: 4,782,336
Training steps: 10,000

The exported local artifacts currently are:

artifacts/shakespeare-gpt/
├── tokenizer.json
├── model.pt
├── config.json
└── metadata.json

The Hugging Face repository is:

Deepjyoti/shakespeare-GPT

2. Current Training Project Structure

Use the existing codebase as the source of truth.

The project currently contains the following relevant areas:

slm-from-scratch/
│
├── data/
│ └── tiny_shakespeare.txt
│
├── scripts/
│ ├── demo_generation.py
│ ├── ...
│
├── src/
│ ├── configs/
│ │ ├── gpt_config.py
│ │ └── training_config.py
│ │
│ ├── tokenization/
│ │ └── char_tokenizer.py
│ │
│ ├── training/
│ │ ├── build.py
│ │ ├── scheduler.py
│ │ └── ...
│ │
│ └── models/
│ └── ... existing GPT implementation ...
│
├── train.py
├── main.py
├── pyproject.toml
└── uv.lock

Before modifying anything, the agent MUST inspect the actual repositoryand determine the exact locations and class names.

Do not assume the structure above is complete.

Run:

find src -maxdepth 4 -type f | sort

Then inspect:

sed -n '1,240p' src/configs/gpt_config.py

sed -n '1,240p' src/tokenization/char_tokenizer.py

Find the GPT implementation:

grep -R "class GPT" -n src

Find model construction:

grep -R "GPTConfig" -n src train.py scripts

Find generation code:

grep -R "generate" -n src scripts

The agent must use the existing implementation rather than recreating anapproximate architecture.

3. Important Constraint

Do NOT change the model architecture

The existing checkpoint contains tensors such as:

input_embedding.token_embedding.embedding.weight
input_embedding.position_embedding.embedding.weight

blocks.0.attention_norm.weight
blocks.0.multi_head_attention.qkv_projection.weight
blocks.0.multi_head_attention.output_projection.weight

blocks.0.feed_forward_norm.weight
blocks.0.feed_forward.input_projection.weight
blocks.0.feed_forward.output_projection.weight

...

final_norm.weight
final_norm.bias
lm_head.weight

The inference implementation must instantiate the same architecture.

The following must remain compatible:

vocab_size = 65
max_sequence_length = 128
embedding_dim = 256
num_heads = 8
num_layers = 6

Do not introduce:

RMSNorm

RoPE

GQA

SwiGLU

MoE

DeltaNet

MTP

different tokenizer

different positional embeddings

Those belong to the next CodeLM version, not this deployment task.

4. Deployment Architecture

The Hugging Face repository should become self-contained:

Deepjyoti/shakespeare-GPT/
│
├── README.md
├── config.json
├── metadata.json
├── tokenizer.json
├── model.safetensors
│
├── model.py
├── tokenizer.py
├── inference.py
└── requirements.txt

If the project requires additional source files to reconstruct themodel, include only the necessary inference files.

Avoid uploading the entire training project.

The distinction is:

Training repository
↓
contains everything required to TRAIN

Model repository
↓
contains everything required to RUN

5. Step 1 --- Inspect the Existing Model

The agent must first identify:

GPT class

GPTConfig class

embedding implementation

attention implementation

feed-forward implementation

transformer block implementation

generation implementation, if one already exists

tokenizer implementation

checkpoint loading implementation

Commands:

grep -R "class GPT" -n src
grep -R "class .*Attention" -n src
grep -R "class .*Feed" -n src
grep -R "class .*Block" -n src
grep -R "load_state_dict" -n .
grep -R "torch.load" -n .
grep -R "generate" -n src scripts

Read the complete relevant files before writing deployment code.

6. Step 2 --- Inspect the Existing Tokenizer

Current tokenizer:

class CharacterTokenizer:
def **init**(self, text: str) -> None:
vocabulary = sorted(set(text))

        self.token_to_id = {
            token: token_id
            for token_id, token in enumerate(vocabulary)
        }

        self.id_to_token = {
            token_id: token
            for token_id, token in enumerate(vocabulary)
        }

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id[token] for token in text]

    def decode(self, token_ids: Iterable[int]) -> str:
        return "".join(
            self.id_to_token[token_id]
            for token_id in token_ids
        )

This means the tokenizer is character-level.

The deployed tokenizer must preserve exactly the same mapping.

Do not regenerate the vocabulary from a different text file.

Do not sort a new vocabulary.

Do not change token IDs.

The tokenizer artifact must represent the exact mapping used duringtraining.

7. Step 3 --- Verify tokenizer.json

Inspect:

cat artifacts/shakespeare-gpt/tokenizer.json

Verify that it contains the 65-character vocabulary.

Create a deployment tokenizer implementation that can load the artifact:

tokenizer = CharacterTokenizer.from_file(
"tokenizer.json"
)

If from_file() does not exist, implement it.

The deployment tokenizer must support:

encode(text)
decode(token_ids)
vocab_size

Test:

text = "ROMEO:"

ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

assert decoded == text

Also verify:

assert tokenizer.vocab_size == 65

8. Step 4 --- Create a Standalone Model Implementation

Create:

model.py

This file should contain only the classes required for inference.

For example, depending on the actual source structure:

GPTConfig
GPT
TransformerBlock
MultiHeadAttention
FeedForward
InputEmbedding

Do not copy classes blindly.

Trace the imports and include only what is required.

The important requirement is:

model.load_state_dict(
checkpoint,
strict=True,
)

must succeed.

9. Step 5 --- Convert the Checkpoint

Current checkpoint:

checkpoints/step_010000.pt

contains:

{
"model_state": ...,
"optimizer_state": ...,
"scheduler_state": ...,
"epoch": ...,
"global_step": ...
}

Inference does not need:

optimizer_state
scheduler_state
epoch

Create a deployment-only weights file.

Preferred format:

model.safetensors

The conversion should extract:

checkpoint["model_state"]

and save only the model tensors.

Do not upload the optimizer state.

This reduces the artifact and removes unnecessary training state.

10. Step 6 --- Verify the Converted Weights

After conversion:

model = GPT(config)

state = load_model_weights(
"model.safetensors"
)

result = model.load_state_dict(
state,
strict=True,
)

print(result)

Expected:

<All keys matched successfully>

Then calculate:

parameters = sum(
p.numel()
for p in model.parameters()
if p.requires_grad
)

assert parameters == 4_782_336

If either assertion fails, STOP.

Do not upload the model until this is fixed.

11. Step 7 --- Create config.json

The configuration must contain enough information to reconstruct themodel.

Example:

{
"model_type": "shakespeare-gpt",
"vocab_size": 65,
"max_sequence_length": 128,
"embedding_dim": 256,
"num_heads": 8,
"num_layers": 6
}

The exact fields must match the actual GPTConfig.

Do not invent configuration fields.

12. Step 8 --- Create inference.py

The inference script must:

Load configuration

Load tokenizer

Build GPT

Load model weights

Select device

Encode prompt

Generate tokens

Decode tokens

Print result

Expected usage:

python inference.py --prompt "ROMEO:"

Optional arguments:

python inference.py \
--prompt "ROMEO:" \
--max-new-tokens 200 \
--temperature 0.8 \
--top-k 40

The exact supported generation parameters must match the existingmodel's generation implementation.

Do not add sampling algorithms that the current model does not supportunless they are implemented and tested.

13. Device Selection

The model should automatically select:

CUDA → CPU

Example behavior:

device = (
torch.device("cuda")
if torch.cuda.is_available()
else torch.device("cpu")
)

Then:

model.to(device)

The model must run on:

NVIDIA GPU with CUDA

CPU

The inference script must never assume CUDA exists.

14. CPU Compatibility Test

This is mandatory.

Run:

python inference.py \
--device cpu \
--prompt "ROMEO:"

The model should generate text.

This is the test that proves another person without an NVIDIA GPU canrun the model.

It will be slower, but it must work.

15. CUDA Compatibility Test

On a CUDA machine:

python inference.py \
--device cuda \
--prompt "ROMEO:"

Expected:

Device: cuda
GPU: <GPU name>

PROMPT:
ROMEO:

GENERATED:
...

16. Reproducibility Test

Run:

python inference.py \
--device cpu \
--prompt "ROMEO:" \
--seed 42

Run it again with the same seed.

The generated output should be reproducible if the samplingimplementation permits deterministic behavior.

If exact reproducibility cannot be guaranteed on a particular device,document that.

17. Add requirements.txt

Keep dependencies minimal.

Likely:

torch
safetensors
huggingface_hub

Do not add:

transformers
accelerate
datasets
trl
peft

unless the actual inference implementation needs them.

The goal is a lightweight custom PyTorch model.

Pin versions only if required for compatibility.

18. Create a download_and_run.py Convenience Script

For users who do not want to manually download files, optionallyprovide:

download_and_run.py

It should use:

from huggingface_hub import hf_hub_download

to retrieve:

model.safetensors
config.json
tokenizer.json

Then execute inference.

The preferred user experience becomes:

pip install -r requirements.txt

python inference.py \
--model-id Deepjyoti/shakespeare-GPT \
--prompt "ROMEO:"

The script should download missing files automatically.

19. Hugging Face Model Repository

Upload:

README.md
config.json
metadata.json
tokenizer.json
model.safetensors
model.py
tokenizer.py
inference.py
requirements.txt

Do not upload:

optimizer_state
scheduler_state
training checkpoints
uv.lock
.git
large training datasets

The Hugging Face repository is a model distribution package, not acopy of the entire training repository.

20. README Requirements

The Hugging Face README must explain:

Model

Shakespeare GPT
4.78M parameters
Character-level tokenizer
Tiny Shakespeare training corpus
128-token context

Architecture

6 Transformer blocks
256 embedding dimension
8 attention heads
character-level vocabulary of 65 tokens

Usage

Show:

pip install -r requirements.txt

python inference.py --prompt "ROMEO:"

CPU

Show:

python inference.py \
--device cpu \
--prompt "ROMEO:"

GPU

Show:

python inference.py \
--device cuda \
--prompt "ROMEO:"

Limitations

Explicitly state that this is:

a small experimental GPT

trained on Tiny Shakespeare

not a general-purpose language model

not a coding model

primarily a demonstration/research model

Do not claim capabilities it does not have.

21. Final Local Validation

Before pushing to Hugging Face, create a clean temporary environment.

For example:

python -m venv /tmp/shakespeare-test
source /tmp/shakespeare-test/bin/activate
pip install -r requirements.txt

Copy only:

model.py
tokenizer.py
inference.py
config.json
tokenizer.json
model.safetensors
requirements.txt

into the temporary directory.

Then run:

python inference.py --device cpu --prompt "ROMEO:"

This is critical.

It proves that inference does not accidentally depend on the originalrepository.

22. Final Clean-Room Test

The strongest test is:

Original training repository
│
X
│
X no imports
│
X no src/
│
▼
Hugging Face model repository
│
▼
pip install
│
▼
python inference.py
│
▼
generated Shakespeare text

If this works, the model is genuinely portable.

23. Optional Hugging Face from_pretrained() API

Do this only after the standalone inference path works.

Eventually we can implement:

from shakespeare_gpt import ShakespeareGPT

model = ShakespeareGPT.from_pretrained(
"Deepjyoti/shakespeare-GPT"
)

This can later be integrated with Hugging Face's custom PyTorch modelmechanisms.

Do not make this a prerequisite for the first deployment.

24. Tests to Add

Create:

tests/
├── test_tokenizer.py
├── test_model_loading.py
├── test_generation.py
└── test_cpu_inference.py

test_tokenizer.py

Verify:

vocab_size == 65
encode → decode roundtrip

test_model_loading.py

Verify:

all state_dict keys match
parameter count == 4,782,336

test_generation.py

Verify:

prompt is accepted
generated sequence has expected length
decoded output is valid text

test_cpu_inference.py

Verify:

model loads on CPU
generation completes

25. Agent Execution Rules

The coding agent performing this task MUST follow these rules.

Rule 1

Inspect the existing code before modifying it.

Rule 2

Do not recreate the GPT architecture from memory.

Rule 3

Do not change tensor names or architecture.

Rule 4

Do not retrain the model.

Rule 5

Do not change the tokenizer mapping.

Rule 6

Do not modify the training pipeline.

Rule 7

Keep deployment code separate from training code.

Rule 8

Every conversion must be verified with:

model.load_state_dict(..., strict=True)

Rule 9

Test CPU inference.

Rule 10

Test CUDA inference when available.

Rule 11

Do not upload a model until a clean-room inference test passes.

26. Definition of Done

The task is complete only when all of these are true:

[ ] Existing GPT implementation inspected
[ ] Existing tokenizer inspected
[ ] Deployment architecture matches checkpoint
[ ] tokenizer.json loads correctly
[ ] vocab_size == 65
[ ] model.safetensors created
[ ] model loads with strict=True
[ ] parameter count == 4,782,336
[ ] config.json reconstructs architecture
[ ] inference.py works
[ ] CPU inference works
[ ] CUDA inference works when available
[ ] requirements.txt works in a clean environment
[ ] Hugging Face repository contains inference code
[ ] README contains installation instructions
[ ] Clean-room test passes

27. After This Deployment Task

Do not immediately modify this Shakespeare model into the codingmodel.

Treat this as:

Shakespeare GPT
│
└── deployment baseline

Then create a separate development branch/version:

CodeLM v1

The coding-model roadmap should start with:

CharacterTokenizer
↓
BPE tokenizer
↓
clean code corpus
↓
deduplication
↓
code data packing
↓
RMSNorm + RoPE + SwiGLU
↓
GQA
↓
larger context
↓
MTP
↓
MoE
↓
hybrid attention
↓
executable coding tasks
↓
agent/tool training
↓
RL / verifiable rewards

Each architectural change must be implemented and benchmarkedindependently.

The Shakespeare deployment should remain frozen as the firstreproducible model release.
