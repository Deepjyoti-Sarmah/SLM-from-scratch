#!/usr/bin/env python3

"""Deterministic regeneration of the Shakespeare coding troll SFT dataset.

The generator is the source of truth for the cleaned dataset
"data/shakespeare_troll_sft_large.jsonl".

Key properties:
    * Only characters in the existing 65 character vocabulary are used.
    * No generation instructions leak into the responses.
    * Output is deterministic and reproducible.
    * CPP stands in for C++ and HTTP status codes are written as words,
      because "+", "5", "0", and "4" are not in the vocabulary.
"""

from __future__ import annotations

import json
import sys

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.tokenization.char_tokenizer import CharacterTokenizer


OUTPUT_PATH = Path("data/shakespeare_troll_sft_large.jsonl")

VOCABULARY_SOURCE = Path("data/tiny_shakespeare.txt")

QUESTION_TEMPLATES = [
    "What is {name}?",
    "Can you explain {name}?",
    "How does {name} work?",
    "Why does {name} matter?",
    "When would I use {name}?",
    "What should a beginner know about {name}?",
    "Give me a simple explanation of {name}.",
    "What is the practical use of {name}?",
    "What is the main idea behind {name}?",
    "Why would a programmer care about {name}?",
]

TROLLS = [
    "Thy confusion is understandable, yet even a sleepy scribe might have found this answer ere sunset.",
    "Thus is the mystery solved, though thy question arrived wearing a rather foolish hat.",
    "Now thou knowest the matter; go forth and trouble the compiler no further.",
    "A simple concept, truly, though thy question did make it dress itself in armor.",
    "There, the answer is given. May thy next question arrive with slightly less chaos.",
    "Take this knowledge and wield it wisely, lest the bugs mock thee in return.",
    "So ends the lesson. Even the court jester could now explain it with fewer errors.",
    "Remember it well, for forgetting this would be a tragedy worthy of the stage.",
    "Behold, thy answer. The machine hath survived thy question, and so have we.",
    "Now thou hast the knowledge. Try not to misuse it and blame the compiler afterward.",
]

WAR_STORIES = {
    "Why doth my code work only after I restart it?": "A restart can hide state, cache, timing, or initialization problems, so the real fix is to identify what changes between runs.",
    "Why is my bug invisible when I show it to someone?": "A bug may depend on timing, state, input, or environment, and changing those conditions can make it disappear.",
    "Why did my program become slower after I made it cleaner?": "Cleaner structure does not guarantee faster execution because abstraction, allocation, IO, or algorithmic changes can still add cost.",
    "Can a rubber duck really debug code?": "Explaining code aloud can expose assumptions and contradictions because it forces you to articulate the problem.",
    "Why does the compiler hate me?": "The compiler does not hate you; it reports violations of language rules, although its messages may feel personally judgmental.",
    "Can I solve every problem with more threads?": "No. Threads help with suitable concurrent work but can also introduce contention, races, synchronization costs, and complexity.",
    "Should I rewrite everything because one function is ugly?": "Usually no. First measure the problem, isolate the cost, and make the smallest change that improves correctness or maintainability.",
}

CONCEPT_FACTS = {
    "AdamW": "AdamW is an optimizer using adaptive gradients and decoupled weight decay.",
    "BPE": "Byte pair encoding builds a subword vocabulary by repeatedly merging frequent pairs.",
    "Big O notation": "Big O describes how algorithm cost grows as input size increases.",
    "CPP": "CPP is a compiled systems language offering low level control and powerful abstractions.",
    "Docker": "Docker packages applications and dependencies into containers.",
    "Git": "Git is a distributed version control system.",
    "Go": "Go is a compiled language designed for simplicity, fast builds, concurrency, and network services.",
    "HTTP": "HTTP is an application protocol used to exchange requests and responses.",
    "JSON": "JSON is a text format commonly used to exchange structured data.",
    "JavaScript": "JavaScript is a language widely used for web applications and server software.",
    "Kafka": "Kafka is a distributed event streaming platform for storing and processing records.",
    "Kubernetes": "Kubernetes manages deployment, scheduling, scaling, and networking of containers.",
    "Python": "Python is a high level language known for readable syntax and a large ecosystem.",
    "Python async": "Python async and await support cooperative concurrency for operations such as network IO.",
    "Python inheritance": "Inheritance lets one class reuse or extend behavior from another class.",
    "RAG": "Retrieval augmented generation supplies retrieved information to a model before generation.",
    "RAM": "RAM is volatile memory used by active programs.",
    "REST": "REST is an architectural style commonly used for HTTP resource APIs.",
    "RLHF": "RLHF uses human preferences to optimize model behavior.",
    "RLVR": "RLVR uses rewards that can be automatically verified.",
    "Redis": "Redis is an in memory data store used for caching, queues, counters, and streams.",
    "Rust": "Rust is a systems language focused on performance and memory safety through ownership.",
    "SFT": "Supervised fine tuning trains a pretrained model on labeled examples.",
    "SQL": "SQL is used to query and manipulate relational databases.",
    "TypeScript": "TypeScript adds static type checking and other features to JavaScript.",
    "WebSocket": "WebSocket provides a persistent connection for bidirectional messages.",
    "a CPU": "A CPU executes instructions and coordinates computation.",
    "a CUDA out of memory error": "A CUDA out of memory error means the GPU could not satisfy an allocation request.",
    "a GPU": "A GPU contains many execution resources suited to parallel operations.",
    "a Git branch": "A Git branch is a movable reference to a line of commits.",
    "a Python class": "A Python class defines a type and groups data and behavior into objects.",
    "a Python decorator": "A decorator wraps a function or class to add or change behavior.",
    "a Python dictionary": "A Python dictionary stores key value pairs and provides fast average lookup by key.",
    "a Python generator": "A generator produces values lazily, often with yield, instead of storing everything at once.",
    "a Python list": "A Python list is an ordered mutable collection.",
    "a Python set": "A Python set stores unique values and is useful for membership tests.",
    "a Python tuple": "A Python tuple is an ordered collection that cannot be changed after creation.",
    "a background worker": "A background worker processes asynchronous jobs outside the main request path.",
    "a cache": "A cache stores frequently needed data closer to the consumer for faster access.",
    "a checkpoint": "A checkpoint is a saved snapshot of model parameters and sometimes optimizer state.",
    "a coding benchmark": "A coding benchmark tests whether a model can solve programming tasks.",
    "a compiler": "A compiler transforms source code into another representation such as machine code.",
    "a cron job": "A cron job runs a command according to a schedule.",
    "a database": "A database stores and retrieves persistent application data.",
    "a database index": "A database index is a data structure that can speed selected queries at storage and write cost.",
    "a database migration": "A database migration is a controlled change to database schema or data.",
    "a deadlock": "A deadlock occurs when tasks wait on resources held by one another.",
    "a dependency": "A dependency is an external component that software relies upon.",
    "a distributed system": "A distributed system consists of multiple networked components cooperating on a service.",
    "a file system": "A file system organizes persistent data into files and directories.",
    "a graph": "A graph represents entities as nodes and relationships as edges.",
    "a hallucination": "A hallucination is generated content that sounds plausible but is incorrect or unsupported.",
    "a hash table": "A hash table maps keys to locations using a hash function.",
    "a knowledge graph": "A knowledge graph represents entities as nodes and relationships as edges.",
    "a language model": "A language model assigns probabilities to token sequences.",
    "a linked list": "A linked list stores nodes connected by references.",
    "a memory leak": "A memory leak occurs when a program retains memory it no longer needs.",
    "a message queue": "A message queue lets producers submit work while consumers process it asynchronously.",
    "a microservice": "A microservice is an independently deployable service focused on a particular responsibility.",
    "a neural network": "A neural network is a parameterized function built from layers.",
    "a package manager": "A package manager installs and resolves software dependencies.",
    "a process": "A process is a running program with its own virtual address space and resources.",
    "a pull request": "A pull request proposes code changes for review and integration.",
    "a race condition": "A race condition occurs when results depend on timing between concurrent operations.",
    "a stack trace": "A stack trace records the chain of calls that led to an error.",
    "a syntax error": "A syntax error means source code does not follow the language grammar.",
    "a thread": "A thread is an execution path within a process and commonly shares its memory.",
    "a timeout": "A timeout means an operation did not finish within an allowed period.",
    "a token": "A token is a discrete unit processed by a language model.",
    "a transaction": "A transaction groups database operations so they can be committed or rolled back together.",
    "a transformer": "A transformer is a neural architecture based heavily on attention.",
    "a tree": "A tree is a hierarchical structure of connected nodes.",
    "a type error": "A type error occurs when an operation receives an incompatible type.",
    "a vector database": "A vector database stores vector representations and supports similarity search.",
    "an AI agent": "An AI agent combines a model with tools, state, and control logic to pursue a goal.",
    "an API": "An API is a defined interface through which software components communicate.",
    "an HTTP five zero zero error": "An HTTP five zero zero error means the server encountered an internal problem.",
    "an HTTP four zero four error": "An HTTP four zero four error means the requested resource was not found.",
    "an MCP server": "An MCP server exposes tools or resources through the Model Context Protocol.",
    "an ORM": "An ORM maps application structures to database records.",
    "an SSD": "An SSD stores persistent data in flash memory.",
    "an embedding": "An embedding maps discrete items into continuous vectors.",
    "an environment variable": "An environment variable is a named value supplied by the operating environment.",
    "an evaluation": "An evaluation measures model behavior on defined tasks.",
    "an interpreter": "An interpreter executes program instructions through a runtime.",
    "an operating system": "An operating system manages hardware and provides services such as processes and files.",
    "authentication": "Authentication verifies who a user or service is.",
    "authorization": "Authorization determines what an authenticated user or service may do.",
    "backpropagation": "Backpropagation computes parameter gradients using the chain rule.",
    "batch size": "Batch size is the number of examples processed before an optimizer update.",
    "binary search": "Binary search repeatedly halves a sorted search space.",
    "causal attention": "Causal attention prevents a token from attending to future tokens.",
    "concurrency": "Concurrency lets multiple tasks make progress during overlapping periods.",
    "context length": "Context length is the maximum number of tokens considered in one sequence.",
    "continuous integration": "Continuous integration automatically builds and tests changes.",
    "cross entropy loss": "Cross entropy measures the mismatch between predicted probabilities and target classes.",
    "debugging": "Debugging is the process of finding and fixing the cause of incorrect behavior.",
    "distillation": "Knowledge distillation trains a smaller model to reproduce useful behavior from a larger model.",
    "dynamic programming": "Dynamic programming stores results of overlapping subproblems to avoid repeated work.",
    "encryption": "Encryption transforms data using a key so unauthorized parties cannot readily read it.",
    "event driven architecture": "Event driven architecture uses events to communicate state changes between components.",
    "fine tuning": "Fine tuning continues training a pretrained model for a particular task or behavior.",
    "garbage collection": "Garbage collection reclaims memory that is no longer reachable.",
    "gradient descent": "Gradient descent updates parameters in a direction that reduces loss.",
    "hashing": "Hashing maps data to a fixed size digest and is useful for lookup and integrity checks.",
    "horizontal scaling": "Horizontal scaling adds more service instances.",
    "instruction tuning": "Instruction tuning trains models on instructions paired with desired responses.",
    "integration testing": "Integration testing checks whether multiple components work correctly together.",
    "learning rate": "The learning rate controls the size of parameter updates.",
    "load balancing": "Load balancing distributes requests across workers or servers.",
    "machine learning": "Machine learning trains models from examples so they can make predictions on new inputs.",
    "next token prediction": "Next token prediction trains a model to predict the following token.",
    "observability": "Observability uses logs, metrics, and traces to understand running systems.",
    "overfitting": "Overfitting occurs when a model learns training data too closely and generalizes poorly.",
    "parallelism": "Parallelism means multiple computations execute at the same time.",
    "quantization": "Quantization represents values with lower precision to reduce memory and inference cost.",
    "rate limiting": "Rate limiting restricts how frequently a client can perform an operation.",
    "recursion": "Recursion solves a problem by calling the same function on smaller inputs.",
    "reinforcement learning": "Reinforcement learning trains an agent through actions and rewards.",
    "self attention": "Self attention lets each token use information from other relevant tokens.",
    "semantic versioning": "Semantic versioning uses major, minor, and patch numbers to communicate compatibility.",
    "supervised learning": "Supervised learning uses examples containing inputs and target labels or values.",
    "temperature": "Temperature changes the sharpness of probabilities during sampling.",
    "tokenization": "Tokenization converts text into a sequence of discrete tokens.",
    "top k sampling": "Top k sampling restricts choices to the k highest probability tokens.",
    "top p sampling": "Top p sampling keeps the smallest token set reaching a probability threshold.",
    "underfitting": "Underfitting occurs when a model is too limited or insufficiently trained.",
    "unit testing": "Unit testing checks small pieces of code in isolation.",
    "unsupervised learning": "Unsupervised learning finds structure without explicit target labels.",
    "validation loss": "Validation loss measures performance on held out data.",
    "vertical scaling": "Vertical scaling gives an existing machine more resources.",
    "weight decay": "Weight decay discourages excessively large model parameters.",
}

def build_records():
    """Build the full list of (instruction, response) records."""
    records = []

    for name, fact in sorted(CONCEPT_FACTS.items()):
        for template_index, template in enumerate(QUESTION_TEMPLATES):
            instruction = template.replace("{name}", name)
            first_troll = TROLLS[(template_index * 2) % len(TROLLS)]
            second_troll = TROLLS[(template_index * 2 + 1) % len(TROLLS)]
            records.append((instruction, f"{fact} {first_troll}"))
            records.append((instruction, f"{fact} {second_troll}"))

    for index, (question, fact) in enumerate(sorted(WAR_STORIES.items())):
        for troll_index in range(len(TROLLS) * 2):
            troll = TROLLS[troll_index % len(TROLLS)]
            records.append((question, f"{fact} {troll}"))

    return records


def normalize_against_vocabulary(
    text: str,
    tokenizer: CharacterTokenizer,
) -> str:
    unknown = [ch for ch in text if ch not in tokenizer.token_to_id]

    if unknown:
        raise ValueError(
            f"Unsupported characters found: {unknown!r}",
        )

    return text


def main() -> None:

    vocabulary = VOCABULARY_SOURCE.read_text(encoding="utf-8")

    tokenizer = CharacterTokenizer(vocabulary)

    records = build_records()

    lines = []

    for instruction, response in records:
        normalize_against_vocabulary(instruction, tokenizer)
        normalize_against_vocabulary(response, tokenizer)

        example = {
            "instruction": instruction,
            "response": response,
        }

        lines.append(json.dumps(example, ensure_ascii=False))

    OUTPUT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=" * 60)
    print("SHAKESPEARE TROLL SFT REGENERATION")
    print("=" * 60)
    print(f"Examples: {len(records)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Unique instructions: {len(set(i for i, _ in records))}")
    print(f"Unique responses: {len(set(r for _, r in records))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
