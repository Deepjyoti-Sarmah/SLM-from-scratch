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
    * Examples use compact "Q: ...\\nA: ..." formatting so the whole
      example fits inside the model's 128-character context window.

Contract:
    formatted = "Q: " + instruction + "\\nA: " + response

    len(formatted) <= 120
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

# 120, not 128, leaves a small safety margin for the model context.
MAX_FORMATTED_LENGTH = 120

QUESTION_TEMPLATES = [
    "What is {name}?",
    "Explain {name}.",
    "How does {name} work?",
    "Why does {name} matter?",
    "When do I use {name}?",
    "Define {name}.",
    "What is {name} for?",
    "How is {name} used?",
    "Is {name} important?",
    "What does {name} do?",
]

TROLLS = [
    "Thy bug bows, forsooth.",
    "Thy compiler exhales.",
    "Thou art less lost.",
    "Mark it, good mortal.",
    "Thy code gains manners.",
    "Confusion quits the field.",
    "Thy logic finds shoes.",
    "The bug shakes in boots.",
    "Thy syntax smiles.",
    "Wisdom knocks at last.",
    "The stack grows calmer.",
    "Thy errors lose rank.",
    "Thy code hath hope.",
    "Reason wears the crown.",
    "Thy tests may cheer.",
    "Fog flees thy brow.",
    "Thy functions bow.",
    "Thy runtime survives.",
    "Thy cache grows smug.",
    "Thy question earns tea.",
]

WAR_TROLLS = [
    "Thy bug bows, forsooth.",
    "The gremlins flee.",
    "Thy duck grows wise.",
    "The stack grows calmer.",
    "Thy compiler exhales.",
    "Thy code gains manners.",
    "Confusion drops its crown.",
    "Thy logic finds shoes.",
    "The bug shakes in boots.",
    "Thy syntax smiles.",
    "Wisdom knocks at last.",
    "Thy errors lose rank.",
    "Thy code hath hope.",
    "Reason wins the duel.",
    "Thy tests may cheer.",
    "Fog flees thy brow.",
    "Thy functions bow.",
    "Thy runtime survives.",
    "Thy cache grows smug.",
    "Mark it, good mortal.",
]

WAR_STORIES = {
    "Why doth my code work only after I restart it?": "Restarting hides state or timing problems.",
    "Why is my bug invisible when I show it to someone?": "Bugs hide in timing or state.",
    "Why did my program become slower after I made it cleaner?": "Cleaner code may run slower.",
    "Can a rubber duck really debug code?": "Explaining aloud exposes assumptions.",
    "Why does the compiler hate me?": "The compiler only reports rule violations.",
    "Can I solve every problem with more threads?": "Threads add races, locks, and complexity.",
    "Should I rewrite everything because one function is ugly?": "Measure, then change the least.",
}

CONCEPT_FACTS = {
    "AdamW": "AdamW uses adaptive gradients and decoupled weight decay.",
    "BPE": "BPE merges frequent pairs into subword units.",
    "Big O notation": "Big O describes how cost grows with input size.",
    "CPP": "CPP is a compiled low level systems language.",
    "Docker": "Docker packages apps and dependencies into containers.",
    "Git": "Git is a distributed version control system.",
    "Go": "Go is a simple compiled language for services.",
    "HTTP": "HTTP is a protocol for web requests and responses.",
    "JSON": "JSON is a text format for structured data.",
    "JavaScript": "JavaScript is a language for web and server code.",
    "Kafka": "Kafka is a platform for streaming event records.",
    "Kubernetes": "Kubernetes orchestrates containers across machines.",
    "Python": "Python is a readable high level language.",
    "Python async": "Python async adds cooperative concurrency with await.",
    "Python inheritance": "Inheritance lets classes reuse behavior.",
    "RAG": "RAG adds retrieved facts before generation.",
    "RAM": "RAM is volatile memory for active programs.",
    "REST": "REST is an HTTP style for resource APIs.",
    "RLHF": "RLHF tunes models with human preferences.",
    "RLVR": "RLVR uses automatically verifiable rewards.",
    "Redis": "Redis is an in memory data store.",
    "Rust": "Rust is a fast memory safe systems language.",
    "SFT": "SFT fine tunes a pretrained model on examples.",
    "SQL": "SQL queries and manages relational databases.",
    "TypeScript": "TypeScript adds types to JavaScript.",
    "WebSocket": "WebSocket is a persistent two way connection.",
    "a CPU": "A CPU executes instructions and coordinates work.",
    "a CUDA out of memory error": "The GPU ran out of memory.",
    "a GPU": "A GPU has many cores for parallel work.",
    "a Git branch": "A Git branch is a movable line of commits.",
    "a Python class": "A Python class groups data and methods.",
    "a Python decorator": "A decorator wraps functions to change behavior.",
    "a Python dictionary": "A Python dictionary stores key value pairs.",
    "a Python generator": "A generator yields values lazily.",
    "a Python list": "A Python list is an ordered mutable collection.",
    "a Python set": "A Python set stores unique values.",
    "a Python tuple": "A Python tuple is immutable and ordered.",
    "a background worker": "Runs jobs off the main request path.",
    "a cache": "A cache keeps hot data close for speed.",
    "a checkpoint": "A checkpoint is a saved model snapshot.",
    "a coding benchmark": "A coding benchmark tests model coding skill.",
    "a compiler": "A compiler turns source code into machine code.",
    "a cron job": "A cron job runs commands on a schedule.",
    "a database": "A database stores and retrieves application data.",
    "a database index": "An index speeds up selected queries.",
    "a database migration": "A migration changes the database schema.",
    "a deadlock": "A deadlock is tasks waiting on one another.",
    "a dependency": "A dependency is code a project relies on.",
    "a distributed system": "Many networked machines work together as one.",
    "a file system": "A file system stores files and directories.",
    "a graph": "A graph holds nodes connected by edges.",
    "a hallucination": "A hallucination is plausible but false output.",
    "a hash table": "A hash table maps keys to slots via hashing.",
    "a knowledge graph": "A knowledge graph links entities and relations.",
    "a language model": "A language model predicts the next token.",
    "a linked list": "A linked list stores nodes in a chain.",
    "a memory leak": "A memory leak keeps memory forever unused.",
    "a message queue": "A message queue passes work to consumers.",
    "a microservice": "A microservice is one independently deployed service.",
    "a neural network": "A neural network is built from parameter layers.",
    "a package manager": "A package manager installs and tracks dependencies.",
    "a process": "A process is a program with its own memory.",
    "a pull request": "A pull request proposes code for review.",
    "a race condition": "A race condition depends on task timing.",
    "a stack trace": "A stack trace lists the calls before an error.",
    "a syntax error": "A syntax error breaks the language grammar.",
    "a thread": "A thread is a path of execution.",
    "a timeout": "A timeout means the operation ran too long.",
    "a token": "A token is one unit of model input.",
    "a transaction": "A transaction commits a group of operations.",
    "a transformer": "A transformer is an attention based network.",
    "a tree": "A tree stores data in connected nodes.",
    "a type error": "A type error uses an incompatible type.",
    "a vector database": "Stores vectors for similarity search.",
    "an AI agent": "An AI agent acts toward a goal using tools.",
    "an API": "An API lets software talk through interfaces.",
    "an HTTP five zero zero error": "The server failed internally.",
    "an HTTP four zero four error": "The resource was not found.",
    "an MCP server": "An MCP server exposes tools and resources.",
    "an ORM": "An ORM maps objects to database rows.",
    "an SSD": "An SSD stores data on flash memory.",
    "an embedding": "An embedding maps items to vectors.",
    "an environment variable": "A named value outside the program.",
    "an evaluation": "An evaluation scores a model on tasks.",
    "an interpreter": "An interpreter runs code through a runtime.",
    "an operating system": "An OS manages hardware and services.",
    "authentication": "Authentication verifies who you are.",
    "authorization": "Authorization controls what you may do.",
    "backpropagation": "Backpropagation sends error gradients backward.",
    "batch size": "Batch size is how many examples per step.",
    "binary search": "Binary search halves a sorted space.",
    "causal attention": "Causal attention hides future tokens.",
    "concurrency": "Concurrency lets tasks make progress together.",
    "context length": "Context length caps how many tokens a model sees.",
    "continuous integration": "CI auto builds and tests each change.",
    "cross entropy loss": "Cross entropy scores prediction error.",
    "debugging": "Debugging finds and fixes bugs.",
    "distillation": "Distillation shrinks a big model's skill.",
    "dynamic programming": "Stores results to avoid rework.",
    "encryption": "Encryption scrambles data with a key.",
    "event driven architecture": "A system driven by events.",
    "fine tuning": "Fine tuning adapts a model to a task.",
    "garbage collection": "Garbage collection frees unused memory.",
    "gradient descent": "Gradient descent lowers loss step by step.",
    "hashing": "Hashing turns data into a fixed digest.",
    "horizontal scaling": "Horizontal scaling adds more servers.",
    "instruction tuning": "Trains on instruction response pairs.",
    "integration testing": "Checks components work together.",
    "learning rate": "Learning rate sizes the parameter updates.",
    "load balancing": "Load balancing spreads requests across servers.",
    "machine learning": "Machine learning learns patterns from examples.",
    "next token prediction": "Predicts the next token from context.",
    "observability": "Observability uses logs, metrics, traces.",
    "overfitting": "Overfitting memorizes training data too well.",
    "parallelism": "Parallelism runs many tasks at once.",
    "quantization": "Quantization uses lower precision numbers.",
    "rate limiting": "Rate limiting caps how often a client acts.",
    "recursion": "Recursion calls a function on smaller inputs.",
    "reinforcement learning": "An agent learns by actions and rewards.",
    "self attention": "Self attention weighs related tokens.",
    "semantic versioning": "Version numbers signal compatibility.",
    "supervised learning": "Supervised learning trains on labeled data.",
    "temperature": "Temperature controls sampling randomness.",
    "tokenization": "Tokenization splits text into tokens.",
    "top k sampling": "Top k samples from the top k tokens.",
    "top p sampling": "Top p keeps tokens until probability sums.",
    "underfitting": "Underfitting is too weak a model.",
    "unit testing": "Unit testing checks code in isolation.",
    "unsupervised learning": "Finds structure without labels.",
    "validation loss": "Validation loss scores held out data.",
    "vertical scaling": "Vertical scaling upgrades one machine.",
    "weight decay": "Weight decay shrinks large parameters.",
}


VERB_PHRASES = [
    "uses",
    "merges",
    "describes",
    "packages",
    "orchestrates",
    "adds",
    "tunes",
    "queries",
    "executes",
    "coordinates",
    "groups",
    "wraps",
    "stores",
    "yields",
    "keeps",
    "runs",
    "turns",
    "tests",
    "speeds",
    "changes",
    "depends",
    "passes",
    "predicts",
    "maps",
    "verifies",
    "controls",
    "scores",
    "sends",
    "halves",
    "hides",
    "trains",
    "finds",
    "scrambles",
    "shrinks",
    "lowers",
    "caps",
    "weighs",
    "checks",
]


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def pronoun_fact_variant(fact: str) -> str:
    """Create a concise factual paraphrase without changing meaning."""
    if fact.startswith("Stores "):
        return "It stores " + lower_first(fact.removeprefix("Stores "))

    if fact.startswith("Runs "):
        return "It runs " + lower_first(fact.removeprefix("Runs "))

    if fact.startswith("Finds "):
        return "It finds " + lower_first(fact.removeprefix("Finds "))

    if fact.startswith("A system driven by events."):
        return "It is driven by events."

    if fact.startswith("Many networked machines work "):
        return "They work " + fact.removeprefix("Many networked machines work ")

    for verb in VERB_PHRASES:
        marker = f" {verb} "
        if marker in fact:
            return f"It {verb} {fact.split(marker, maxsplit=1)[1]}"

    return fact


def factual_variants(fact: str) -> list[str]:
    """Return short factual variants for repeated questions on a concept."""
    variants = [fact]

    pronoun_variant = pronoun_fact_variant(fact)
    if pronoun_variant not in variants:
        variants.append(pronoun_variant)

    brief_body = fact
    if fact.startswith(("A ", "An ", "The ", "Many ", "Stores ", "Runs ", "Finds ")):
        brief_body = lower_first(fact)

    brief_variant = "In brief, " + brief_body
    if brief_variant not in variants:
        variants.append(brief_variant)

    plain_variant = fact.removesuffix(".") + ", plainly."
    if plain_variant not in variants:
        variants.append(plain_variant)

    return variants


def select_response(
    *,
    instruction: str,
    fact: str,
    troll_start: int,
    variant_start: int,
    troll_pool: list[str],
) -> str:
    """Choose a varied factual answer and troll that fit the hard limit."""
    variants = factual_variants(fact)

    for variant_offset in range(len(variants)):
        variant = variants[(variant_start + variant_offset) % len(variants)]
        for troll_offset in range(len(troll_pool)):
            troll = troll_pool[(troll_start + troll_offset) % len(troll_pool)]
            response = f"{variant} {troll}"
            if len(format_example(instruction, response)) <= MAX_FORMATTED_LENGTH:
                return response

    raise ValueError(
        "No concise SFT response fits length limit for "
        f"instruction={instruction!r} fact={fact!r}"
    )


def build_records():
    """Build the full list of (instruction, response) records."""
    records = []

    for concept_index, (name, fact) in enumerate(sorted(CONCEPT_FACTS.items())):
        for template_index, template in enumerate(QUESTION_TEMPLATES):
            instruction = template.replace("{name}", name)

            records.append(
                (
                    instruction,
                    select_response(
                        instruction=instruction,
                        fact=fact,
                        troll_start=(concept_index + template_index * 2),
                        variant_start=template_index,
                        troll_pool=TROLLS,
                    ),
                )
            )
            records.append(
                (
                    instruction,
                    select_response(
                        instruction=instruction,
                        fact=fact,
                        troll_start=(concept_index + template_index * 2 + 1),
                        variant_start=(template_index + 1),
                        troll_pool=TROLLS,
                    ),
                )
            )

    for story_index, (question, fact) in enumerate(sorted(WAR_STORIES.items())):
        for troll_index in range(len(WAR_TROLLS)):
            response = select_response(
                instruction=question,
                fact=fact,
                troll_start=(story_index + troll_index),
                variant_start=troll_index,
                troll_pool=WAR_TROLLS,
            )
            records.append((question, response))

    return records


def format_example(
    instruction: str,
    response: str,
) -> str:
    return f"Q: {instruction}\nA: {response}"


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

        formatted = format_example(
            instruction=instruction,
            response=response,
        )

        if len(formatted) > MAX_FORMATTED_LENGTH:
            raise ValueError(
                "SFT example exceeds length limit: "
                f"{len(formatted)} > {MAX_FORMATTED_LENGTH}\n"
                f"{formatted!r}"
            )

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
    print(f"Max formatted length: {MAX_FORMATTED_LENGTH}")
    print(f"Wrote: {OUTPUT_PATH}")
    print(f"Unique instructions: {len(set(i for i, _ in records))}")
    print(f"Unique responses: {len(set(r for _, r in records))}")
    print("=" * 60)


if __name__ == "__main__":
    main()