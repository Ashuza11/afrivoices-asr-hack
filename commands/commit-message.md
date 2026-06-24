---
description: Create a commit message by analyzing git diffs
allowed-tools: Bash(git status:*), Bash(git diff --staged), Bash(git commit:*)
---

## Context:

- Current git status: !`git status`
- Current git diff: !`git diff --staged`

Analyze the staged changes and create a commit message suited to an ML/ASR research project
(Python, Jupyter/Kaggle notebooks, HuggingFace Transformers, Whisper fine-tuning).
Use present tense and explain "why" something changed, not just "what".

## Commit types with emojis:

Only use the following emojis:

- ✨ `feat:` - New feature or notebook cell (new loader, new language, new pipeline step)
- 🐛 `fix:` - Bug fix (decode error, OOM crash, wrong path, broken cell)
- 🧪 `exp:` - Experiment change (different model, hyperparameters, training strategy)
- 📊 `data:` - Data loading or preprocessing change (new shard logic, audio decoding)
- 🚀 `perf:` - Performance or efficiency improvement (batch size, memory, speed)
- 📝 `docs:` - Documentation or README update
- 🔨 `refactor:` - Refactoring without behaviour change

## Format:

```
<emoji> <type>: <concise_description>
<optional_body_explaining_why>
```

## Output:

1. Show a short summary of what changed and which files/cells are affected
2. Propose a commit message with the appropriate emoji
3. Ask for confirmation before committing — DO NOT auto-commit
