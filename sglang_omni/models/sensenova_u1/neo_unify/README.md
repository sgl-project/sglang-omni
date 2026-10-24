# SenseNova-U1 Vendored Model Files

The Python files in this directory are adapted from:

- Repository: https://github.com/OpenSenseNova/SenseNova-U1
- Commit: 2f42002f9b819506c9deb44599f0809e30252aba
- License: Apache-2.0 (see the upstream LICENSE at this commit).

They were modified for SGLang multimodal_gen native integration.
For SGLang-Omni, the platform import was changed. The pipeline runs native
T2I and I2I only; thinking, interleaved output, and SRT KV transfer are not
included.

The implementation requires the Transformers version pinned in `pyproject.toml`.
Checkpoints load through registered `AutoModel` classes and tensor state
dictionaries. Legacy Python import aliases and pickled model objects are outside
the supported checkpoint format.
