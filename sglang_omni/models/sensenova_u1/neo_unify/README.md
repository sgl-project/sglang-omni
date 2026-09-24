# SenseNova-U1 Vendored Model Files

The Python files in this directory are adapted from:

- Repository: https://github.com/OpenSenseNova/SenseNova-U1
- Commit: 2f42002f9b819506c9deb44599f0809e30252aba
- License: Apache-2.0 (see the upstream LICENSE at this commit).

They were modified for SGLang multimodal_gen native integration.
For SGLang-Omni, the platform import was changed and the optional SRT
thinking fallback exception is kept local. The first pipeline uses native T2I
without SRT thinking or KV transfer.
