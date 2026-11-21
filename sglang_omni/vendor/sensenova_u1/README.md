# SenseNova-U1 Vendored Model Files

The Python files in this directory are adapted from:

- Repository: https://github.com/OpenSenseNova/SenseNova-U1
- Commit: 2f42002f9b819506c9deb44599f0809e30252aba

They were modified for SGLang multimodal_gen native integration.

This copy preserves the native T2I implementation from SGLang commit `e4d4a9a7de5454cd8b854b21af723c2155a55a3d`, under `python/sglang/multimodal_gen/runtime/models/sensenova_u1/neo_unify/`. Model computations match it, except that Think Mode recomputes the cached prefix length after the reasoning step so denoising attends to the reasoning, as sgl-project/sglang#47746 does upstream. Registration and execution adapters live in `sglang_omni.models.sensenova_u1.native`.

This copy drops code that no released SenseNova checkpoint and no supported Transformers release reaches: the conversation templates other than neo1_0, the deep flow-matching head, the dynamic time schedule, the attention backend setters, an offload hook that nothing sets, and the Transformers 4 compatibility seams.

Image editing in `image_edit.py` is adapted from SGLang-Omni #2983 at `026469802ec65e6b677e27491315d7f1602a8614`. It preserves reference conditioning, CFG branch selection and Euler denoising while calling the model components already shipped in this package. UMM transport, TP synchronization, cancellation and Cache-DiT integration are outside this editing implementation.

Reference resize budgets, 32-pixel alignment, Pillow interpolation and CUDA timestep/noise-scale embeddings follow the historical SGLang model implementation at `baa9befa381511cd4d56f83980951e5a45fffc6d`. Transparent inputs retain the white-background compositing from #2983. Output dimensions use the request or sampling defaults; the aspect-ratio inference from #2983 and SGLang #39081 is not included. Think Mode, CFG normalization and custom CFG windows remain unsupported for editing.
