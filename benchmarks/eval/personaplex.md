# PersonaPlex reference evaluation

These optional scripts require CUDA and real checkpoints. They are evaluation
entry points, not pytest tests. Run them from the repository root. Install the
reference checkout into its own Python environment because its torch constraint
differs from the serving runtime.

## Greedy comparison and reproducibility

```bash
python -m benchmarks.eval.personaplex_parity \
    --reference-source /path/to/personaplex \
    --reference-python /path/to/reference-env/bin/python \
    --checkpoint /path/to/personaplex-checkpoint \
    --output-dir /path/to/reference-results
```

`--checkpoint` accepts a local directory or a model identifier supported by the
shared checkpoint resolver. Both implementations receive the same resolved weights,
tokenizer, packaged voice and caller recording. `--stage-args` forwards pipeline
overrides, such as `'--lm.engine.mem_fraction_static 0.5'`. `--atol` controls the
per-sample tolerance (default `1e-4`). `--reference-repo` only supplies the reference
CLI's config lookup; actual weights are passed as local paths.

The assistant and service cases require the complete input sample count, then
check at least 100 matching leading frames and text agreement over that prefix.
**Passing these checks does not establish full-output parity or answer quality.**
The script also checks greedy repetition and same-seed/different-seed behavior.

Every run regenerates the reference output and writes its log to the result
directory. There is no reference-output cache or mandatory revision/hash check.
Record the source and checkpoint versions with published measurements.

## Component comparison

```bash
python -m benchmarks.eval.personaplex_components \
    --reference-source /path/to/personaplex \
    --reference-python /path/to/reference-env/bin/python \
    --checkpoint /path/to/moshiko-pytorch-bf16-checkpoint \
    --dump /path/to/results/moshi-reference.safetensors
```

This uses the public `kyutai/moshiko-pytorch-bf16` base checkpoint, not the
PersonaPlex fine-tune. It compares Mimi encoding/decoding, input embeddings and
teacher-forced depformer logits in FP32/BF16. TF32 and cuDNN autotuning are disabled.
The final depformer step deliberately diagnoses the reference ring difference;
its emulated result is not proof that the unmodified implementations match.

`personaplex_reference_dump.py` runs under the reference interpreter and regenerates
the dump on every invocation. Large dumps, checkpoints and generated audio belong
in external result directories.
