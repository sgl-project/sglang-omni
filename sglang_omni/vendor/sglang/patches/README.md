# Bundled SGLang runtime changes

Omni carries the runtime changes it needs without waiting for their SGLang PRs to merge. The base is the immutable source revision in runtime.json, also pinned in pyproject.toml. The manifest records the ordered patch files, their hashes, and before/after hashes for changed files and the runtime interfaces Omni adapts. Installation rejects a mismatched base instead of applying a patch with offset guesses.

The GDN patch is the production-source delta from SGLang 9f6678aa to 31913fac, submitted as [SGLang #46619](https://github.com/sgl-project/sglang/pull/46619). It belongs to Omni #3234. It captures the existing GDN prefill chain and adds the FP32 checkpoint plumbing; it does not enable prefix caching.

Run sgl-omni-prepare-runtime after installing this branch. It copies the installed base into an environment-local directory keyed by source and manifest contents, applies the patches, verifies the result, and writes the environment's activation file. Existing installed or editable SGLang source is never rewritten. Fresh processes import the prepared package; a runtime check before ModelWorker initialization verifies the active module contents.

When the pinned SGLang source gains an equivalent implementation, remove the corresponding patch, update the manifest and dependency pin together, and repeat the runtime/quality checks. A package version string alone does not identify these source changes. Generate patch payloads from the reviewed SGLang commits and update the expected source/output hashes together.
