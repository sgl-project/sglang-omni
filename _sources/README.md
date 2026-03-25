# SGLang-Omni Documentation

We recommend new contributors to start from writing documentation, which helps you quickly understand the SGLang-Omni codebase.
Most documentation files are located under the `docs/` folder.

## Docs Workflow

### Install Dependency

```bash
apt-get update && apt-get install -y pandoc parallel retry
pip install -r requirements.txt
```

### Update Documentation

Update your Jupyter notebooks in the appropriate subdirectories under `docs/`. If you add new files, remember to update `index.rst` (or relevant `.rst` files) accordingly.

- **`pre-commit run --all-files`** manually runs all configured checks, applying fixes if possible. If it fails the first time, re-run it to ensure lint errors are fully resolved. Make sure your code passes all checks **before** creating a Pull Request.

```bash
bash serve.sh

# if you want to ues a custom port
PORT=8080 make serve
```
---

## Documentation Style Guidelines

- For common functionalities, we prefer **Jupyter Notebooks** over Markdown so that all examples can be executed and validated by our docs CI pipeline. For complex features (e.g., distributed serving), Markdown is preferred.
- Keep in mind the documentation execution time when writing interactive Jupyter notebooks. Each interactive notebook will be run and compiled against every commit to ensure they are runnable, so it is important to apply some tips to reduce the documentation compilation time:
  - Use small models (e.g., `qwen/qwen2.5-0.5b-instruct`) for most cases to reduce server launch time.
  - Reuse the launched server as much as possible to reduce server launch time.
- Do not use absolute links (e.g., `https://docs.sglang-omni.ai/get_started/install.html`). Always prefer relative links (e.g., `../get_started/install.md`).
- Follow the existing examples to learn how to launch a server, send a query and other common styles.
