import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

root = Path(__file__).resolve().parent
config = json.loads((root / "config.json").read_text())
output = root / "outputs"
output.mkdir(exist_ok=True)
assert torch.cuda.is_available(), "CUDA is required"
assert torch.cuda.device_count() == 1, "This run uses exactly one granted GPU"
metadata = {
    "python": sys.version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu": torch.cuda.get_device_name(0),
    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "suites": [],
}
(output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
for suite in config["suites"]:
    directory = root / "code" / suite["source"]
    command = [sys.executable, "-m", "pytest", *suite["paths"], "-m", suite["marker"], "-q", "-ra", "--tb=short", "--junitxml=" + str(output / (suite["id"] + ".xml"))]
    started = time.time()
    print("START", suite["id"], flush=True)
    environment = dict(os.environ, PYTHONPATH=str(directory))
    with (output / (suite["id"] + ".log")).open("w") as log:
        result = subprocess.run(command, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT)
    metadata["suites"].append({"id": suite["id"], "returncode": result.returncode, "seconds": time.time() - started, "command": command})
    (output / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("END", suite["id"], result.returncode, flush=True)
raise SystemExit(any(suite["returncode"] for suite in metadata["suites"]))
