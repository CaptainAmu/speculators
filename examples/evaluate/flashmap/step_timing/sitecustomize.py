"""Times every draft step of vLLM's DFlash-family speculators (FlashMap, DFlash).

bench_flashmap_vllm.sh puts this directory on the server's PYTHONPATH when
STEP_TIMING=1, so Python imports it at startup. Once vLLM's DFlash speculator
module is imported, its ``propose`` is wrapped with CUDA events, and every 256
steps the GPU time of each step is appended to ``$FLASHMAP_STEP_TIMING`` as JSON
lines ``{"reqs": <batch size>, "ms": <time>}`` (which synchronizes the host with
the GPU then, so it is for measuring only).
"""

import importlib.abc
import importlib.util
import json
import os
import sys
from pathlib import Path

_MODULE = "vllm.v1.worker.gpu.spec_decode.dflash.speculator"
_FLUSH = 256


def _patch(module) -> None:
    import torch  # noqa: PLC0415

    speculator = module.DFlashSpeculator
    propose = speculator.propose
    path = Path(os.environ["FLASHMAP_STEP_TIMING"])
    pending: list = []

    def timed_propose(self, input_batch, *args, **kwargs):
        if kwargs.get("dummy_run") or torch.cuda.is_current_stream_capturing():
            return propose(self, input_batch, *args, **kwargs)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = propose(self, input_batch, *args, **kwargs)
        end.record()
        pending.append((input_batch.num_reqs, start, end))
        if len(pending) >= _FLUSH:
            end.synchronize()
            with path.open("a") as f:
                for reqs, s, e in pending:
                    f.write(json.dumps({"reqs": reqs, "ms": s.elapsed_time(e)}) + "\n")
            pending.clear()
        return result

    speculator.propose = timed_propose


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):  # noqa: ARG002
        if name != _MODULE:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if os.environ.get("FLASHMAP_STEP_TIMING"):
    sys.meta_path.insert(0, _Finder())
