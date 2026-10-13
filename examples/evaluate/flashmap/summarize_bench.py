"""Helpers for bench_flashmap_vllm.sh.

``prompts SRC DST``: write a ``vllm bench serve`` custom dataset (one
``{"prompt": ...}`` per line) from chat-templated prompts, given either the same
way or as DeepSpec's ``{"conversations": [{"content": ...}, ...]}``.

``table DIR``: one row per benchmark JSON in DIR: output tok/s, mean TPOT and
ITL, mean TTFT, and the acceptance length (accepted drafts per draft, plus the
token the target always contributes), from the server's spec-decode counters
before and after the run.
"""

import json
import re
import sys
from pathlib import Path

_COUNTERS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)


def write_prompts(src: str, dst: str) -> None:
    with open(src) as f_in, open(dst, "w") as f_out:
        for line in f_in:
            row = json.loads(line)
            prompt = row.get("prompt") or row["conversations"][0]["content"]
            f_out.write(json.dumps({"prompt": prompt}) + "\n")


def read_counters(path: Path) -> dict[str, float]:
    counters = dict.fromkeys(_COUNTERS, 0.0)
    if not path.exists():
        return counters
    for line in path.read_text().splitlines():
        match = re.match(r"^([a-z_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if match and match.group(1) in counters:
            counters[match.group(1)] += float(match.group(3))
    return counters


def table(directory: str) -> None:
    rows = []
    for path in sorted(Path(directory).glob("*_c*.json")):
        result = json.loads(path.read_text())
        name = path.stem
        before = read_counters(path.with_name(f"{name}.metrics_before.txt"))
        after = read_counters(path.with_name(f"{name}.metrics_after.txt"))
        drafts = after[_COUNTERS[0]] - before[_COUNTERS[0]]
        accepted = after[_COUNTERS[2]] - before[_COUNTERS[2]]
        accept_len = 1 + accepted / drafts if drafts else float("nan")
        label, rest = name.split("_", 1)
        dataset, conc = rest.rsplit("_c", 1)
        rows.append(
            (
                label,
                dataset,
                int(conc),
                result.get("output_throughput", float("nan")),
                result.get("mean_tpot_ms", float("nan")),
                result.get("mean_itl_ms", float("nan")),
                result.get("mean_ttft_ms", float("nan")),
                accept_len,
                result.get("completed", 0),
            )
        )
    header = ("server", "dataset", "conc", "out tok/s", "TPOT ms", "ITL ms",
              "TTFT ms", "accept len", "reqs")
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in sorted(rows, key=lambda r: (r[1], r[2], r[0])):
        label, dataset, conc, tput, tpot, itl, ttft, alen, done = row
        print(
            f"| {label} | {dataset} | {conc} | {tput:.1f} | {tpot:.2f} | {itl:.2f} "
            f"| {ttft:.1f} | {alen:.2f} | {done} |"
        )


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "prompts":
        write_prompts(*args)
    elif command == "table":
        table(*args)
    else:
        raise SystemExit(f"unknown command {command!r}")
