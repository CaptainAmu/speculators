"""Helpers for bench_flashmap_vllm.sh.

``prompts SRC DST``: write a ``vllm bench serve`` custom dataset (one
``{"prompt": ...}`` per line) from chat-templated prompts, given either the same
way or as DeepSpec's ``{"conversations": [{"content": ...}, ...]}``.

``table DIR``: one row per benchmark point in DIR (its ``<name>.meta.json``
names the server, dataset and concurrency): output tok/s, the speedup over the
``target`` server at the same point, mean TPOT, ITL and TTFT, and the acceptance
length (accepted drafts per draft, plus the token the target always contributes)
from the server's spec-decode counters before and after the point.

``compare DIR [BASELINE]``: per point, how many responses are identical to the
BASELINE server's (default ``target``) and the mean fraction of the baseline
response that agrees before the first difference (at temperature 0 speculative
decoding is lossless up to floating-point differences between batch shapes).

``steps DIR``: the median GPU time of a draft step per server and batch size,
from the ``<server>.steps.jsonl`` files written with STEP_TIMING=1.
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
# Fewer timed steps at a batch size than this are not reported.
_MIN_STEPS = 20


def write_prompts(src: str, dst: str) -> None:
    with Path(src).open() as f_in, Path(dst).open("w") as f_out:
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


def points(directory: str):
    """``(meta, result, path)`` for every benchmark point with a result."""
    for meta_path in sorted(Path(directory).glob("*.meta.json")):
        name = meta_path.name.removesuffix(".meta.json")
        result_path = meta_path.with_name(f"{name}.json")
        if result_path.exists():
            meta = json.loads(meta_path.read_text())
            yield meta, json.loads(result_path.read_text()), result_path


def acceptance_length(result_path: Path) -> float:
    name = result_path.stem
    before = read_counters(result_path.with_name(f"{name}.metrics_before.txt"))
    after = read_counters(result_path.with_name(f"{name}.metrics_after.txt"))
    drafts = after[_COUNTERS[0]] - before[_COUNTERS[0]]
    accepted = after[_COUNTERS[2]] - before[_COUNTERS[2]]
    return 1 + accepted / drafts if drafts else float("nan")


def table(directory: str, baseline: str = "target") -> None:
    rows = [
        {
            **meta,
            "tput": result.get("output_throughput", float("nan")),
            "tpot": result.get("mean_tpot_ms", float("nan")),
            "itl": result.get("mean_itl_ms", float("nan")),
            "ttft": result.get("mean_ttft_ms", float("nan")),
            "accept": acceptance_length(path),
            "done": result.get("completed", 0),
        }
        for meta, result, path in points(directory)
    ]
    base = {
        (r["dataset"], r["concurrency"]): r["tput"]
        for r in rows
        if r["server"] == baseline
    }
    print(
        "| server | dataset | conc | out tok/s | speedup | TPOT ms | ITL ms "
        "| TTFT ms | accept len | reqs |"
    )
    print("|" + "---|" * 10)
    for r in sorted(rows, key=lambda r: (r["dataset"], r["concurrency"], r["server"])):
        reference = base.get((r["dataset"], r["concurrency"]))
        speedup = f"{r['tput'] / reference:.2f}x" if reference else "-"
        print(
            f"| {r['server']} | {r['dataset']} | {r['concurrency']} | {r['tput']:.1f} "
            f"| {speedup} | {r['tpot']:.2f} | {r['itl']:.2f} | {r['ttft']:.1f} "
            f"| {r['accept']:.2f} | {r['done']} |"
        )


def compare(directory: str, baseline: str = "target") -> None:
    texts = {
        (meta["server"], meta["dataset"], meta["concurrency"]): (
            result.get("generated_texts") or []
        )
        for meta, result, _ in points(directory)
    }
    print("| server | dataset | conc | identical | agreeing prefix |")
    print("|---|---|---|---|---|")
    for (server, dataset, conc), outputs in sorted(texts.items()):
        reference = texts.get((baseline, dataset, conc))
        if server == baseline or not reference:
            continue
        pairs = list(zip(reference, outputs))
        same = sum(a == b for a, b in pairs)
        prefix = []
        for a, b in pairs:
            n = next(
                (i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)),
            )
            prefix.append(n / max(len(a), 1))
        identical = f"{same}/{len(pairs)}"
        mean_prefix = sum(prefix) / max(len(prefix), 1)
        print(f"| {server} | {dataset} | {conc} | {identical} | {mean_prefix:.3f} |")


def steps(directory: str) -> None:
    print("| server | batch | draft step ms (median) | p90 | steps |")
    print("|---|---|---|---|---|")
    for path in sorted(Path(directory).glob("*.steps.jsonl")):
        by_reqs: dict[int, list[float]] = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            by_reqs.setdefault(row["reqs"], []).append(row["ms"])
        label = path.name.removesuffix(".steps.jsonl")
        for reqs in sorted(by_reqs):
            times = sorted(by_reqs[reqs])
            if len(times) < _MIN_STEPS:
                continue
            median, p90 = times[len(times) // 2], times[int(0.9 * len(times))]
            print(f"| {label} | {reqs} | {median:.3f} | {p90:.3f} | {len(times)} |")


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "prompts":
        write_prompts(*args)
    elif command == "table":
        table(*args)
    elif command == "compare":
        compare(*args)
    elif command == "steps":
        steps(*args)
    else:
        raise SystemExit(f"unknown command {command!r}")
