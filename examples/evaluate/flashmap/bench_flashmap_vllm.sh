#!/bin/bash
# Throughput and latency of FlashMap in vLLM against the target alone and DFlash.
#
# For every server configuration in SERVERS this launches `vllm serve`, warms it
# up with a few full requests, runs `vllm bench serve` on each dataset at each
# concurrency (closed loop: the concurrency is the number of requests in flight,
# each point after two warmup requests of its own), saves the JSON results with
# the server's speculative-decoding counters, and stops the server.
# summarize_bench.py then prints one row per (server, dataset, concurrency):
# output tok/s, speedup over the `target` server, mean TPOT/ITL/TTFT and the
# acceptance length; and, with STEP_TIMING=1, the median GPU time of a draft
# step per batch size.
#
# Usage:
#   TARGET=Qwen/Qwen3-8B \
#   SERVERS="target=none dflash=dflash:z-lab/Qwen3-8B-DFlash-b16 flashmap=flashmap:./FlashMap-Qwen3-8B" \
#   DATASETS="gsm8k humaneval mt-bench" CONCURRENCY="1 4 16" \
#   PROMPT_DIR=./bench_prompts OUT=./runs/qwen3_8b \
#   bash examples/evaluate/flashmap/bench_flashmap_vllm.sh
#
# SERVERS entries are <label>=none | <label>=<method>:<draft>[:<env>=<value>...],
#   <method> one of dflash, flashmap. Trailing <env>=<value> pairs are exported
#   for that server only, e.g. fm2=flashmap:./FlashMap-Qwen3-8B:FLASHMAP_NUM_STEPS=2
#   or fm8=flashmap:./FlashMap-Qwen3-8B:NUM_SPEC=8 (verify 8 of the 16 drafts).
# Requests sample at TEMPERATURE with no top-k/top-p (--generation-config vllm
#   ignores the model's sampling defaults); above 0, drafts are sampled too
#   ("draft_sample_method": "probabilistic").
# PROMPT_DIR holds <dataset>.jsonl files of chat-templated prompts, either one
#   {"prompt": ...} per line or DeepSpec's {"conversations": [{"content": ...}]}.
# Optional: TEMPERATURE (0), MAX_TOKENS (1024), PROMPTS_PER_STREAM (8) and
#   MIN_PROMPTS (16): a point sends max(MIN_PROMPTS, PROMPTS_PER_STREAM x
#   concurrency) requests, NUM_SPEC (16; FlashMap verifies NUM_SPEC drafts,
#   DFlash NUM_SPEC - 1), GPU_UTIL (0.85), MAX_MODEL_LEN (8192), PORT (8123),
#   STEP_TIMING (0), EXTRA_SERVE_ARGS.

set -euo pipefail

TARGET="${TARGET:?set TARGET to the verifier model}"
SERVERS="${SERVERS:?set SERVERS}"
DATASETS="${DATASETS:-gsm8k humaneval mt-bench}"
CONCURRENCY="${CONCURRENCY:-1 4 16}"
PROMPT_DIR="${PROMPT_DIR:?set PROMPT_DIR}"
OUT="${OUT:-runs/flashmap_vllm}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
PROMPTS_PER_STREAM="${PROMPTS_PER_STREAM:-8}"
MIN_PROMPTS="${MIN_PROMPTS:-16}"
NUM_SPEC="${NUM_SPEC:-16}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
PORT="${PORT:-8123}"
STEP_TIMING="${STEP_TIMING:-0}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT/prompts"
for dataset in $DATASETS; do
    python "$HERE/summarize_bench.py" prompts \
        "$PROMPT_DIR/$dataset.jsonl" "$OUT/prompts/$dataset.jsonl"
done
{
    echo "$0 $*"
    env | grep -E '^(TARGET|SERVERS|DATASETS|CONCURRENCY|TEMPERATURE|MAX_TOKENS|PROMPTS_PER_STREAM|MIN_PROMPTS|NUM_SPEC|GPU_UTIL|MAX_MODEL_LEN|EXTRA_SERVE_ARGS)=' || true
} >> "$OUT/bench_command.txt"

SERVER_PID=""

run_server() {
    local label="$1" spec="$2"
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "a server already answers on port $PORT; stop it or set PORT" >&2
        return 1
    fi
    local -a args=(serve "$TARGET" --port "$PORT" --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_UTIL" --no-enable-prefix-caching
        --max-num-seqs 64 --generation-config vllm)
    local -a envs=()
    if [[ "$spec" != "none" ]]; then
        IFS=: read -r -a parts <<< "$spec"
        local method="${parts[0]}" draft="${parts[1]}"
        envs=("${parts[@]:2}")
        local num_spec=$NUM_SPEC
        for pair in "${envs[@]}"; do
            [[ "$pair" == NUM_SPEC=* ]] && num_spec="${pair#NUM_SPEC=}"
        done
        # A FlashMap block is all drafts; a DFlash block's first slot is the anchor.
        [[ "$method" == "dflash" ]] && num_spec=$(( num_spec - 1 ))
        # FlashMap runs on vLLM's DFlash runtime.
        local spec_json="{\"model\": \"$draft\", \"method\": \"dflash\""
        spec_json+=", \"num_speculative_tokens\": $num_spec"
        if [[ "$TEMPERATURE" != "0" ]]; then
            spec_json+=", \"draft_sample_method\": \"probabilistic\""
        fi
        spec_json+="}"
        args+=(--speculative-config "$spec_json")
    fi
    if [[ "$STEP_TIMING" == "1" ]]; then
        envs+=("PYTHONPATH=$HERE/step_timing${PYTHONPATH:+:$PYTHONPATH}"
            "FLASHMAP_STEP_TIMING=$OUT/$label.steps.jsonl")
    fi
    # shellcheck disable=SC2086
    env "${envs[@]}" vllm "${args[@]}" $EXTRA_SERVE_ARGS > "$OUT/$label.server.log" 2>&1 &
    SERVER_PID=$!
    until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "server $label died; see $OUT/$label.server.log" >&2
            SERVER_PID=""
            return 1
        fi
        sleep 2
    done
}

stop_server() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap stop_server EXIT
trap "exit 130" TERM INT

bench() {  # bench <dataset> <concurrency> <num prompts> <result dir> <result file>
    vllm bench serve --backend vllm --port "$PORT" --model "$TARGET" \
        --dataset-name custom --dataset-path "$OUT/prompts/$1.jsonl" \
        --skip-chat-template --custom-output-len "$MAX_TOKENS" \
        --num-prompts "$3" --max-concurrency "$2" --num-warmups 2 \
        --temperature "$TEMPERATURE" --request-rate inf \
        --save-result --save-detailed --result-dir "$4" --result-filename "$5"
}

for entry in $SERVERS; do
    label="${entry%%=*}"
    spec="${entry#*=}"
    echo "=== $label ($spec) ==="
    run_server "$label" "$spec" || continue
    # Warm pass: a few full requests through every captured path.
    warm_dataset="${DATASETS%% *}"
    bench "$warm_dataset" 4 8 "$OUT/warmup" "$label.json" > "$OUT/$label.warmup.log" 2>&1 \
        || echo "warm pass of $label failed" >&2
    for dataset in $DATASETS; do
        for conc in $CONCURRENCY; do
            name="${label}__${dataset}__c${conc}"
            prompts=$(( PROMPTS_PER_STREAM * conc ))
            prompts=$(( prompts > MIN_PROMPTS ? prompts : MIN_PROMPTS ))
            printf '{"server": "%s", "dataset": "%s", "concurrency": %d}\n' \
                "$label" "$dataset" "$conc" > "$OUT/$name.meta.json"
            curl -s "http://localhost:$PORT/metrics" > "$OUT/$name.metrics_before.txt" || true
            bench "$dataset" "$conc" "$prompts" "$OUT" "$name.json" > "$OUT/$name.log" 2>&1 \
                || echo "bench $name failed" >&2
            curl -s "http://localhost:$PORT/metrics" > "$OUT/$name.metrics_after.txt" || true
        done
    done
    stop_server
done

python "$HERE/summarize_bench.py" table "$OUT"
python "$HERE/summarize_bench.py" compare "$OUT"
if [[ "$STEP_TIMING" == "1" ]]; then
    python "$HERE/summarize_bench.py" steps "$OUT"
fi
