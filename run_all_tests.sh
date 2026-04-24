#!/bin/bash
#
# Run a progression of 6 Whisper fine-tuning experiments on Cantonese data.
#
# Test 1  — Baseline: khleeloo recipe (freeze encoder, freeze 12/32 decoder layers)
# Test 1b — Baseline with fp16 + 8-bit Adam (lower VRAM, check for quality parity)
# Test 2  — More decoder: same as baseline but freeze only 8 decoder layers
# Test 3 — Light encoder: unfreeze top 2 encoder layers with separate LR
# Test 4 — Combined: unfreeze top 2 encoder + more decoder (tests 2+3 together)
# Test 5 — LoRA: decoder-only LoRA with all decoder layers trainable
# Test 6 — Multi-dataset: baseline recipe but mixing yue + zh-HK + zh-CN data
#
# The progression isolates variables: tests 1-4 compare encoder/decoder unfreezing
# strategies, test 5 tests parameter-efficient fine-tuning, and test 6 tests
# whether adding Mandarin/HK data helps or hurts Cantonese CER.
#
# Usage:
#   ./run_all_tests.sh                    # run all 6 tests
#   ./run_all_tests.sh --tests 1,3,5      # run specific tests
#   ./run_all_tests.sh --skip-cleanup     # keep model dirs for inspection
#   ./run_all_tests.sh --tests 6 --skip-cleanup
#   ./run_all_tests.sh --tests 1 --steps upload,convert,transcribe  # skip training
#   ./run_all_tests.sh --tests 1 --steps train                      # train only

set -eu

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
TESTS_TO_RUN="1,1b,2,3,4,5,6"
SKIP_CLEANUP=false
STEPS="train,upload,convert,transcribe,cleanup"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tests)
            TESTS_TO_RUN="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --skip-cleanup)
            SKIP_CLEANUP=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--tests 1,3,5] [--steps train,upload,convert,transcribe,cleanup] [--skip-cleanup]"
            exit 1
            ;;
    esac
done

# Parse steps into a lookup
declare -A RUN_STEP
for s in train upload convert transcribe cleanup; do
    RUN_STEP[$s]=false
done
IFS=',' read -ra STEP_LIST <<< "$STEPS"
for s in "${STEP_LIST[@]}"; do
    s=$(echo "$s" | tr -d ' ')
    if [[ -z "${RUN_STEP[$s]+x}" ]]; then
        echo "Unknown step: $s (valid: train, upload, convert, transcribe, cleanup)"
        exit 1
    fi
    RUN_STEP[$s]=true
done

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
RESULTS_DIR="./test_results"
LOG_FILE="$RESULTS_DIR/run_all.log"
mkdir -p "$RESULTS_DIR"

# Tee all output to log file with timestamps
exec > >(while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S') $line"; done | tee -a "$LOG_FILE") 2>&1

echo "========================================"
echo "Starting test run"
echo "Tests to run: $TESTS_TO_RUN"
echo "Steps:        $STEPS"
echo "Skip cleanup: $SKIP_CLEANUP"
echo "Results dir:  $RESULTS_DIR"
echo "========================================"

# Track results
declare -A TEST_STATUS
declare -A TEST_TRANSCRIPT

# ---------------------------------------------------------------------------
# Common parameters
# ---------------------------------------------------------------------------
COMMON_ARGS=(
    --model openai/whisper-large-v3
    --dataset_path data/cv-corpus-25.0-2026-03-09/yue
    --all_tsv train.tsv
    --validation_tsv dev.tsv
    --holdback_tsv test.tsv
    --lr_schedule linear
    --warmup 1000
    --epochs 30
    --train_batch_size 16
    --eval_batch_size 32
    --eval_accumulation_steps 1
    --grad_accum 1
    --eval_steps 500
    --save_steps 500
    --nopunct_in_eval
    --best_model_metric cer_raw
    --streaming_eval
    --seed 42
)

# ---------------------------------------------------------------------------
# Helper: run one test
# ---------------------------------------------------------------------------
run_test() {
    local test_num="$1"
    local test_name="$2"
    local output_dir="$3"
    shift 3
    local test_args=("$@")

    local ct2_dir="${output_dir}-ct2"
    local transcript_file="$RESULTS_DIR/test_${test_name}_transcript.txt"

    echo ""
    echo "========================================"
    echo "TEST $test_num — $test_name"
    echo "  output_dir: $output_dir"
    echo "========================================"

    # ---- Train ----
    if [ "${RUN_STEP[train]}" = true ]; then
        echo "[test $test_num] Training..."
        python3 train_whisper.py "${COMMON_ARGS[@]}" "${test_args[@]}" --output_dir "$output_dir"
        echo "[test $test_num] Training succeeded"
    fi

    # ---- Upload ----
    if [ "${RUN_STEP[upload]}" = true ]; then
        echo "[test $test_num] Uploading model..."
        python3 upload_model.py \
            --model_path "$output_dir/final" \
            --repo_name "$(basename "$output_dir")" <<< "y"
        echo "[test $test_num] Upload succeeded"
    fi

    # ---- Convert to CT2 ----
    if [ "${RUN_STEP[convert]}" = true ]; then
        echo "[test $test_num] Converting to CT2..."
        python3 convert_ct2.py --model "$output_dir/final" --output_dir "$ct2_dir"
        echo "[test $test_num] CT2 conversion succeeded"
    fi

    # ---- Transcribe ----
    if [ "${RUN_STEP[transcribe]}" = true ]; then
        echo "[test $test_num] Transcribing..."
        local transcribe_out_dir="$RESULTS_DIR/transcribe_${test_name}"
        ./transcribe.sh "$ct2_dir" "test_${test_name}" ~/1-yue.opus
        cp -r "out/test_${test_name}" "$transcribe_out_dir"
        echo "[test $test_num] Transcription saved to $transcribe_out_dir"
        TEST_TRANSCRIPT[$test_num]="$transcribe_out_dir"
    fi

    # ---- Cleanup ----
    if [ "${RUN_STEP[cleanup]}" = true ] && [ "$SKIP_CLEANUP" = false ]; then
        echo "[test $test_num] Cleaning up..."
        ./clean-cache.sh
        rm -rf "$output_dir" "$ct2_dir"
        rm -rf "out/test_${test_name}"
    fi

    TEST_STATUS[$test_num]="OK"
    echo "[test $test_num] Done"
}

# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------
test_1() {
    run_test 1 "baseline" "./whisper-large-v3-yue-test1-baseline" \
        --freeze_encoder --freeze_decoder_layers 12 --lr 5e-8
}

test_1b() {
    run_test 1b "baseline-fp16-8bit" "./whisper-large-v3-yue-test1b-baseline-fp16-8bit" \
        --freeze_encoder --freeze_decoder_layers 12 --lr 5e-8 \
        --fp16 --optim adamw_bnb_8bit
}

test_2() {
    run_test 2 "dec8" "./whisper-large-v3-yue-test2-dec8" \
        --freeze_encoder --freeze_decoder_layers 8 --lr 5e-8
}

test_3() {
    run_test 3 "enc2" "./whisper-large-v3-yue-test3-enc2" \
        --unfreeze_encoder_layers 2 --encoder_lr 5e-8 --freeze_decoder_layers 12 --lr 5e-8 \
        --train_batch_size 8 --grad_accum 2
}

test_4() {
    run_test 4 "combined" "./whisper-large-v3-yue-test4-combined" \
        --unfreeze_encoder_layers 2 --encoder_lr 5e-8 --freeze_decoder_layers 8 --lr 5e-8 \
        --train_batch_size 8 --grad_accum 2
}

test_5() {
    run_test 5 "lora" "./whisper-large-v3-yue-test5-lora" \
        --freeze_encoder --freeze_decoder_layers 0 \
        --lora --lora_r 16 --lora_alpha 32 --lora_target decoder --lora_merge_on_save \
        --lr 1e-6
}

test_6() {
    run_test 6 "mix" "./whisper-large-v3-yue-zh-test6-mix" \
        --dataset_path data/cv-corpus-25.0-2026-03-09/yue,data/cv-corpus-25.0-2026-03-09/zh-HK,data/cv-corpus-25.0-2026-03-09/zh-CN \
        --all_tsv train.tsv,train.tsv,train.tsv \
        --validation_tsv dev.tsv,dev.tsv,dev.tsv \
        --holdback_tsv test.tsv,test.tsv,test.tsv \
        --dataset_ratio 3:2:1 \
        --freeze_encoder --freeze_decoder_layers 12 --lr 5e-8
}

# ---------------------------------------------------------------------------
# Run selected tests
# ---------------------------------------------------------------------------
IFS=',' read -ra SELECTED <<< "$TESTS_TO_RUN"
for t in "${SELECTED[@]}"; do
    t=$(echo "$t" | tr -d ' ')
    if declare -f "test_$t" > /dev/null 2>&1; then
        "test_$t"
    else
        echo "Unknown test: $t (skipping)"
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
echo "SUMMARY"
echo "========================================"
for t in "${SELECTED[@]}"; do
    t=$(echo "$t" | tr -d ' ')
    status="${TEST_STATUS[$t]:-NOT RUN}"
    transcript="${TEST_TRANSCRIPT[$t]:-none}"
    printf "  Test %-2s  %-25s  transcript: %s\n" "$t" "$status" "$transcript"
done
echo "========================================"
echo "Log: $LOG_FILE"
echo "Done."
