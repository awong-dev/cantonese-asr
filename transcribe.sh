#!/bin/bash
set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <model> <output_name> <audio_file> [extra whisperx args...]"
    echo ""
    echo "Example:"
    echo "  $0 whisper-large-v3-yue/final-ct2 my-run ~/1-yue.opus"
    echo "  $0 whisper-large-v3-yue/final-ct2 my-run ~/1-yue.opus --batch_size 8"
    exit 1
fi

MODEL="$1"
OUTPUT_NAME="$2"
AUDIO_FILE="$3"
shift 3

whisperx \
    --model "$MODEL" \
    --align_model ctl/wav2vec2-large-xlsr-cantonese \
    --language yue \
    --output_dir "out/$OUTPUT_NAME" \
    "$@" \
    "$AUDIO_FILE"
