DEVICE=$1
MODEL_PATH=$2
CONTEXT=minimum
IFS='/' read -r MODEL_FAMILY MODEL_NAME <<< "$MODEL_PATH"
GENERATED_BY="${MODEL_PATH//\//--}" # replace / with --
TEMPERATURE=1.0
NUM_SAMPLES=25
MODE=independent-with-ref

# List of projects to run
PROJECTS=("PA19" "PA20" "PA21" "PA22")
IDS_FILE="../../constants/ids_train_val_test.json"
JSON_CONTENT=$(cat "$IDS_FILE")

start_time=$(date +%s)

for PROJECT in "${PROJECTS[@]}"; do
    echo "Running project $PROJECT..."
    CUDA_VISIBLE_DEVICES=$DEVICE python inference.py \
        --model-path "$MODEL_PATH" \
        --temperature "$TEMPERATURE" \
        --num-sample "$NUM_SAMPLES" \
        --project "$PROJECT" \
        --context "$CONTEXT" \
        --mode "$MODE" \
        --output generations/$MODEL_FAMILY--$MODEL_NAME--$MODE/${GENERATED_BY}-${PROJECT}-JavaBench.jsonl \
        --max-new-tokens 768 \
        --all_ids_dict "$JSON_CONTENT" \
        --split all # train, val, test, or all

done

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Total script duration: ${duration}s"
