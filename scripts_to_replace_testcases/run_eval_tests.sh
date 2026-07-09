# ./run_eval.sh
GENERATIONS_PATH=$1
OUTPUT_PATH="${GENERATIONS_PATH/test-4-execution/test-execution}"

python evaluation.py class-wise \
    --output $OUTPUT_PATH \
    $GENERATIONS_PATH