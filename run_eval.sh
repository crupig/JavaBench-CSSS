GENERATIONS_PATH=$1
OUTPUT_PATH="${GENERATIONS_PATH/generations/generations-tested}"

python evaluation.py class-wise \
    --output $OUTPUT_PATH \
    $GENERATIONS_PATH