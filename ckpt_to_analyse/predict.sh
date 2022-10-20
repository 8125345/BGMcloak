#!/bin/sh

PYTHON=python

NUM_STEP=150000
MODEL_NAME=train1
CHUNK_FRAMES=8
CHUNK_PADDING=3
DATASET_DIR=/deepiano_data/dataset
MODEL_DIR=./models/${MODEL_NAME}
TFLITE_MODEL_DIR=./tflite-models/${MODEL_NAME}/${NUM_STEP}-${CHUNK_FRAMES}
OUT_PREDICT_DIR=./out-predict/${MODEL_NAME}/${NUM_STEP}-${CHUNK_FRAMES}
OUT_PREDICT_CLIENT_DIR=./out-predict-client/${MODEL_NAME}/${NUM_STEP}-${CHUNK_FRAMES}
OUT_ANALYSER_DIR=./out-analyser/${MODEL_NAME}/${NUM_STEP}-${CHUNK_FRAMES}
OUT_ANALYSER_CLIENT_DIR=./out-analyser-client/${MODEL_NAME}/${NUM_STEP}-${CHUNK_FRAMES}
OUT_PREDICT_NOISE_DIR=./out-predict-noise/${MODEL_NAME}/${NUM_STEP}-${CHUNK_FRAMES}
OUT_PREDICT_CLIENT_NOISE_DIR=./out-predict-client-noise/${MODEL_NAME}/${NUM_STEP}-${CHUNK_FRAMES}


echo "export_tflite"
${PYTHON} tflite_export_tflite.py \
--model_dir ${MODEL_DIR}  \
--output_model_path ${TFLITE_MODEL_DIR}/exported_model.tflite \
--chunk_frames ${CHUNK_FRAMES}  &&
wait &&

cp ${MODEL_DIR}/model.ckpt-${NUM_STEP}.*  ${TFLITE_MODEL_DIR}
cp ${MODEL_DIR}/checkpoint  ${TFLITE_MODEL_DIR}

echo "single-note"
${PYTHON} tflite_predict_client.py \
--chunk_padding ${CHUNK_PADDING} \
--model_path ${TFLITE_MODEL_DIR}/exported_model.tflite  \
--input_dirs ${DATASET_DIR}/single-note   \
--output_dir ${OUT_PREDICT_DIR}/single-note \
--output_client_dir ${OUT_PREDICT_CLIENT_DIR}/single-note \
--output_json_dir ${OUT_PREDICT_CLIENT_DIR}/json-single-note  &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_CLIENT_DIR}/single-note \
--output_dir ${OUT_ANALYSER_CLIENT_DIR}/single-note &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/single-note \
--output_dir ${OUT_ANALYSER_DIR}/single-note &&
wait &&


echo "high-note"
${PYTHON} tflite_predict_client.py \
--chunk_padding ${CHUNK_PADDING} \
--model_path ${TFLITE_MODEL_DIR}/exported_model.tflite  \
--input_dirs ${DATASET_DIR}/high-note   \
--output_dir ${OUT_PREDICT_DIR}/high-note \
--output_client_dir ${OUT_PREDICT_CLIENT_DIR}/high-note \
--output_json_dir ${OUT_PREDICT_CLIENT_DIR}/json-high-note  &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_CLIENT_DIR}/high-note \
--output_dir ${OUT_ANALYSER_CLIENT_DIR}/high-note &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/high-note \
--output_dir ${OUT_ANALYSER_DIR}/high-note &&
wait &&


echo "ai-tagging"
${PYTHON} tflite_predict_client.py \
--chunk_padding ${CHUNK_PADDING} \
--model_path ${TFLITE_MODEL_DIR}/exported_model.tflite  \
--input_dirs ${DATASET_DIR}/ai-tagging   \
--output_dir ${OUT_PREDICT_DIR}/ai-tagging \
--output_client_dir ${OUT_PREDICT_CLIENT_DIR}/ai-tagging \
--output_json_dir ${OUT_PREDICT_CLIENT_DIR}/json-ai-tagging  &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_CLIENT_DIR}/ai-tagging \
--output_dir ${OUT_ANALYSER_CLIENT_DIR}/ai-tagging &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/ai-tagging \
--output_dir ${OUT_ANALYSER_DIR}/ai-tagging &&
wait &&


echo "maestro"
${PYTHON} tflite_predict_client.py \
--chunk_padding ${CHUNK_PADDING} \
--model_path ${TFLITE_MODEL_DIR}/exported_model.tflite  \
--input_dirs ${DATASET_DIR}/maestro-v3.0.0   \
--output_dir ${OUT_PREDICT_DIR}/maestro \
--output_client_dir ${OUT_PREDICT_CLIENT_DIR}/maestro \
--output_json_dir ${OUT_PREDICT_CLIENT_DIR}/json-maestro  &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_CLIENT_DIR}/maestro \
--output_dir ${OUT_ANALYSER_CLIENT_DIR}/maestro &&
wait &&

${PYTHON} tflite_analyser.py \
--input_dir ${OUT_PREDICT_DIR}/maestro \
--output_dir ${OUT_ANALYSER_DIR}/maestro &&
wait &&


echo "predict_client_noise"
${PYTHON} tflite_predict_client_noise.py \
--chunk_padding ${CHUNK_PADDING} \
--model_path ${TFLITE_MODEL_DIR}/exported_model.tflite  \
--input_dirs ${DATASET_DIR}/noise  \
--output_dir ${OUT_PREDICT_NOISE_DIR} \
--output_client_dir ${OUT_PREDICT_CLIENT_NOISE_DIR} \
--output_json_dir ${OUT_PREDICT_CLIENT_NOISE_DIR}/json &&
wait &&

echo "done"
