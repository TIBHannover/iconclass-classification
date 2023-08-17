#!/usr/bin/bash

TEST_PATH="/nfs/data/reflectai/iconclass/unify/brill/test/msg/"
TEST_ANNOTATION_PATH="/nfs/data/reflectai/iconclass/unify/brill/test.jsonl"
CLASSIFIER_PATH="/nfs/data/reflectai/iconclass/unify/classifiers.jsonl"
ONTOLOGY_PATH="/nfs/data/reflectai/iconclass/unify/ontology.jsonl"
MAPPING_PATH="/nfs/data/reflectai/iconclass/unify/mapping_trace_counts.jsonl"
PREDICTION_PATH="unify_brill_test_prediction_prob_chain_40000.h5"
RESULT_PATH="unify_brill_test_ap_prob_chain.jsonl"

MODEL_PATH="/nfs/home/springsteinm/output/reflectai/iconclass/"
OUTPUT_PATH="/data/iart/output/reflectai/iconclass/"
# clip

runs=( \
    unify_clip_fine_4x64_batch_4x64_packed_blib_instruction/230617/ \
)


for run in "${runs[@]}"
do	
    echo ${run}

    mkdir -p ${OUTPUT_PATH}/${run}/predictions/
    apptainer exec --bind /data:/data --env PYTHONPATH="..",TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" /nfs/home/springsteinm/env/iart_pytorch_1_11_0_220603.sif python test.py \
    --dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
    --model image_text_heads --heads clip --transformer_d_model 768 \
    --mapping_path ${MAPPING_PATH} \
    --classifier_path ${CLASSIFIER_PATH} \
    --test_path ${TEST_PATH} \
    --test_annotation_path ${TEST_ANNOTATION_PATH} \
    --use_probability_chain \
    --resume_from_checkpoint ${MODEL_PATH}/${run}/epoch=*-step=39999.ckpt \
    --prediction_path ${OUTPUT_PATH}/${run}/predictions/${PREDICTION_PATH} \
    --lr 1e-4 --local_loss --gather_with_grad --log_every_n_steps 1 \
    --train_size 224 \
    --use_center_crop \
    --val_size 224 \
    --batch_size 32 \
    --precision 16 \
    --gpus -1  \
    --encoder open_clip \
    --decoder transformer_level_wise \
    --ontology_path ${ONTOLOGY_PATH} \
    --train_merge_one_hot

    apptainer exec --bind /data:/data --env PYTHONPATH=. /nfs/home/springsteinm/env/iart_pytorch_1_11_0_220603.sif python tools/calculate_scores.py -f 0 10 100 1000 \
    --mapping_path ${MAPPING_PATH} \
    -p ${OUTPUT_PATH}/${run}/predictions/${PREDICTION_PATH} \
    -o ${OUTPUT_PATH}/${run}/predictions/${RESULT_PATH}
done
