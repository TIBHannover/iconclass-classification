#!/usr/bin/bash

TEST_PATH="/nfs/data/iart/iconclass/data_202002/packed/test/msg/"
TEST_ANNOTATION_PATH="/nfs/data/reflectai/iconclass/unify/iconclass/test.jsonl"
CLASSIFIER_PATH="/nfs/data/reflectai/iconclass/unify/classifiers.jsonl"
ONTOLOGY_PATH="/nfs/data/reflectai/iconclass/unify/ontology.jsonl"
MAPPING_PATH="/nfs/data/reflectai/iconclass/unify/mapping_trace_counts.jsonl"
PREDICTION_PATH="unify_iconclass_test_prediction_prob_chain_40000.h5"
RESULT_PATH="unify_iconclass_test_ap_prob_chain.jsonl"

MODEL_PATH="/nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/"
OUTPUT_PATH="/data/iart/output/reflectai/iconclass_wacv_round_2/"
# clip

runs=( \
    # 4x64_clip_txt_fine_onto/v1/ \
    4x64_clip_txt_300_fine_onto/v1 \
    4x64_clip_kw_fine_onto/v1 \
    4x64_clip_txt_fine_onto/v1 \
    4x64_fine_onto/v1 \
    4x64_clip_blip_fine_onto/v1 \
    4x64_clip_gpt_fine_onto/v \
)


for run in "${runs[@]}"
do	
    echo ${run}

    mkdir -p ${OUTPUT_PATH}/${run}/predictions/
    apptainer exec --bind /data:/data --env PYTHONPATH=".." /nfs/home/springsteinm/env/iart_pytorch_1_11_0_220603.sif python test.py \
    --dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
    --model image_text_heads --heads ontology --transformer_d_model 768 \
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


runs=( \
    4x64_clip_txt_300_fine_onto/v1 \
    4x64_clip_kw_fine_onto/v1 \
    4x64_clip_txt_fine_onto/v1 \
    4x64_fine_onto/v1 \
    4x64_clip_blip_fine_onto/v1 \
    4x64_clip_gpt_fine_onto/v \
)


for run in "${runs[@]}"
do	
    echo ${run}

    mkdir -p ${OUTPUT_PATH}/${run}/predictions/
    apptainer exec --bind /data:/data --env PYTHONPATH=".." /nfs/home/springsteinm/env/iart_pytorch_1_11_0_220603.sif python test.py \
    --dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
    --model image_text_heads --heads ontology --transformer_d_model 768 \
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
    --train_merge_one_hot \
    --clip_delete_text_tower

    apptainer exec --bind /data:/data --env PYTHONPATH=. /nfs/home/springsteinm/env/iart_pytorch_1_11_0_220603.sif python tools/calculate_scores.py -f 0 10 100 1000 \
    --mapping_path ${MAPPING_PATH} \
    -p ${OUTPUT_PATH}/${run}/predictions/${PREDICTION_PATH} \
    -o ${OUTPUT_PATH}/${run}/predictions/${RESULT_PATH}
done




# apptainer exec --bind /data:/data --env PYTHONPATH=. ~/env/iart_pytorch_1_11_0_220603.sif python tools/plot_level_ap.py --input_paths \
#  /data/iart/output/reflectai/iconclass/unify_cos_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl \
#  /data/iart/output/reflectai/iconclass/unify_clip_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl \
#  /data/iart/output/reflectai/iconclass/unify_flat_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl \
#  /data/iart/output/reflectai/iconclass/unify_unify_yolo_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl   \
#  /data/iart/output/reflectai/iconclass/unify_onto_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl \
# -l "Flat" "Flat-H" "Flat-W" "CLIP" "CAT" \
# -o /nfs/home/springsteinm/output/reflectai/iconclass/icarus_test_ap_fine_0.pdf -f 0