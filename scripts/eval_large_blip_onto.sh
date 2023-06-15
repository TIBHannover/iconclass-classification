#!/usr/bin/bash


runs=( \
    # /nfs/home/springsteinm/output/iart/iconclass/onto_fine_4x64_batch_4x64_packed_blib_instruction/230611 \
    /nfs/home/springsteinm/output/iart/iconclass/onto_fine_4x64packed/230613 \
)


for model_path in "${runs[@]}"
do	
    echo ${model_path}
    
    mkdir -p ${model_path}/predictions/
    
    apptainer exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python test.py \
    --dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
    --model image_text_heads --heads ontology --transformer_d_model 768 \
    --mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
    --classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
    --test_path /nfs/data/iart/iconclass/data_202002/packed/test/msg/ \
    --test_annotation_path /nfs/data/iart/iconclass/data_202002/test.jsonl --use_probability_chain \
    --resume_from_checkpoint ${model_path}/epoch=*-step=39999.ckpt \
    --prediction_path ${model_path}/predictions/test_prediction_prob_chain_40000.h5 \
    --lr 1e-4 --local_loss --gather_with_grad --log_every_n_steps 1 \
    --train_size 224 \
    --use_center_crop \
    --val_size 224 \
    --batch_size 32 \
    --precision 16 \
    --gpus -1  \
    --encoder open_clip \
    --decoder transformer_level_wise \
    --ontology_path /nfs/data/iart/iconclass/200817/ontology.jsonl --train_merge_one_hot

    apptainer exec --env PYTHONPATH=. ~/env/iart_pytorch_1_11_0_220603.sif python tools/calculate_scores.py -f 0 10 100 1000 \
    --mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
    -p ${model_path}/predictions/test_prediction_prob_chain_40000.h5 \
    -o ${model_path}/predictions/ap_prob_chain.jsonl
done