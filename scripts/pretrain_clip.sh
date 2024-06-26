# This script is not done

sbatch -G 4 -w devbox5 -N 1 singularity.sh exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg  \
--model image_text_heads --heads clip ontology --transformer_d_model 768 \
--train_path /nfs/data/iart/iconclass/data_202002/raw/train/msg/ \
--train_annotation_path /nfs/data/iart/iconclass/data_202002/train.jsonl \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/raw/val/msg/ \
--val_annotation_path /nfs/data/iart/iconclass/data_202002/val.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass/clip+onto_batch_4x64/220617 --use_wandb --wandb_name clip+onto_batch_4x64/220617 \
--lr 1e-4 --local_loss --gather_with_grad --log_every_n_steps 1 --sched_type cosine --lr_init 0.0 --lr_rampup 500 --lr_rampdown 41000 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 64 \
--val_check_interval 1000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder transformer_level_wise \
--ontology_path /nfs/data/iart/iconclass/200817/ontology.jsonl \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8


sbatch -G 4 -w devbox5 -N 1 singularity.sh exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg  \
--model image_text_heads --heads clip ontology --transformer_d_model 768 \
--train_path /nfs/data/iart/iconclass/data_202002/raw/train/msg/ \
--train_annotation_path /nfs/data/iart/iconclass/data_202002/train.jsonl \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/raw/val/msg/ \
--val_annotation_path /nfs/data/iart/iconclass/data_202002/val.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass/clip+onto_batch_4x64/220617 --use_wandb --wandb_name clip+onto_batch_4x64/220617 \
--lr 1e-4 --local_loss --gather_with_grad --log_every_n_steps 1 --sched_type cosine --lr_init 0.0 --lr_rampup 500 --lr_rampdown 41000 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 64 \
--val_check_interval 1000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder transformer_level_wise \
--ontology_path /nfs/data/iart/iconclass/200817/ontology.jsonl \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8


sbatch -G 4 -w devbox5 -N 1 singularity.sh exec ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads --heads clip --transformer_d_model 768 \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/iart/iconclass/data_202002/train.jsonl \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/iart/iconclass/data_202002/val.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass/clip_batch_4x64_packed/220705 --use_wandb --wandb_name clip_batch_4x64_packed/220705 \
--lr 1e-4 --local_loss --gather_with_grad --log_every_n_steps 1 --sched_type cosine --lr_init 0.0 --lr_rampup 500 --lr_rampdown 41000 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 64 \
--val_check_interval 100000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder transformer_level_wise \
--ontology_path /nfs/data/iart/iconclass/200817/ontology.jsonl \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8