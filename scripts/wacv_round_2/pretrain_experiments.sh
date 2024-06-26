####################################################################################
# TXT LONG
####################################################################################

sbatch -G 4 -w devbox5 -N 1 --mem 96G singularity.sh exec --env PYTHONPATH=.. ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter \
--max_traces 5 \
--num_sanity_val_steps 0 \
--generate_targets flat yolo ontology clip \
--labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads \
--heads clip \
--transformer_d_model 768 \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/train.jsonl \
--mapping_path /nfs/data/reflectai/iconclass/unify/mapping_trace.jsonl \
--classifier_path /nfs/data/reflectai/iconclass/unify/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/val.jsonl \
--ontology_path /nfs/data/reflectai/iconclass/unify/ontology.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_txt_300/v1 \
--use_wandb \
--wandb_name 4x64_clip_txt_300/v1 \
--lr 1e-4 \
--local_loss \
--gather_with_grad \h
--log_every_n_steps 200 \
--sched_type cosine \
--lr_init 0.0 \
--lr_rampup 500 \
--lr_rampdown 41000 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 64 \
--val_check_interval 1000000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--flip_ration 0.5 \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder transformer_level_wise \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8 \
--context_length 300


sbatch --mem 128G -G t2080ti:4 -N 1 singularity.sh exec --env PYTHONPATH=.. ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter \
--max_traces 5 \
--num_sanity_val_steps 1 \
--generate_targets flat yolo ontology clip \
--labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads \
--heads ontology  \
--transformer_d_model 768 \
--load_clip_from_checkpoint  /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_txt_300/v1/epoch=0-step=39999.ckpt \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/train.jsonl \
--mapping_path /nfs/data/reflectai/iconclass/unify/mapping_trace.jsonl \
--classifier_path /nfs/data/reflectai/iconclass/unify/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/val.jsonl \
--ontology_path /nfs/data/reflectai/iconclass/unify/ontology.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_txt_300_fine_onto/v1 \
--use_wandb \
--wandb_name 4x64_clip_txt_300_fine_onto/v1 \
--lr 1e-4 \
--local_loss \
--gather_with_grad \
--log_every_n_steps 1000 \
--sched_type cosine \
--lr_init 0.0 \
--lr_rampup 500 \
--lr_rampdown 41000  \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 64 \
--val_check_interval 1000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--flip_ration 0.5 \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder transformer_level_wise \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8 \
--context_length 300 \
--clip_delete_text_tower \
--skip_loading_pretrained


####################################################################################
# TXT
####################################################################################

sbatch -G 4 -N 1 --mem 96G singularity.sh exec --env PYTHONPATH=.. ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter \
--max_traces 5 \
--num_sanity_val_steps 0 \
--generate_targets flat yolo ontology clip \
--labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads \
--heads clip \
--transformer_d_model 768 \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/train.jsonl \
--mapping_path /nfs/data/reflectai/iconclass/unify/mapping_trace.jsonl \
--classifier_path /nfs/data/reflectai/iconclass/unify/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/val.jsonl \
--ontology_path /nfs/data/reflectai/iconclass/unify/ontology.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_txt/v1 \
--use_wandb \
--wandb_name 4x64_clip_txt/v1 \
--lr 1e-4 \
--local_loss \
--gather_with_grad \
--log_every_n_steps 200 \
--sched_type cosine \
--lr_init 0.0 \
--lr_rampup 500 \
--lr_rampdown 41000 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 64 \
--val_check_interval 1000000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--flip_ration 0.5 \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder transformer_level_wise \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8

sbatch -w devbox5 --mem 128G -G 4 -N 1 singularity.sh exec --env PYTHONPATH=.. ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter \
--max_traces 5 \
--num_sanity_val_steps 5 \
--generate_targets flat yolo ontology clip \
--labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads \
--heads ontology  \
--transformer_d_model 768 \
--load_clip_from_checkpoint  /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_txt/v1/epoch=0-step=39999.ckpt \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/train.jsonl \
--mapping_path /nfs/data/reflectai/iconclass/unify/mapping_trace.jsonl \
--classifier_path /nfs/data/reflectai/iconclass/unify/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/val.jsonl \
--ontology_path /nfs/data/reflectai/iconclass/unify/ontology.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_txt_fine_onto/v1 \
--use_wandb \
--wandb_name 4x64_clip_txt_fine_onto/v1 \
--lr 1e-4 \
--local_loss \
--gather_with_grad \
--log_every_n_steps 1000 \
--sched_type cosine \
--lr_init 0.0 \
--lr_rampup 500 \
--lr_rampdown 41000  \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 64 \
--val_check_interval 1000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--flip_ration 0.5 \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder transformer_level_wise \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8



####################################################################################
# KW
####################################################################################

sbatch  -G 4 -N 1 --mem 128G singularity.sh exec --env PYTHONPATH=.. ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter \
--max_traces 5 \
--num_sanity_val_steps 0 \
--generate_targets flat yolo ontology clip \
--labels_path /nfs/data/iart/iconclass/iconclass_kw_en.msg \
--model image_text_heads \
--heads clip \
--transformer_d_model 768 \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/train.jsonl \
--mapping_path /nfs/data/reflectai/iconclass/unify/mapping_trace.jsonl \
--classifier_path /nfs/data/reflectai/iconclass/unify/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/val.jsonl \
--ontology_path /nfs/data/reflectai/iconclass/unify/ontology.jsonl \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_kw/v1/epoch=0-step=21999.ckpt \
--output_path /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_kw/v1 \
--use_wandb \
--wandb_name 4x64_clip_kw/v1 \
--lr 1e-4 \
--local_loss \
--gather_with_grad \
--log_every_n_steps 200 \
--sched_type cosine \
--lr_init 0.0 \
--lr_rampup 500 \
--lr_rampdown 41000 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 64 \
--val_check_interval 1000000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--flip_ration 0.5 \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder transformer_level_wise \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8

sbatch --mem 128G -G 4 -N 1 singularity.sh exec --env PYTHONPATH=.. ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter \
--max_traces 5 \
--num_sanity_val_steps 0 \
--generate_targets flat yolo ontology clip \
--labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads \
--heads ontology  \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_kw_fine_onto/v1/epoch=0-step=999-v1.ckpt \
--transformer_d_model 768 \
--load_clip_from_checkpoint  /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_kw/v1/epoch=1-step=39999.ckpt \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/train.jsonl \
--mapping_path /nfs/data/reflectai/iconclass/unify/mapping_trace.jsonl \
--classifier_path /nfs/data/reflectai/iconclass/unify/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/val.jsonl \
--ontology_path /nfs/data/reflectai/iconclass/unify/ontology.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_kw_fine_onto/v1 \
--use_wandb \
--wandb_name 4x64_clip_kw_fine_onto/v1 \
--lr 1e-4 \
--local_loss \
--gather_with_grad \
--log_every_n_steps 1000 \
--sched_type cosine \
--lr_init 0.0 \
--lr_rampup 500 \
--lr_rampdown 41000  \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 64 \
--val_check_interval 100000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--flip_ration 0.5 \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder transformer_level_wise \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8 \
--clip_delete_text_tower 



####################################################################################
# BLIP
####################################################################################

sbatch -G 4 -N 1 --mem 128G singularity.sh exec --env PYTHONPATH=.. ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter \
--max_traces 5 \
--num_sanity_val_steps 0 \
--generate_targets flat yolo ontology clip \
--labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--externel_labels_path /nfs/data/iart/iconclass/data_202002/train_label_blib_instruction.jsonl \
--model image_text_heads \
--heads clip \
--transformer_d_model 768 \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_blip/v1/epoch=0-step=25999.ckpt \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/train.jsonl \
--mapping_path /nfs/data/reflectai/iconclass/unify/mapping_trace.jsonl \
--classifier_path /nfs/data/reflectai/iconclass/unify/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/val.jsonl \
--ontology_path /nfs/data/reflectai/iconclass/unify/ontology.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_blip/v1 \
--use_wandb \
--wandb_name 4x64_clip_blip/v1 \
--lr 1e-4 \
--local_loss \
--gather_with_grad \
--log_every_n_steps 200 \
--sched_type cosine \
--lr_init 0.0 \
--lr_rampup 500 \
--lr_rampdown 41000 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 64 \
--val_check_interval 1000000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--flip_ration 0.5 \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder transformer_level_wise \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8

sbatch --mem 128G -G 4 -N 1 singularity.sh exec --env PYTHONPATH=.. ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter \
--max_traces 5 \
--num_sanity_val_steps 5 \
--generate_targets flat yolo ontology clip \
--labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--externel_labels_path /nfs/data/iart/iconclass/data_202002/train_label_blib_instruction.jsonl \
--model image_text_heads \
--heads ontology  \
--transformer_d_model 768 \
--load_clip_from_checkpoint  /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_blip/v1/epoch=0-step=39999.ckpt \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/train.jsonl \
--mapping_path /nfs/data/reflectai/iconclass/unify/mapping_trace.jsonl \
--classifier_path /nfs/data/reflectai/iconclass/unify/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/val.jsonl \
--ontology_path /nfs/data/reflectai/iconclass/unify/ontology.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_blip_fine_onto/v1 \
--use_wandb \
--wandb_name 4x64_clip_blip_fine_onto/v1 \
--lr 1e-4 \
--local_loss \
--gather_with_grad \
--log_every_n_steps 1000 \
--sched_type cosine \
--lr_init 0.0 \
--lr_rampup 500 \
--lr_rampdown 41000  \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 64 \
--val_check_interval 1000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--flip_ration 0.5 \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder transformer_level_wise \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8

####################################################################################
# GPT
####################################################################################

sbatch -G t2080ti:4 -N 1 --mem 128G singularity.sh exec --env PYTHONPATH=.. ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter \
--max_traces 5 \
--num_sanity_val_steps 0 \
--generate_targets flat yolo ontology clip \
--labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--externel_labels_path /nfs/data/iart/iconclass/data_202002/train_label_gpt2_100000.jsonl \
--model image_text_heads \
--heads clip \
--transformer_d_model 768 \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/train.jsonl \
--mapping_path /nfs/data/reflectai/iconclass/unify/mapping_trace.jsonl \
--classifier_path /nfs/data/reflectai/iconclass/unify/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/val.jsonl \
--ontology_path /nfs/data/reflectai/iconclass/unify/ontology.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_gpt/v1 \
--use_wandb \
--wandb_name 4x64_clip_gpt/v1 \
--lr 1e-4 \
--local_loss \
--gather_with_grad \
--log_every_n_steps 200 \
--sched_type cosine \
--lr_init 0.0 \
--lr_rampup 500 \
--lr_rampdown 41000 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 64 \
--val_check_interval 1000000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--flip_ration 0.5 \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder transformer_level_wise \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8

sbatch --mem 128G -G 4 -N 1 singularity.sh exec --env PYTHONPATH=.. ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter \
--max_traces 5 \
--num_sanity_val_steps 5 \
--generate_targets flat yolo ontology clip \
--labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--externel_labels_path /nfs/data/iart/iconclass/data_202002/train_label_gpt2_100000.jsonl \
--model image_text_heads \
--heads ontology  \
--transformer_d_model 768 \
--load_clip_from_checkpoint  /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_gpt/v1/epoch=0-step=39999.ckpt \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/train.jsonl \
--mapping_path /nfs/data/reflectai/iconclass/unify/mapping_trace.jsonl \
--classifier_path /nfs/data/reflectai/iconclass/unify/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/val.jsonl \
--ontology_path /nfs/data/reflectai/iconclass/unify/ontology.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_clip_gpt_fine_onto/v1 \
--use_wandb \
--wandb_name 4x64_clip_gpt_fine_onto/v1 \
--lr 1e-4 \
--local_loss \
--gather_with_grad \
--log_every_n_steps 1000 \
--sched_type cosine \
--lr_init 0.0 \
--lr_rampup 500 \
--lr_rampdown 41000  \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 64 \
--val_check_interval 1000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--flip_ration 0.5 \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder transformer_level_wise \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8 \
--clip_delete_text_tower 

####################################################################################
# LAION-400M
####################################################################################


sbatch --mem 128G -G 4 -N 1 singularity.sh exec --env PYTHONPATH=.. ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter \
--max_traces 5 \
--num_sanity_val_steps 5 \
--generate_targets flat yolo ontology clip \
--model image_text_heads \
--heads ontology  \
--transformer_d_model 768 \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/train.jsonl \
--mapping_path /nfs/data/reflectai/iconclass/unify/mapping_trace.jsonl \
--classifier_path /nfs/data/reflectai/iconclass/unify/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/reflectai/iconclass/unify/iconclass/val.jsonl \
--ontology_path /nfs/data/reflectai/iconclass/unify/ontology.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass_wacv_round_2/4x64_fine_onto/v1 \
--use_wandb \
--wandb_name 4x64_fine_onto/v1 \
--lr 1e-4 \
--local_loss \
--gather_with_grad \
--log_every_n_steps 1000 \
--sched_type cosine \
--lr_init 0.0 \
--lr_rampup 500 \
--lr_rampdown 41000  \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 64 \
--val_check_interval 1000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--flip_ration 0.5 \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder transformer_level_wise \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8 \
--clip_delete_text_tower 