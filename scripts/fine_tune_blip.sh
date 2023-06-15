sbatch --mem 32G -G 1 -N 1 singularity.sh exec  ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 5 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg --externel_labels_path /nfs/data/iart/iconclass/data_202002/train_label_blib_instruction.jsonl \
--model image_text_heads --heads cos_cel \
--load_clip_from_checkpoint  /nfs/home/springsteinm/output/iart/iconclass/clip_batch_4x64_packed_blib_instruction/230607/epoch=2-step=39999.ckpt \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/iart/iconclass/data_202002/train.jsonl \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/iart/iconclass/data_202002/val.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass/cos_fine_32_batch_4x64_packed_blib_instruction/230610 \
--use_wandb --wandb_name cos_fine_32_batch_4x64_packed_blib_instruction/230610 \
--lr 1e-5 --local_loss --gather_with_grad --log_every_n_steps 1 --sched_type cosine --lr_init 0.0 --lr_rampup 500 --lr_rampdown 41000 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 32 \
--val_check_interval 1000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder flat \
--ontology_path /nfs/data/iart/iconclass/200817/ontology.jsonl \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8 \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/cos_fine_32_batch_4x64_packed_blib_instruction/230610/

sbatch --mem 32G -G 1 -N 1 singularity.sh exec  ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg --externel_labels_path /nfs/data/iart/iconclass/data_202002/train_label_blib_instruction.jsonl \
--model image_text_heads --heads clip --transformer_d_model 768 \
--load_clip_from_checkpoint  /nfs/home/springsteinm/output/iart/iconclass/clip_batch_4x64_packed_blib_instruction/230607/epoch=2-step=39999.ckpt \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/iart/iconclass/data_202002/train.jsonl \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/iart/iconclass/data_202002/val.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass/clip_fine_32_batch_4x64_packed_blib_instruction/230610 \
--use_wandb --wandb_name clip_fine_32_batch_4x64_packed_blib_instruction/230610 \
--lr 1e-5 --local_loss --gather_with_grad --log_every_n_steps 1 --sched_type cosine --lr_init 0.0 --lr_rampup 500 --lr_rampdown 41000 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 32 \
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
--limit_val_batches 8 \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/clip_fine_32_batch_4x64_packed_blib_instruction/230610/



sbatch --mem 32G -G 1 -N 1 singularity.sh exec  ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 5 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg --externel_labels_path /nfs/data/iart/iconclass/data_202002/train_label_blib_instruction.jsonl \
--model image_text_heads --heads flat \
--load_clip_from_checkpoint  /nfs/home/springsteinm/output/iart/iconclass/clip_batch_4x64_packed_blib_instruction/230607/epoch=2-step=39999.ckpt \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/iart/iconclass/data_202002/train.jsonl \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/iart/iconclass/data_202002/val.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass/flat_fine_32_batch_4x64_packed_blib_instruction/230610 \
--use_wandb --wandb_name flat_fine_32_batch_4x64_packed_blib_instruction/230610 \
--lr 1e-5 --local_loss --gather_with_grad --log_every_n_steps 1 --sched_type cosine --lr_init 0.0 --lr_rampup 500 --lr_rampdown 41000 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 32 \
--val_check_interval 1000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder flat \
--ontology_path /nfs/data/iart/iconclass/200817/ontology.jsonl \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8 \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/flat_fine_32_batch_4x64_packed_blib_instruction/230610/


sbatch --mem 32G -G 1 -N 1 singularity.sh exec  ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 5 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg --externel_labels_path /nfs/data/iart/iconclass/data_202002/train_label_blib_instruction.jsonl \
--model image_text_heads --heads yolo \
--load_clip_from_checkpoint  /nfs/home/springsteinm/output/iart/iconclass/clip_batch_4x64_packed_blib_instruction/230607/epoch=2-step=39999.ckpt \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/iart/iconclass/data_202002/train.jsonl \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/iart/iconclass/data_202002/val.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass/yolo_fine_32_batch_4x64_packed_blib_instruction/230610 \
--use_wandb --wandb_name yolo_fine_32_batch_4x64_packed_blib_instruction/230610 \
--lr 1e-5 --local_loss --gather_with_grad --log_every_n_steps 1 --sched_type cosine --lr_init 0.0 --lr_rampup 500 --lr_rampdown 41000 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 32 \
--val_check_interval 1000 -v \
--checkpoint_save_interval 1000  \
--strategy ddp \
--precision 16 \
--gpus -1  \
--weight_decay 1e-01 \
--gradient_clip_val 0.5 \
--encoder open_clip \
--decoder flat \
--ontology_path /nfs/data/iart/iconclass/200817/ontology.jsonl \
--train_random_trace \
--train_merge_one_hot \
--max_steps 40000 \
--limit_val_batches 8 \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/yolo_fine_32_batch_4x64_packed_blib_instruction/230610/


sbatch --mem 32G -w devbox5 -G 1 -N 1 singularity.sh exec  ~/env/iart_pytorch_1_11_0_220603.sif python train.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 5 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg --externel_labels_path /nfs/data/iart/iconclass/data_202002/train_label_blib_instruction.jsonl \
--model image_text_heads --heads ontology  --transformer_d_model 768 \
--load_clip_from_checkpoint  /nfs/home/springsteinm/output/iart/iconclass/clip_batch_4x64_packed_blib_instruction/230607/epoch=2-step=39999.ckpt \
--train_path /nfs/data/iart/iconclass/data_202002/packed/train/msg/ \
--train_annotation_path /nfs/data/iart/iconclass/data_202002/train.jsonl \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--val_path /nfs/data/iart/iconclass/data_202002/packed/val/msg/ \
--val_annotation_path /nfs/data/iart/iconclass/data_202002/val.jsonl \
--output_path /nfs/home/springsteinm/output/iart/iconclass/onto_fine_32_batch_4x64_packed_blib_instruction/230610 \
--use_wandb --wandb_name onto_fine_32_batch_4x64_packed_blib_instruction/230610 \
--lr 1e-5 --local_loss --gather_with_grad --log_every_n_steps 1 --sched_type cosine --lr_init 0.0 --lr_rampup 500 --lr_rampdown 41000  \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 32 \
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
--limit_val_batches 8 \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/onto_fine_32_batch_4x64_packed_blib_instruction/230610/