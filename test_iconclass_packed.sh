/nfs/home/springsteinm/output/iart/iconclass/clip+onto_batch_4x64_packed_gpt/220628/epoch=1-step=39999.ckpt
/nfs/home/springsteinm/output/iart/iconclass/onto_batch_4x64_packed/220621/epoch=0-step=39999.ckpt
/nfs/home/springsteinm/output/iart/iconclass/clip_batch_4x64_packed_gpt/220628/epoch=0-step=39999.ckpt
/nfs/home/springsteinm/output/iart/iconclass/clip+yolo_batch_4x64/220617/epoch=0-step=39999.ckpt
/nfs/home/springsteinm/output/iart/iconclass/yolo_batch_4x64/220617/epoch=0-step=39999.ckpt
/nfs/home/springsteinm/output/iart/iconclass/clip_batch_4x64/220617/epoch=0-step=39999.ckpt
/nfs/home/springsteinm/output/iart/iconclass/clip+onto_batch_4x64/220617/epoch=0-step=39999.ckpt
/nfs/home/springsteinm/output/iart/iconclass/clip+onto_batch_4x64_packed/220621/epoch=0-step=39999.ckpt
/nfs/home/springsteinm/output/iart/iconclass/onto_batch_4x64/220617/epoch=0-step=39999.ckpt


sbatch -w devbox5  -G 1 -N 1 singularity.sh exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python test.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads --heads ontology clip --transformer_d_model 768 \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--test_path /nfs/data/iart/iconclass/data_202002/packed/test/msg/ \
--test_annotation_path /nfs/data/iart/iconclass/data_202002/test.jsonl \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/clip+onto_batch_4x64_packed_gpt/220628/epoch=1-step=39999.ckpt --prediction_path /nfs/home/springsteinm/output/iart/iconclass/clip+onto_batch_4x64_packed_gpt/220628/test_prediction_40000.h5 \
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

sbatch -w devbox5 -G 1 -N 1 singularity.sh exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python test.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads --heads ontology --transformer_d_model 768 \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--test_path /nfs/data/iart/iconclass/data_202002/packed/test/msg/ \
--test_annotation_path /nfs/data/iart/iconclass/data_202002/test.jsonl \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/onto_batch_4x64_packed/220621/epoch=0-step=39999.ckpt --prediction_path /nfs/home/springsteinm/output/iart/iconclass/onto_batch_4x64_packed/220621/test_prediction_40000.h5 \
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


sbatch -w devbox5 -G 1 -N 1 singularity.sh exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python test.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads --heads clip --transformer_d_model 768 \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--test_path /nfs/data/iart/iconclass/data_202002/packed/test/msg/ \
--test_annotation_path /nfs/data/iart/iconclass/data_202002/test.jsonl \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/clip_batch_4x64_packed_gpt/220628/epoch=0-step=39999.ckpt --prediction_path /nfs/home/springsteinm/output/iart/iconclass/clip_batch_4x64_packed_gpt/220628/test_prediction_40000.h5 \
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

sbatch -w devbox5 -G 1 -N 1 singularity.sh exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python test.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads --heads yolo clip \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--test_path /nfs/data/iart/iconclass/data_202002/packed/test/msg/ \
--test_annotation_path /nfs/data/iart/iconclass/data_202002/test.jsonl \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/clip+yolo_batch_4x64/220617/epoch=0-step=39999.ckpt --prediction_path /nfs/home/springsteinm/output/iart/iconclass/clip+yolo_batch_4x64/220617/test_prediction_40000.h5 \
--lr 1e-4 --local_loss --gather_with_grad --log_every_n_steps 1 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 32 \
--precision 16 \
--gpus -1  \
--encoder open_clip \
--decoder flat \
--ontology_path /nfs/data/iart/iconclass/200817/ontology.jsonl --train_merge_one_hot

sbatch -w devbox5 -G 1 -N 1 singularity.sh exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python test.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads --heads yolo \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--test_path /nfs/data/iart/iconclass/data_202002/packed/test/msg/ \
--test_annotation_path /nfs/data/iart/iconclass/data_202002/test.jsonl \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/yolo_batch_4x64/220617/epoch=0-step=39999.ckpt --prediction_path /nfs/home/springsteinm/output/iart/iconclass/yolo_batch_4x64/220617/test_prediction_40000.h5 \
--lr 1e-4 --local_loss --gather_with_grad --log_every_n_steps 1 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 32 \
--precision 16 \
--gpus -1  \
--encoder open_clip \
--decoder flat \
--ontology_path /nfs/data/iart/iconclass/200817/ontology.jsonl --train_merge_one_hot

sbatch -w devbox5 -G 1 -N 1 singularity.sh exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python test.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads --heads clip --transformer_d_model 768 \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--test_path /nfs/data/iart/iconclass/data_202002/packed/test/msg/ \
--test_annotation_path /nfs/data/iart/iconclass/data_202002/test.jsonl \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/clip_batch_4x64/220617/epoch=0-step=39999.ckpt --prediction_path /nfs/home/springsteinm/output/iart/iconclass/clip_batch_4x64/220617/test_prediction_40000.h5 \
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

sbatch -w devbox5 -G 1 -N 1 singularity.sh exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python test.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads --heads clip ontology --transformer_d_model 768 \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--test_path /nfs/data/iart/iconclass/data_202002/packed/test/msg/ \
--test_annotation_path /nfs/data/iart/iconclass/data_202002/test.jsonl \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/clip+onto_batch_4x64/220617/epoch=0-step=39999.ckpt --prediction_path /nfs/home/springsteinm/output/iart/iconclass/clip+onto_batch_4x64/220617/test_prediction_40000.h5 \
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

sbatch -w devbox5 -G 1 -N 1 singularity.sh exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python test.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads --heads ontology clip --transformer_d_model 768 \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--test_path /nfs/data/iart/iconclass/data_202002/packed/test/msg/ \
--test_annotation_path /nfs/data/iart/iconclass/data_202002/test.jsonl \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/clip+onto_batch_4x64_packed/220621/epoch=0-step=39999.ckpt --prediction_path /nfs/home/springsteinm/output/iart/iconclass/clip+onto_batch_4x64_packed/220621/test_prediction_40000.h5 \
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

sbatch -w devbox5 -G 1 -N 1 singularity.sh exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python test.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads --heads ontology  --transformer_d_model 768 \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--test_path /nfs/data/iart/iconclass/data_202002/packed/test/msg/ \
--test_annotation_path /nfs/data/iart/iconclass/data_202002/test.jsonl \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/onto_batch_4x64/220617/epoch=0-step=39999.ckpt --prediction_path /nfs/home/springsteinm/output/iart/iconclass/onto_batch_4x64/220617/test_prediction_40000.h5 \
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

### c-hmcnn

sbatch -w devbox5 -G 1 -N 1 singularity.sh exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python test.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads --heads c_hmcnn  \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--test_path /nfs/data/iart/iconclass/data_202002/packed/test/msg/ \
--test_annotation_path /nfs/data/iart/iconclass/data_202002/test.jsonl \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/c_hmcnn_v1_fine_batch_4x2_packed_gpt/220705/epoch=0-step=39999.ckpt --prediction_path /nfs/home/springsteinm/output/iart/iconclass/c_hmcnn_v1_fine_batch_4x2_packed_gpt/220705/test_prediction_40000.h5 \
--lr 1e-4 --local_loss --gather_with_grad --log_every_n_steps 1 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 2 \
--precision 16 \
--gpus -1  \
--encoder open_clip \
--decoder flat \
--ontology_path /nfs/data/iart/iconclass/200817/ontology.jsonl --train_merge_one_hot


sbatch -w devbox5 -G 1 -N 1 singularity.sh exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python test.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads --heads c_hmcnn  \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--test_path /nfs/data/iart/iconclass/data_202002/packed/test/msg/ \
--test_annotation_path /nfs/data/iart/iconclass/data_202002/test.jsonl \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/c_hmcnn_v2_fine_batch_4x2_packed_gpt/220705/epoch=0-step=39999.ckpt --prediction_path /nfs/home/springsteinm/output/iart/iconclass/c_hmcnn_v2_fine_batch_4x2_packed_gpt/220705/test_prediction_40000.h5 \
--lr 1e-4 --local_loss --gather_with_grad --log_every_n_steps 1 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 2 \
--precision 16 \
--gpus -1  \
--encoder open_clip \
--decoder flat \
--ontology_path /nfs/data/iart/iconclass/200817/ontology.jsonl --train_merge_one_hot


sbatch -w devbox5 -G 1 -N 1 singularity.sh exec --env TORCH_DISTRIBUTED_DEBUG=DETAIL,NCCL_DEBUG="INFO" ~/env/iart_pytorch_1_11_0_220603.sif python test.py \
--dataset iconclass_iter --max_traces 5 --num_sanity_val_steps 0 --generate_targets flat yolo ontology clip --labels_path /nfs/data/iart/iconclass/iconclass_txt_en.msg \
--model image_text_heads --heads c_hmcnn  \
--mapping_path /nfs/data/iart/iconclass/200817/mapping.jsonl \
--classifier_path /nfs/data/iart/iconclass/200817/classifiers.jsonl \
--test_path /nfs/data/iart/iconclass/data_202002/packed/test/msg/ \
--test_annotation_path /nfs/data/iart/iconclass/data_202002/test.jsonl \
--resume_from_checkpoint /nfs/home/springsteinm/output/iart/iconclass/c_hmcnn_v3_fine_batch_4x2_packed_gpt/220705/epoch=0-step=39999.ckpt --prediction_path /nfs/home/springsteinm/output/iart/iconclass/c_hmcnn_v3_fine_batch_4x2_packed_gpt/220705/test_prediction_40000.h5 \
--lr 1e-4 --local_loss --gather_with_grad --log_every_n_steps 1 \
--train_size 224 \
--use_center_crop \
--val_size 224 \
--batch_size 2 \
--precision 16 \
--gpus -1  \
--encoder open_clip \
--decoder flat \
--ontology_path /nfs/data/iart/iconclass/200817/ontology.jsonl --train_merge_one_hot