
apptainer exec --bind /data:/data --env PYTHONPATH=. /nfs/home/springsteinm/env/iart_pytorch_1_11_0_220603.sif python tools/generate_result_tex_table.py \
 --input_paths \
 /nfs/home/springsteinm/output/iart/iconclass/onto_fine_32_batch_4x64_packed/230610/predictions/ap_prob_chain.jsonl \
 /nfs/home/springsteinm/output/iart/iconclass/onto_fine_32_batch_4x64_packed_blib_instruction/230610/predictions/ap_prob_chain.jsonl \
 /nfs/home/springsteinm/output/iart/iconclass/onto_fine_32_batch_4x64_packed_gpt/230610/predictions/ap_prob_chain.jsonl \
 /nfs/home/springsteinm/output/iart/iconclass/onto_fine_32_packed/230610/predictions/ap_prob_chain.jsonl \
 -l "kw" "blib" "gpt" "plain"



apptainer exec --bind /data:/data --env PYTHONPATH=. /nfs/home/springsteinm/env/iart_pytorch_1_11_0_220603.sif python tools/generate_result_tex_table.py \
 --input_paths \
 /data/iart/output/reflectai/iconclass/unify_cos_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_brill_test_ap_prob_chain.jsonl \
 /data/iart/output/reflectai/iconclass/unify_clip_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_brill_test_ap_prob_chain.jsonl \
 /data/iart/output/reflectai/iconclass/unify_flat_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_brill_test_ap_prob_chain.jsonl \
 /data/iart/output/reflectai/iconclass/unify_unify_yolo_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_brill_test_ap_prob_chain.jsonl   \
 /data/iart/output/reflectai/iconclass/unify_onto_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_brill_test_ap_prob_chain.jsonl \
 -l "Cos" "CLIP (Fine)" "Flat" "YOLO" "Hierarchical"

apptainer exec --bind /data:/data --env PYTHONPATH=. /nfs/home/springsteinm/env/iart_pytorch_1_11_0_220603.sif python tools/generate_result_tex_table.py \
 --input_paths \
 /data/iart/output/reflectai/iconclass/unify_cos_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_brill_all_ap_prob_chain.jsonl \
 /data/iart/output/reflectai/iconclass/unify_clip_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_brill_all_ap_prob_chain.jsonl \
 /data/iart/output/reflectai/iconclass/unify_flat_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_brill_all_ap_prob_chain.jsonl \
 /data/iart/output/reflectai/iconclass/unify_unify_yolo_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_brill_all_ap_prob_chain.jsonl   \
 /data/iart/output/reflectai/iconclass/unify_onto_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_brill_all_ap_prob_chain.jsonl \
 -l "Cos" "CLIP (Fine)" "Flat" "YOLO" "Hierarchical"


apptainer exec --bind /data:/data --env PYTHONPATH=. /nfs/home/springsteinm/env/iart_pytorch_1_11_0_220603.sif python tools/generate_result_tex_table.py \
 --input_paths \
 /data/iart/output/reflectai/iconclass/unify_cos_fine_batch_4x64_packed_gpt/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl \
 /data/iart/output/reflectai/iconclass/unify_clip_batch_4x64_packed_gpt/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl \
 /data/iart/output/reflectai/iconclass/unify_flat_fine_batch_4x64_packed_gpt/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl \
 /data/iart/output/reflectai/iconclass/unify_unify_yolo_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl   \
 /data/iart/output/reflectai/iconclass/unify_onto_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl \
 -l "Cos" "CLIP (Fine)" "Flat" "YOLO" "Hierarchical"

singularity.sh exec --env PYTHONPATH=. ~/env/iart_pytorch_1_11_0_220603.sif python tools/plot_level_ap.py --input_paths \
 /data/iart/output/reflectai/iconclass/unify_cos_fine_batch_4x64_packed_gpt/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl \
 /data/iart/output/reflectai/iconclass/unify_clip_batch_4x64_packed_gpt/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl \
 /data/iart/output/reflectai/iconclass/unify_flat_fine_batch_4x64_packed_gpt/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl \
 /data/iart/output/reflectai/iconclass/unify_unify_yolo_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl   \
 /data/iart/output/reflectai/iconclass/unify_onto_fine_4x64_batch_4x64_packed_blib_instruction/230617/predictions/unify_iconclass_test_ap_prob_chain.jsonl \
-l "Hierarchical" "YOLO" "CLIP" "Flat" "Cos" \
-o /nfs/home/springsteinm/output/iart/iconclass/test_ap_fine_0.pdf -f 0