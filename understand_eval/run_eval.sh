python3 -m accelerate.commands.launch \
    --num_processes=8 \
    --main_process_port=39535 \
    -m lmms_eval \
    --model vargpt_llava \
    --model_args pretrained="path/to/VARGPT_LLaVA-7B-v1" \
    --tasks mmmu \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava-hf_mmmu \
    --output_path ./logs/


