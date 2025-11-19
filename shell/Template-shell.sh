#train
python3 main.py --config configs/Template_LBBDM_f4.yaml --train --sample_at_start --save_top --gpu_ids 0 \
--resume_model path/to/model_ckpt --resume_optim path/to/optim_ckpt

#test
python3 main.py --config configs/Template_LBBDM_f4.yaml --sample_to_eval --gpu_ids 0 \
--resume_model path/to/model_ckpt --resume_optim path/to/optim_ckpt

#preprocess and evaluation
# python SSIM_PSNR_LPIPS.py