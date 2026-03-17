CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=25675 train.py \
    --cfg /home/codes/TP-IQA/configs/Pure/vit_small_pre_coder_livec.yaml \
    --tensorboard --tag livec_baseline\
    --dist --scene \
    --visual --quality \
    --alpha 1.0 --beta 0.5 \
    --epoch 40 --seed 1024 --repeat --rnum 10