
export PYTHONPATH="/raid/homes/giovanni.puccetti/Repos/dinoclip/dinov2"
export CUDA_VISIBLE_DEVICES=3,4

torchrun --nnodes 1 --nproc-per-node 2 --master-addr 127.0.0.1 --master-port 19500 -m dinov2.train.train \
    --config-file dinov2/configs/train/vitl16_short.yaml \
    --output-dir /raid/homes/giovanni.puccetti/Repos/dinoclip/dino_vit16_test \
    train.dataset_path=ImageNet:split=TRAIN:root=/raid/datasets/imagenet-1k/data:extra=/raid/datasets/imagenet-1k/data
