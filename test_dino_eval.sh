
export PYTHONPATH="/raid/homes/giovanni.puccetti/Repos/dinoclip/dinov2"
export CUDA_VISIBLE_DEVICES="5"

python -m dinov2.eval.knn \
    --config-file /raid/homes/giovanni.puccetti/Repos/dinoclip/dino_vit16_test/config.yaml \
    --pretrained-weights /raid/homes/giovanni.puccetti/Repos/dinoclip/dino_vit16_test/eval/training_124999/teacher_checkpoint.pth \
    --output-dir /raid/homes/giovanni.puccetti/Repos/dinoclip/dino_vit16_test/eval/training_124999/knn \
    --train-dataset ImageNet:split=TRAIN:root=/raid/datasets/imagenet-1k/data:extra=/raid/datasets/imagenet-1k/data \
    --val-dataset ImageNet:split=VAL:root=/raid/datasets/imagenet-1k/data:extra=/raid/datasets/imagenet-1k/data \
    --batch-size 4096
