
export PYTHONPATH="/raid/homes/giovanni.puccetti/Repos/dinoclip/dinov2"
export CUDA_VISIBLE_DEVICES="4"

python -m dinov2.eval.knn \
    --config-file /raid/homes/giovanni.puccetti/Repos/dinoclip/dinov2/dinov2/configs/silc_vit_b_32.yaml \
    --pretrained-weights /raid/homes/giovanni.puccetti/Repos/dinoclip/notebooks/80m_dinov2_teacher_dict_epoch_2.pt \
    --output-dir /raid/homes/giovanni.puccetti/Repos/dinoclip/notebooks/80m_eval/epoch_2/ \
    --train-dataset ImageNet:split=TRAIN:root=/raid/datasets/imagenet-1k/data:extra=/raid/datasets/imagenet-1k/data \
    --val-dataset ImageNet:split=VAL:root=/raid/datasets/imagenet-1k/data:extra=/raid/datasets/imagenet-1k/data \
    --batch-size 4096
