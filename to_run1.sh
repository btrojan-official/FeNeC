python search.py \
    --dataset resnet \
    --model resnet \
    --use_logits true \
    --num_of_trials 600 \
    --num_of_tasks 6 \
    --output_file searches_results/cifar100_resnet_logits_GPSampler.csv \
    --sampler GPSampler \
    --result_dir searches_results/cifar100_resnet_logits_GPSampler

python search.py \
    --dataset resnet \
    --model resnet \
    --use_logits true \
    --num_of_trials 600 \
    --num_of_tasks 6 \
    --output_file searches_results/cifar100_resnet_logits_QMCSampler.csv \
    --sampler QMCSampler \
    --result_dir searches_results/cifar100_resnet_logits_QMCSampler

python search.py \
    --dataset tinyimagenet \
    --model resnet \
    --use_logits true \
    --num_of_trials 600 \
    --num_of_tasks 6 \
    --output_file searches_results/tinyimagenet_resnet_logits_TPESampler.csv \
    --sampler TPESampler \
    --result_dir searches_results/tinyimagenet_resnet_logits_TPESampler

python search.py \
    --dataset tinyimagenet \
    --model resnet \
    --use_logits true \
    --num_of_trials 600 \
    --num_of_tasks 6 \
    --output_file searches_results/tinyimagenet_resnet_logits_GPSampler.csv \
    --sampler GPSampler \
    --result_dir searches_results/tinyimagenet_resnet_logits_GPSampler

python search.py \
    --dataset tinyimagenet \
    --model resnet \
    --use_logits true \
    --num_of_trials 600 \
    --num_of_tasks 6 \
    --output_file searches_results/tinyimagenet_resnet_logits_QMCSampler.csv \
    --sampler QMCSampler \
    --result_dir searches_results/tinyimagenet_resnet_logits_QMCSampler