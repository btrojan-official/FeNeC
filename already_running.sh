###### NO LOGITS ######




# ResNet Cifar 100 no logits

python search.py \
    --dataset resnet \
    --model resnet \
    --use_logits false \
    --num_of_trials 600 \
    --num_of_tasks 6 \
    --output_file searches_results/cifar100_resnet_no_logits_TPESampler.csv \
    --sampler TPESampler \
    --result_dir searches_results/cifar100_resnet_no_logits_TPESampler


  python search.py \
    --dataset tinyimagenet \
    --model resnet \
    --use_logits false \
    --num_of_trials 600 \
    --num_of_tasks 6 \
    --output_file searches_results/tinyimagenet_resnet_no_logits_TPESampler.csv \
    --sampler TPESampler \
    --result_dir searches_results/tinyimagenet_resnet_no_logits_TPESampler