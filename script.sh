
## Training with best parameters in paper on CUB200
python run.py \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--optimizers {metric_loss_optimizer: {RMSprop: {weight_decay: 0.0001, momentum: 0.9, lr: 0.000005}}} \
--sampler {MPerClassSampler: {m: 1}} \
--splits_to_eval [val] \
--loss_funcs {metric_loss~OVERRIDE~: {HierarchicalProxyLoss: {embedding_size: 128, scale: 4.5166, proxy_per_class: 5, w1: 0.4, w2: 0.6}}} \
--experiment_name CUB

## Evaluating the model trained on CUB200
python run.py \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--splits_to_eval [test] \
--experiment_name CUB \
--evaluate_ensemble



## Training with best parameters on CARS196
python run.py \
--dataset~OVERRIDE~ {CARS196: {download: True}} \
--optimizers {metric_loss_optimizer: {RMSprop: {weight_decay: 0.0001, momentum: 0.9, lr: 0.000005}}} \
--sampler {MPerClassSampler: {m: 1}} \
--config_general [default, with_cars196] \
--splits_to_eval [val] \
--loss_funcs {metric_loss~OVERRIDE~: {HierarchicalProxyLoss: {embedding_size: 128, scale: 4.5166, proxy_per_class: 4, w1: 0.4, w2: 0.8}}} \
--experiment_name CARS

## Evaluating the model trained on CARS196
python run.py \
--dataset~OVERRIDE~ {CARS196: {download: True}} \
--splits_to_eval [test] \
--experiment_name CARS \
--evaluate_ensemble



## Training with best parameters on SOP
python run.py \
--dataset~OVERRIDE~ {StanfordOnlineProducts: {download: True}} \
--optimizers {metric_loss_optimizer: {RMSprop: {weight_decay: 0.0001, momentum: 0.9, lr: 0.0005}}} \
--sampler {MPerClassSampler: {m: 1}} \
--config_general [default, with_sop] \
--splits_to_eval [val] \
--loss_funcs {metric_loss~OVERRIDE~: {HierarchicalProxyLoss: {embedding_size: 128, scale: 4.5166, proxy_per_class: 3, w1: 0.2, w2: 0.4}}} \
--experiment_name SOP

## Evaluating the model trained on SOP
python run.py \
--dataset~OVERRIDE~ {StanfordOnlineProducts: {download: True}} \
--splits_to_eval [test] \
--experiment_name SOP \
--evaluate_ensemble
