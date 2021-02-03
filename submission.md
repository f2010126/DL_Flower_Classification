##### Which track(s) are you participating in? {Fast Networks Track, Large Networks Track, Both}
Fast Networks Track

##### What are the number of learnable parameters in your model?
Fast Networks Track -
Large Networks Track -

##### Briefly describe your approach
Fast Networks Track -
Large Networks Track -

##### Command to train your model from scratch
Fast Networks Track -
Large Networks Track -


##### Command to evaluate your model
Fast Networks Track -
Large Networks Track -

python -m src.main --model FeatExtEfficientNetB4 --epochs 1 --batch_size 20 --data-augmentation efficientnet_augmentations --data-augmentation-validation efficientnet_augmentations_valid --optimizer sgd --opti_momentum 0.7 --opti_alpha 0.7 --scheduler cosine --t_max 100 --t_0 100

SqueezeNet
{'lr': 0.0004766109491313375, 'optimizer': 'RMSprop', 'scheduler': 'CosineAnnealingWarmRestarts', 'rms_alpha': 0.30882554552480584, 'rms_momentum': 0.39423885159393096, 'warm_t_0': 96}

python -m src.main --model FeatExtSqueeze --epochs 1 --batch_size 20 --data-augmentation pretrain_augmentations --data-augmentation-validation validation_augmentations --optimizer rms --opti_momentum 0.39423885159393096 --opti_alpha 0.30882554552480584 --scheduler cosine_warm --t_max 100 --t_0 96

