##### Which track(s) are you participating in? {Fast Networks Track, Large Networks Track, Both}
Fast Networks Track

##### What are the number of learnable parameters in your model?
Fast Networks Track -
Large Networks Track -

##### Briefly describe your approach
Fast Networks Track -
Large Networks Track - 
    1. Added separate command line arguments for validation data augmentation, 
        optimiser parameters, scheduler and scheduler parameters
    2. Created every model with <25Million parameters from torchvision.models. Impelmented feature extraction 
        which stores the weights used to give the highest test score in the given epochs.
    3. Added one more optimiser, 2 schedulers and related parameters.
    4. Added files for hyperparameter optimisation using BOHB
    5. Ran top 3 models for as long as possible. Selected the one with the best metrics. 


##### Command to train your model from scratch
Fast Networks Track -
Large Networks Track -


##### Command to evaluate your model
Fast Networks Track -
Large Networks Track -

Command to train
EfficientNet with hpo params added
python -m src.main --model --exp_name effnetb4 FeatExtEfficientNetB4 --epochs 1 --batch_size 20 --data-augmentation efficientnet_augmentations --data-augmentation-validation efficientnet_augmentations_valid --optimizer rms --learning_rate 0.006708615356600079 --opti_momentum 0.6417581190040514 --opti_alpha 0.3821706684860047 --scheduler cosine_warm --t_max 100 --t_0 95

SqueezeNet with hpo params added
python -m src.main --model FeatExtSqueeze --exp_name squeezenet --epochs 1 --batch_size 20 --data-augmentation pretrain_augmentations --data-augmentation-validation validation_augmentations --optimizer rms --learning_rate 0.0004766109491313375 --opti_momentum 0.39423885159393096 --opti_alpha 0.30882554552480584 --scheduler cosine_warm --t_max 100 --t_0 96

Resnet50 with hpo params added
python -m src.main --model FeatExtResnet50 --exp_name resnet50 --epochs 1 --batch_size 20 --data-augmentation pretrain_augmentations --data-augmentation-validation validation_augmentations --optimizer rms --learning_rate 0.0007048424134958733 --opti_momentum 0.7485456269263905 --opti_alpha  0.4289551813622236 --scheduler cosine --t_max 122 --t_0 96