import os
import argparse
import logging
import time
import numpy as np
import copy

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchsummary import summary
from torchvision.datasets import ImageFolder

from src.cnn import *
from src.eval.evaluate import eval_fn, accuracy
from src.training import train_fn
from src.data_augmentations import *


def set_up_data_aug(data_aug_train, data_aug_val):
    if data_aug_train is None:
        data_aug_train = transforms.ToTensor()
    elif isinstance(data_aug_train, list):
        data_aug_train = transforms.Compose(data_aug_train)
    elif not isinstance(data_aug_train, transforms.Compose):
        raise NotImplementedError
    #  for Validation set
    if data_aug_val is None:
        data_aug_val = transforms.ToTensor()
    elif isinstance(data_aug_val, list):
        data_aug_val = transforms.Compose(data_aug_val)
    elif not isinstance(data_aug_val, transforms.Compose):
        raise NotImplementedError

    return data_aug_train, data_aug_val


def main(data_dir,
         torch_model,
         num_epochs=10,
         batch_size=50,
         learning_rate=0.001,
         train_criterion=torch.nn.CrossEntropyLoss,
         model_optimizer='adam',
         momentum=0.9,
         alpha=0.99,
         model_schedule='cosine',
         t_0=150,
         t_max=150,
         data_augmentations=None,
         data_augmentations_validation=None,
         save_model_str=None,
         use_all_data_to_train=False,
         exp_name=''):
    """
    Training loop for configurableNet.
    :param data_dir: dataset path (str)
    :param num_epochs: (int)
    :param batch_size: (int)
    :param learning_rate: model optimizer learning rate (float)
    :param train_criterion: Which loss to use during training (torch.nn._Loss)
    :param model_optimizer: Which model optimizer to use during training (torch.optim.Optimizer)
    :param momentum: for RMSProp and SGD
    :param alpha: for RMSProp
    :param data_augmentations: List of data augmentations to apply such as rescaling.
        (list[transformations], transforms.Composition[list[transformations]], None)
        If none only ToTensor is used
    :param data_augmentations_validation: same as above.
    :param model_schedule: What scheduler to use
    :param t_max: for Cosine Annealing
    :param t_0: for Cosine with warm restart
    :return:
    """

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_augmentations, data_augmentations_validation = set_up_data_aug(data_aug_train=data_augmentations,
                                                                        data_aug_val=data_augmentations_validation)

    # Load the dataset
    train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=data_augmentations)
    val_data = ImageFolder(os.path.join(data_dir, 'val'), transform=data_augmentations_validation)
    test_data = ImageFolder(os.path.join(data_dir, 'test'),
                            transform=data_augmentations_validation)

    channels, img_height, img_width = train_data[0][0].shape

    # image size
    input_shape = (channels, img_height, img_width)

    # instantiate training criterion
    train_criterion = train_criterion().to(device)
    score = []

    if use_all_data_to_train:
        train_loader = DataLoader(dataset=ConcatDataset([train_data, val_data, test_data]),
                                  batch_size=batch_size,
                                  shuffle=True)
        logging.warning('Training with all the data (train, val and test).')
    else:
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_data,
                                batch_size=batch_size,
                                shuffle=False)

    model = torch_model(input_shape=input_shape,
                        num_classes=len(train_data.classes)).to(device)
    # save best weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0.0
    train_params = model.param_to_train()
    # instantiate optimizer and scheduler
    # TODO: play with this
    print(f"optimiser and scheduler details- {model_schedule}/{model_optimizer}/{alpha}/{momentum}/{t_0}/{t_max}")
    if model_optimizer == 'adam':
        optimizer = torch.optim.Adam(train_params, lr=learning_rate)
    elif model_optimizer == 'rms':
        optimizer = torch.optim.RMSprop(train_params, lr=learning_rate, momentum=momentum,
                                        alpha=alpha)
    else:
        optimizer = torch.optim.SGD(train_params, lr=learning_rate, momentum=momentum)

    if model_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_max)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, t_0)

    # Info about the model being trained
    # You can find the number of learnable parameters in the model here
    logging.info('Model being trained:')
    summary(model, input_shape,
            device='cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    for epoch in range(num_epochs):
        logging.info('#' * 50)
        logging.info('Epoch [{}/{}]'.format(epoch + 1, num_epochs))

        train_score, train_loss = train_fn(model, optimizer, train_criterion, train_loader, device)
        logging.info('Train accuracy: %f', train_score)

        if not use_all_data_to_train:
            test_score = eval_fn(model, val_loader, device)
            logging.info('Validation accuracy: %f', test_score)
            score.append(test_score)
            # update weights for best accuracy
            if test_score > best_score:
                best_score = test_score
                best_model_wts = copy.deepcopy(model.state_dict())
        else:
            # use the best train score. what else?
            if train_score > best_score:
                best_score = train_score
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

    # save best weighst to model
    print(f"best score: {best_score}")
    model.load_state_dict(best_model_wts)
    if save_model_str:
        # Save the model checkpoint can be restored via "model = torch.load(save_model_str)"
        model_save_dir = os.path.join(os.getcwd(), save_model_str)

        if not os.path.exists(model_save_dir):
            os.mkdir(model_save_dir)

        save_model_str = os.path.join(model_save_dir, exp_name + '_model_' + str(int(time.time())))
        torch.save(model.state_dict(), save_model_str)

    if not use_all_data_to_train:
        logging.info('Accuracy at each epoch: ' + str(score))
        logging.info('Mean of accuracies across all epochs: ' + str(100 * np.mean(score)) + '%')
        logging.info('Accuracy of model at final epoch: ' + str(100 * score[-1]) + '%')


if __name__ == '__main__':
    """
    This is just an example of a training pipeline.

    Feel free to add or remove more arguments, change default values or hardcode parameters to use.
    """
    loss_dict = {'cross_entropy': torch.nn.CrossEntropyLoss}  # Feel free to add more
    opti_dict = {'sgd': torch.optim.SGD, 'adam': torch.optim.Adam, 'rms': torch.optim.RMSprop}  # Feel free to add more
    scheduler_dict = {'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
                      'cosine_warm': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts}

    cmdline_parser = argparse.ArgumentParser('DL WS20/21 Competition')

    cmdline_parser.add_argument('-m', '--model',
                                default='SampleModel',
                                help='Class name of model to train',
                                type=str)
    cmdline_parser.add_argument('-e', '--epochs',
                                default=50,
                                help='Number of epochs',
                                type=int)
    cmdline_parser.add_argument('-b', '--batch_size',
                                default=282,
                                help='Batch size',
                                type=int)
    cmdline_parser.add_argument('-D', '--data_dir',
                                default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     '..', 'dataset'),
                                help='Directory in which the data is stored (can be downloaded)')
    cmdline_parser.add_argument('-l', '--learning_rate',
                                default=2.244958736283895e-05,
                                help='Optimizer learning rate',
                                type=float)
    cmdline_parser.add_argument('-L', '--training_loss',
                                default='cross_entropy',
                                help='Which loss to use during training',
                                choices=list(loss_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-o', '--optimizer',
                                default='adam',
                                help='Which optimizer to use during training',
                                choices=list(opti_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-om', '--opti_momentum',
                                default=0.9,
                                help='Momentum for SGD or RMSProp',
                                type=float)
    cmdline_parser.add_argument('-oa', '--opti_alpha',
                                default=0.99,
                                help='Alpha for RMSProp',
                                type=float)
    cmdline_parser.add_argument('-s', '--scheduler',
                                default='cosine',
                                help='Which scheduler to use during training',
                                choices=list(scheduler_dict.keys()),
                                type=str)
    cmdline_parser.add_argument('-tm', '--t_max',
                                default=150,
                                help='T_max for CosineAnnealingLR',
                                type=int)
    cmdline_parser.add_argument('-to', '--t_0',
                                default=150,
                                help='T_0 for CosineAnnealingWarmRestarts',
                                type=int)
    cmdline_parser.add_argument('-p', '--model_path',
                                default='models',
                                help='Path to store model',
                                type=str)
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('-n', '--exp_name',
                                default='default',
                                help='Name of this experiment',
                                type=str)
    cmdline_parser.add_argument('-d', '--data-augmentation',
                                default='resize_and_colour_jitter',
                                help='Data augmentation to apply to data before passing to the model.'
                                     + 'Must be available in data_augmentations.py')
    cmdline_parser.add_argument('-dv', '--data-augmentation-validation',
                                default='resize_to_64x64',
                                help='Data augmentation to apply to data before passing to the model.'
                                     + 'Must be available in data_augmentations.py')
    cmdline_parser.add_argument('-a', '--use-all-data-to-train',
                                action='store_true',
                                help='Uses the train, validation, and test data to train the model if enabled.')

    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    main(
        data_dir=args.data_dir,
        torch_model=eval(args.model),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_criterion=loss_dict[args.training_loss],
        model_optimizer=args.optimizer,
        momentum=args.opti_momentum,
        alpha=args.opti_alpha,
        model_schedule=args.scheduler,
        t_0=args.t_0,
        t_max=args.t_max,
        data_augmentations=eval(args.data_augmentation),
        data_augmentations_validation=eval(args.data_augmentation_validation),
        # Check data_augmentations.py for sample augmentations
        save_model_str=args.model_path,
        exp_name=args.exp_name,
        use_all_data_to_train=args.use_all_data_to_train
    )
