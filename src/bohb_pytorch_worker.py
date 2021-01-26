import os
from src.cnn import *
from src.data_augmentations import *
import torchvision
import argparse
import logging
import time
import numpy as np
import copy
import logging
from torchsummary import summary
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import ImageFolder
from hpbandster.core.worker import Worker

logging.basicConfig(level=logging.DEBUG)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on device: {device}")


class PyTorchWorker(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # load dataset here, do the batching a bit later? I want batch as an HP

    @staticmethod
    def get_configspace() -> CS.Configuration:
        """ Define a conditional hyperparameter search-space.

        hyperparameters:
          lr              from 1e-6 to 1e-1 (float, log) and default=1e-2
          sgd_momentum    from 0.00 to 0.99 (float) and default=0.9
          optimizer            Adam or  SGD (categoric)
        conditions:
          include sgd_momentum  only if       optimizer = SGD
        Returns:
            Configurationspace

        Note:
            please name the hyperparameters as given above (needed for testing).
        Hint:
            use example = CS.EqualsCondition(..,..,..) and then
            cs.add_condition(example) to add a conditional hyperparameter
            for SGD's momentum.
        """
        cs = CS.ConfigurationSpace(seed=0)
        # START TODO ################
        lr_hp = CS.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value=1e-2, log=True)
        optimizer_hp = CSH.CategoricalHyperparameter(name='optimizer', choices=['Adam', 'SGD'])
        sgd_momentum_hp = CS.UniformFloatHyperparameter('sgd_momentum', lower=0.00, upper=0.99, default_value=0.9)

        sgd_cond = CS.EqualsCondition(sgd_momentum_hp, optimizer_hp, 'SGD')
        cs.add_hyperparameters([lr_hp, optimizer_hp, sgd_momentum_hp])
        cs.add_conditions([sgd_cond])
        # END TODO ################
        return cs

    def evaluate_accuracy(self, model, data_loader):
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in data_loader:
                output = model(x)
                # test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(y.view_as(pred)).sum().item()
        # import pdb; pdb.set_trace()
        accuracy = correct / len(data_loader.sampler)
        return accuracy

    def compute(self, config, budget, working_directory, *args, **kwargs):
        # Load the dataset
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset') + '/tiny'
        train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=resize_and_colour_jitter)
        val_data = ImageFolder(os.path.join(data_dir, 'val'), transform=resize_and_colour_jitter)
        test_data = ImageFolder(os.path.join(data_dir, 'test'), transform=resize_and_colour_jitter)

        channels, img_height, img_width = train_data[0][0].shape

        # image size
        input_shape = (channels, img_height, img_width)
        # instantiate training criterion
        # TODO: adjust this. somethine else?
        train_criterion = torch.nn.CrossEntropyLoss().to(device)

        # not usin all data to train
        # TODO: adjust this
        batch_size = 50  # change this
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
        val_loader = DataLoader(dataset=val_data,
                                batch_size=batch_size,
                                shuffle=False)
        # TODO: get actual test data
        test_loader = DataLoader(dataset=val_data,
                                 batch_size=batch_size,
                                 shuffle=True)

        # make a model yes, doing feature extraction here
        # TODO: play with the model
        model = SampleModel(input_shape=input_shape,
                            num_classes=len(train_data.classes)).to(device)
        train_params = model.parameters()  # model.param_to_train()
        # TODO: play with this
        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(train_params, lr=config['lr'])
        else:
            optimizer = torch.optim.SGD(train_params, lr=config['lr'], momentum=config['sgd_momentum'])

        logging.info('Model being trained:')
        for epoch in range(int(budget)):
            loss = 0
            model.train()
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(x)
                loss = train_criterion(output, y)
                loss.backward()
                optimizer.step()

        train_accuracy = self.evaluate_accuracy(model, train_loader)
        validation_accuracy = self.evaluate_accuracy(model, val_loader)
        test_accuracy = self.evaluate_accuracy(model, test_loader)
        return ({
            'loss': 1 - validation_accuracy,  # remember: HpBandSter always minimizes!
            'info': {'test accuracy': test_accuracy,
                     'train accuracy': train_accuracy,
                     'validation accuracy': validation_accuracy,
                     'model': str(model), }
        })
