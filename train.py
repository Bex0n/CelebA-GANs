import argparse
import yaml

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from dataset import CelebADataset
from models.wgan import (
    Discriminator,
    Generator,
    WGAN
)

torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser(description='Trainer for GANs')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/wgan.yaml')
parser.add_argument('--checkpoint', '-p',
                    dest="checkpoint",
                    metavar='FILE',
                    help='path to model checkpoint',
                    default=None)
args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(e)
data_params = config['data_params']
hardware_params = config['hardware_params']
model_params = config['model_params']
trainig_params = config['training_params']

wgan = WGAN(**model_params)
if args.checkpoint:
    print('Loading model from checkpoint.')
    wgan = WGAN.load_from_checkpoint(args.checkpoint, **model_params)

"""
A work-around to address issues with pytorch's celebA dataset class.

Download and Extract
URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
"""
data = CelebADataset(data_path=data_params['data_path'],
                     batch_size=data_params['batch_size'],
                     num_workers=data_params['num_workers'],
                     pin_memory=data_params['pin_memory'])
# data.prepare_data()
data.setup()

tb_logger = TensorBoardLogger(save_dir='logs/', name='WGAN')
trainer = Trainer(logger=tb_logger,
                  accelerator=config['hardware_params']['accelerator'],
                  devices=config['hardware_params']['devices'],
                  max_epochs=config['training_params']['max_epochs'])
trainer.fit(model=wgan, datamodule=data)
