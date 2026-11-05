"""Omni branch of the project's shared scripts/train.py entry point."""
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from .data import OmniDataModule
from .module import OmniLightningModule


def train_omni(config, resume=None):
    trainer_config = dict(config['trainer'])
    if trainer_config.pop('accumulate_grad_batches', 1) != 1:
        raise ValueError('Use training.gradient_accum_steps for Omni; Lightning accumulation must be 1')
    if trainer_config.pop('gradient_clip_val', None) is not None:
        raise ValueError('Use training.gradient_clip_val for Omni manual optimization')
    pl.seed_everything(int(config.get('seed', 42)), workers=True)
    torch.set_float32_matmul_precision('high')
    output = Path(config['output_dir'])
    checkpoints = output / 'checkpoints'
    dm = OmniDataModule(config)
    module = OmniLightningModule(config)
    initial = config['training'].get('init_checkpoint')
    if initial and resume:
        raise ValueError('init_checkpoint and --resume are mutually exclusive')
    if initial:
        payload = torch.load(initial, map_location='cpu', weights_only=False)
        state = payload.get('state_dict', payload)
        if state and all(k.startswith('model.') for k in state):
            state = {k.removeprefix('model.'): v for k, v in state.items()}
        module.model.load_state_dict(state, strict=True)
    callbacks = [LearningRateMonitor(logging_interval='step')]
    if trainer_config.get('enable_checkpointing', True):
        callbacks.append(ModelCheckpoint(dirpath=checkpoints, filename='omni-{step:06d}',
                        every_n_train_steps=int(config['training']['save_every_n_steps']),
                        save_top_k=int(config['training']['save_total_limit']),
                        monitor='step', mode='max', save_last='link', save_on_train_epoch_end=False))
    # The preserved recipe validates every 50,000 optimizer steps, across
    # epoch boundaries. A fresh 9,000-step run therefore does no periodic val.
    trainer = pl.Trainer(**trainer_config, accumulate_grad_batches=1,
                         check_val_every_n_epoch=None, use_distributed_sampler=False,
                         callbacks=callbacks,
                         logger=[TensorBoardLogger(str(output), name='logs'), CSVLogger(str(output), name='csv')])
    if trainer.is_global_zero:
        checkpoints.mkdir(parents=True, exist_ok=True)
        for destination in (output / 'config.yaml', checkpoints / 'config.yaml'):
            destination.write_text(yaml.safe_dump(config, sort_keys=False))
    trainer.fit(module, datamodule=dm, ckpt_path=resume)
