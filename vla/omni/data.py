"""Lightning data integration preserving task plans and microbatch order."""
from copy import deepcopy
import math
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .build import build_dataset, omni_planner_collate_fn


class OptimizerBatchLoader:
    """Group consecutive microbatches before computing the task denominators.

    Each item is one optimizer step. Images stay on CPU until their individual
    forward pass, so accumulation does not multiply GPU activation memory.
    Progress records committed steps, never prefetched samples.
    """
    def __init__(self, owner):
        self.owner = owner

    def __len__(self):
        return self.owner.steps_per_epoch

    def __iter__(self):
        dm = self.owner
        ds = dm.train_dataset
        ds.set_data_epoch(dm.data_epoch)
        loader = dm.make_loader(ds, dm.train_config, epoch=dm.data_epoch)
        iterator = iter(loader)
        try:
            for _ in range(dm.completed_steps * dm.accumulation):
                next(iterator)
            while True:
                window = []
                for _ in range(dm.accumulation):
                    try:
                        window.append(next(iterator))
                    except StopIteration:
                        if window:
                            raise RuntimeError('Incomplete Omni accumulation window; align samples_per_epoch with batch/workers/world/accumulation')
                        return
                yield {'microbatches': window}
        finally:
            shutdown = getattr(iterator, '_shutdown_workers', None)
            if shutdown is not None:
                shutdown()


class OmniDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = deepcopy(config)
        self.accumulation = int(config['training']['gradient_accum_steps'])
        if self.accumulation < 1:
            raise ValueError('gradient_accum_steps must be positive')
        self.train_config = config['data']['training_dataloader']
        self.train_dataset = None
        self.val_dataset = None
        self.data_epoch = 0
        self.completed_steps = 0
        self._resume_topology = None

    @property
    def topology(self):
        return {'world_size': self.trainer.world_size,
                'num_workers': int(self.train_config.get('num_workers', 0)),
                'batch_size': int(self.train_config['batch_size']),
                'accumulation': self.accumulation}

    def setup(self, stage=None):
        # The port's iterable dataset shards itself; Lightning must not add a
        # DistributedSampler. Set global rank explicitly for spawned DDP too.
        os.environ['RANK'] = str(self.trainer.global_rank)
        os.environ['WORLD_SIZE'] = str(self.trainer.world_size)
        if self.train_dataset is None:
            self.train_dataset = build_dataset(self.train_config['dataset'])
        val = self.config['data'].get('validation_dataloader')
        if val and self.val_dataset is None:
            self.val_dataset = build_dataset(val['dataset'])
        if self._resume_topology is not None and self._resume_topology != self.topology:
            raise ValueError('Omni data resume requires the same world_size, workers, batch_size and accumulation')
        if self.completed_steps > self.steps_per_epoch:
            raise ValueError('Checkpoint data cursor exceeds the current epoch')

    @property
    def steps_per_epoch(self):
        cfg = self.train_config
        workers = max(1, int(cfg.get('num_workers', 0)))
        width = workers * self.trainer.world_size
        shard_batch = self.train_dataset.worker_shard_batch_size or 1
        usable = len(self.train_dataset) // (width * shard_batch) * (width * shard_batch)
        per_worker = usable // width
        size = int(cfg['batch_size'])
        batches = workers * (per_worker // size if cfg.get('drop_last', True) else math.ceil(per_worker / size))
        if batches == 0 or batches % self.accumulation:
            raise ValueError(f'{batches} microbatches per rank cannot form full accumulation windows of {self.accumulation}; adjust samples_per_epoch')
        return batches // self.accumulation

    def make_loader(self, dataset, config, epoch=0):
        options = {k: v for k, v in config.items() if k not in {'type', 'dataset', 'collate_fn'}}
        if config.get('collate_fn', {}).get('type') != 'omni_planner_collate_fn':
            raise ValueError('Omni requires omni_planner_collate_fn')
        # Recreate workers each epoch, matching the original fetcher, so every
        # worker sees the updated deterministic task/goal plan.
        options['persistent_workers'] = False
        if not options.get('num_workers', 0):
            options.pop('prefetch_factor', None)
        generator = torch.Generator().manual_seed(int(self.config.get('seed', 42)) + epoch + 100003 * self.trainer.global_rank)
        return DataLoader(dataset, collate_fn=omni_planner_collate_fn, generator=generator, **options)

    def train_dataloader(self):
        return OptimizerBatchLoader(self)

    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        self.val_dataset.set_data_epoch(0)
        return self.make_loader(self.val_dataset, self.config['data']['validation_dataloader'])

    def commit_step(self):
        self.completed_steps += 1

    def finish_epoch(self):
        if self.completed_steps == self.steps_per_epoch:
            self.data_epoch += 1
            self.completed_steps = 0

    def state_dict(self):
        return {'data_epoch': self.data_epoch, 'completed_steps': self.completed_steps, 'topology': self.topology}

    def load_state_dict(self, state_dict):
        self.data_epoch = int(state_dict['data_epoch'])
        self.completed_steps = int(state_dict['completed_steps'])
        self._resume_topology = state_dict['topology']
        if self._resume_topology != self.topology:
            raise ValueError('Omni data resume requires the same world_size, workers, batch_size and accumulation')
