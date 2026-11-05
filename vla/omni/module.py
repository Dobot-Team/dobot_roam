"""Qwen3.5 Omni policy trained by Lightning with the original objective."""
from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.distributed as dist

from .build import build_model
from .ported.projects.vln.constant.omni import TASK_NAMES
from .ported.utils.build_optimizer import build_optimizer
from .ported.utils.build_lr_scheduler import build_lr_scheduler


def move_tensors(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_tensors(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_tensors(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(v, device) for v in value)
    return value


def sum_across_ranks(value):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


class OmniLightningModule(pl.LightningModule):
    def __init__(self, config, model=None):
        super().__init__()
        self.config = deepcopy(config)
        self.save_hyperparameters({'config': self.config})
        self.model = build_model(config['model']['planner']) if model is None else model
        self.automatic_optimization = False

    def forward(self, batch):
        return self.model(batch)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, dict) and 'microbatches' in batch:
            return batch
        return move_tensors(batch, device)

    def configure_optimizers(self):
        optimizer = build_optimizer(self.config['training']['optimizer'], self.model)
        scheduler = build_lr_scheduler(self.config['training']['lr_scheduler'], optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        window = batch['microbatches']
        if len(window) != int(self.config['training']['gradient_accum_steps']):
            raise ValueError('Unexpected Omni accumulation window size')
        denominator = sum(self.model.task_loss.denominators(b) for b in window).to(self.device)
        sum_across_ranks(denominator)
        optimizer = self.optimizers()
        optimizer.zero_grad()
        statistics = {}
        for microbatch in window:
            inputs = move_tensors(microbatch, self.device)
            inputs['_task_loss_denominators'] = denominator
            outputs = self(inputs)
            losses = self.model.get_loss_dict(inputs, outputs)
            # The denominator already includes every microbatch and rank.
            # TaskAware loss compensates DDP's gradient averaging by world_size.
            self.manual_backward(losses['optimization_loss'])
            for name, values in losses['_loss_statistics'].items():
                statistics[name] = statistics.get(name, 0) + values.detach()
            del outputs, losses, inputs
        self.clip_gradients(optimizer, gradient_clip_val=float(self.config['training']['gradient_clip_val']), gradient_clip_algorithm='norm')
        optimizer.step()
        self.lr_schedulers().step()
        self.trainer.datamodule.commit_step()
        total = self._log_statistics(statistics, 'train', on_step=True)
        return {'loss': total}

    def _log_statistics(self, statistics, prefix, on_step=False):
        total = torch.zeros((), device=self.device)
        for task in TASK_NAMES:
            name = f'loss/{task}/trajectory'
            values = sum_across_ranks(statistics[name].clone())
            raw = values[0] / values[1].clamp_min(1e-12)
            total += raw
            self.log(f'{prefix}/{name}', raw, on_step=on_step, on_epoch=not on_step, sync_dist=False, batch_size=1)
        self.log(f'{prefix}/total_loss', total, on_step=on_step, on_epoch=not on_step, sync_dist=False, batch_size=1)
        return total

    def on_train_epoch_end(self):
        self.trainer.datamodule.finish_epoch()

    def on_validation_epoch_start(self):
        self._val_statistics = {f'loss/{t}/trajectory': torch.zeros(3, device=self.device) for t in TASK_NAMES}
        self._val_distance = torch.zeros(4, device=self.device, dtype=torch.float64)

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        losses = self.model.get_loss_dict(batch, output)
        for name, values in losses['_loss_statistics'].items():
            self._val_statistics[name] += values.detach()
        valid = batch['future_trajectory_mask'].bool()
        distance = (output['pred_future_trajectory'][..., :2].float() - batch['future_trajectory'][..., :2].float()).norm(dim=-1)
        self._val_distance[0] += distance[valid].sum()
        self._val_distance[1] += valid.sum()
        indices = torch.arange(valid.shape[1], device=valid.device).expand_as(valid).masked_fill(~valid, -1).max(dim=1).values
        rows = (indices >= 0).nonzero(as_tuple=True)[0]
        self._val_distance[2] += distance[rows, indices[rows]].sum()
        self._val_distance[3] += len(rows)

    def on_validation_epoch_end(self):
        self._log_statistics(self._val_statistics, 'val')
        d = sum_across_ranks(self._val_distance)
        self.log('val/ade', d[0] / d[1].clamp_min(1), sync_dist=False)
        self.log('val/fde', d[2] / d[3].clamp_min(1), sync_dist=False)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['omni_format_version'] = 1

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('omni_format_version') != 1:
            raise ValueError('--resume requires an Omni Lightning checkpoint; use training.init_checkpoint for DBN weights')
