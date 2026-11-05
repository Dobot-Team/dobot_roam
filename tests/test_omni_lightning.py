"""Exercise loss parity, data ordering and Lightning checkpoint continuation."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

from vla.omni.build import build_dataset, omni_planner_collate_fn
from vla.omni.data import OmniDataModule
from vla.omni.module import OmniLightningModule
from vla.omni.ported.projects.vln.model.loss.task_aware_trajectory_loss import TaskAwareTrajectoryRegressionLoss
from test_omni_components import create_raw_manifest, tokenizer_file


class TinyPlanner(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.2, -.3, .1]))
        self.task_loss = TaskAwareTrajectoryRegressionLoss(loss_fn='mse', coord_scale=[2., 2., 1.])

    def forward(self, batch):
        return {'pred_future_trajectory': self.weight.view(1, 1, 3).expand(len(batch['task_id']), 8, 3)}

    def get_loss_dict(self, batch, output):
        return self.task_loss(batch, output)


def fixture_config(path, workers=0):
    raw = create_raw_manifest(path)
    ds = dict(type='DobotOmniRGBDataset', json_path=raw, num_input_frames=5,
              input_hz=2., img_size=64, waypoint_dt=.3, trajectory_dt=.1,
              history=31, raw_camera='cam1', mode='train', seed=42,
              task_sampling={'ratios': {'roaming': .3, 'intent': .2, 'point_nav': .3, 'image_nav': .2}, 'samples_per_epoch': 32},
              data_processor_list=[dict(type='OmniVLNPromptProcessor', tokenizer_path=tokenizer_file(path),
                                       use_origin_prompt=True, image_token_id=62, video_token_id=63,
                                       vision_start_id=60, vision_end_id=61)])
    loader = dict(type='DataLoader', dataset=ds, batch_size=2, num_workers=workers, drop_last=True,
                  collate_fn={'type': 'omni_planner_collate_fn'})
    val = deepcopy(loader); val['dataset']['mode'] = 'val'
    return {'seed': 42, 'model': {'model_type': 'qwen35_omni'},
            'data': {'training_dataloader': loader, 'validation_dataloader': val},
            'training': {'gradient_accum_steps': 2, 'gradient_clip_val': 100.,
                         'optimizer': {'type': 'SGD', 'lr': .03},
                         'lr_scheduler': {'type': 'ConstantLR', 'factor': 1., 'total_iters': 100}}}


def trainer(path, steps, callbacks=None):
    return pl.Trainer(accelerator='cpu', devices=1, max_steps=steps, max_epochs=-1,
                      logger=False, enable_progress_bar=False, enable_model_summary=False,
                      default_root_dir=path, callbacks=callbacks or [],
                      enable_checkpointing=bool(callbacks), use_distributed_sampler=False,
                      num_sanity_val_steps=0, limit_val_batches=0)


@pytest.mark.parametrize('workers', [0, 2])
def test_lightning_gradient_matches_original_effective_batch(tmp_path, workers):
    cfg = fixture_config(tmp_path, workers)
    dm = OmniDataModule(cfg); t = trainer(tmp_path, 1); dm.trainer = t; dm.setup('fit')
    window = next(iter(dm.train_dataloader()))['microbatches']
    combined = {k: torch.cat([b[k] for b in window]) for k in ['task_id', 'future_trajectory', 'future_trajectory_mask']}
    reference = TinyPlanner(); optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
    reference.get_loss_dict(combined, reference(combined))['optimization_loss'].backward()
    optimizer.step()
    module = OmniLightningModule(cfg, model=TinyPlanner())
    t.fit(module, datamodule=OmniDataModule(cfg))
    torch.testing.assert_close(module.model.weight, reference.weight, rtol=1e-6, atol=1e-7)
    assert t.global_step == 1


@pytest.mark.parametrize('workers', [0, 2])
@pytest.mark.parametrize('stop_at,total', [(1, 3), (8, 9)])
def test_lightning_checkpoint_continues_data_and_optimizer(tmp_path, workers, stop_at, total):
    cfg = fixture_config(tmp_path, workers)
    full = OmniLightningModule(cfg, model=TinyPlanner())
    trainer(tmp_path/'full', total).fit(full, datamodule=OmniDataModule(cfg))
    callback = ModelCheckpoint(dirpath=tmp_path/'ckpts', every_n_train_steps=1, save_last=True)
    first = OmniLightningModule(cfg, model=TinyPlanner())
    trainer(tmp_path/'first', stop_at, [callback]).fit(first, datamodule=OmniDataModule(cfg))
    checkpoint = callback.last_model_path
    payload = torch.load(checkpoint, weights_only=False)
    assert payload['OmniDataModule']['completed_steps'] == stop_at
    resumed = OmniLightningModule(cfg, model=TinyPlanner())
    dm = OmniDataModule(cfg)
    t = trainer(tmp_path/'resume', total)
    t.fit(resumed, datamodule=dm, ckpt_path=checkpoint)
    torch.testing.assert_close(resumed.model.weight, full.model.weight, rtol=1e-6, atol=1e-7)
    assert t.global_step == total
    assert dm.completed_steps == total % 8


def test_validation_aggregates_valid_points(tmp_path):
    cfg = fixture_config(tmp_path)
    module = OmniLightningModule(cfg, model=TinyPlanner())
    t = pl.Trainer(accelerator='cpu', devices=1, logger=False, enable_checkpointing=False,
                   enable_progress_bar=False, use_distributed_sampler=False, limit_val_batches=2)
    metrics = t.validate(module, datamodule=OmniDataModule(cfg), verbose=False)[0]
    assert np.isfinite(metrics['val/ade'])
    assert np.isfinite(metrics['val/fde'])
    assert metrics['val/total_loss'] >= 0


@pytest.mark.parametrize('checkpointing', [True, False])
def test_shared_entrypoint_checkpoint_options(tmp_path, monkeypatch, checkpointing):
    from vla.omni import train as entrypoint
    cfg = fixture_config(tmp_path)
    cfg['training'].update(save_every_n_steps=1, save_total_limit=2)
    cfg['trainer'] = dict(accelerator='cpu', devices=1, strategy='auto',
                          max_steps=1, max_epochs=-1, precision='32-true',
                          val_check_interval=100, limit_val_batches=0,
                          log_every_n_steps=1, num_sanity_val_steps=0,
                          enable_progress_bar=False, enable_model_summary=False,
                          enable_checkpointing=checkpointing)
    cfg['output_dir'] = str(tmp_path/'run')
    module = OmniLightningModule(cfg, model=TinyPlanner())
    monkeypatch.setattr(entrypoint, 'OmniLightningModule', lambda config: module)
    entrypoint.train_omni(cfg)
    checkpoint_dir = tmp_path/'run/checkpoints'
    assert (checkpoint_dir/'config.yaml').is_file()
    if checkpointing:
        assert (checkpoint_dir/'last.ckpt').is_symlink()
        assert (checkpoint_dir/'last.ckpt').resolve().is_file()
    else:
        assert not list(checkpoint_dir.glob('*.ckpt'))
