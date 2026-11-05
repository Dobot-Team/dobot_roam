"""Run with torchrun --standalone --nproc_per_node=2 after fixture preparation."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytorch_lightning as pl
import torch
from vla.omni.data import OmniDataModule
from vla.omni.module import OmniLightningModule
from test_omni_lightning import TinyPlanner

config = json.loads(Path(sys.argv[1]).read_text())
rank = int(os.environ['RANK'])
world = int(os.environ['WORLD_SIZE'])
windows = []
for r in range(world):
    dm = OmniDataModule(config)
    dm.trainer = SimpleNamespace(global_rank=r, world_size=world)
    dm.setup('fit')
    windows.extend(next(iter(dm.train_dataloader()))['microbatches'])
os.environ['RANK'] = str(rank)
os.environ['WORLD_SIZE'] = str(world)
combined = {k: torch.cat([b[k] for b in windows]) for k in ['task_id', 'future_trajectory', 'future_trajectory_mask']}
reference = TinyPlanner()
optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
reference.get_loss_dict(combined, reference(combined))['optimization_loss'].backward()
optimizer.step()
module = OmniLightningModule(config, model=TinyPlanner())
trainer = pl.Trainer(accelerator='cpu', devices=world, strategy='ddp', max_steps=1,
                     logger=False, enable_checkpointing=False, enable_progress_bar=False,
                     enable_model_summary=False, use_distributed_sampler=False,
                     limit_val_batches=0, num_sanity_val_steps=0)
trainer.fit(module, datamodule=OmniDataModule(config))
torch.testing.assert_close(module.model.weight, reference.weight, rtol=1e-6, atol=1e-7)
print(f'DDP_PARITY_OK rank={rank} weight={module.model.weight.detach().tolist()}', flush=True)
