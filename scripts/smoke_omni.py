#!/usr/bin/env python3
"""Run a bounded real-data Omni Lightning check without changing the recipe."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import yaml
from vla.omni.train import train_omni


def subset_manifest(source, destination, count=12):
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with source.open('rb') as stream:
        for i in range(count):
            stream.seek(source.stat().st_size * i // count)
            if i:
                stream.readline()
            line = stream.readline()
            if line:
                rows.append(json.loads(line))
    destination.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/omni_v2/qwen35_08B_omni_v2_base_v2v3data.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    original = config['data']['training_dataloader']['dataset']
    roots = [original['json_path'], *original['additional_json_paths']]
    subset_roots = []
    for index, root in enumerate(roots):
        root = Path(root)
        destination = out / 'manifests' / str(index)
        manifests = list(root.rglob('*.jsonl')) if root.is_dir() else [root]
        if not manifests:
            raise FileNotFoundError(f'No manifests under {root}')
        for source in manifests:
            subset_manifest(source, destination / source.relative_to(root if root.is_dir() else root.parent))
        subset_roots.append(str(destination))
    for key in ('training_dataloader', 'validation_dataloader'):
        loader = config['data'][key]
        loader.update(batch_size=2, num_workers=0, persistent_workers=False)
        ds = loader['dataset']
        ds['json_path'] = subset_roots[0]
        ds['additional_json_paths'] = subset_roots[1:]
        ds['task_sampling']['samples_per_epoch'] = 80 * args.devices
        ds['resample_on_error'] = False
    config['trainer'].update(devices=args.devices, max_steps=args.steps, num_nodes=1,
                             limit_val_batches=2, val_check_interval=1, log_every_n_steps=1,
                             enable_model_summary=False)
    if args.devices == 1:
        config['trainer']['strategy'] = 'auto'
    config['training'].update(save_every_n_steps=1, save_total_limit=2)
    config['output_dir'] = str(out / 'run')
    destination = out / 'smoke.yaml'
    destination.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f'Smoke config: {destination}', flush=True)
    if not args.prepare_only:
        torch.set_num_threads(2)
        train_omni(config)


if __name__ == '__main__':
    main()
