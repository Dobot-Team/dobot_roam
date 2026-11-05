"""Behavior tests for Omni tasks, visual inputs and distributed task losses."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers

from vla.omni.ported.projects.vln.constant.omni import TASK_NAMES, SOURCE_NAMES
from vla.omni.ported.projects.vln.data.dataset.omni_sampling import OmniTaskSourceSampler
from vla.omni.ported.projects.vln.data.dataset.data_processor.omni_tasks import (
    TrackingTaskProcessor, IntentTaskProcessor, PointNavigationTaskProcessor,
    ImageNavigationTaskProcessor, stop_at_goal)
from vla.omni.ported.projects.vln.data.dataset.data_processor.omni_prompt_processor import OmniVLNPromptProcessor
from vla.omni.ported.projects.vln.data.dataset.dobot_omni_rgb_dataset import DobotOmniRGBDataset
from vla.omni.ported.projects.vln.data.dataset.dobot_omtrack_rgb_dataset import DobotOmTrackRGBDataset
from vla.omni.ported.projects.vln.data.collate_fn.omni_planner_collate import omni_planner_collate_fn
from vla.omni.ported.projects.vln.model.loss.task_aware_trajectory_loss import TaskAwareTrajectoryRegressionLoss

torch.set_num_threads(1)


def loss_batch(tasks, pred_values, mask=None):
    p = torch.tensor(pred_values, dtype=torch.float32).view(-1, 1, 1).expand(-1, 8, 3).clone().requires_grad_()
    b = {"task_id": torch.tensor(tasks), "future_trajectory": torch.zeros_like(p),
         "future_trajectory_mask": torch.ones(len(tasks), 8, dtype=torch.bool) if mask is None else mask}
    return b, {"pred_future_trajectory": p}


@pytest.mark.parametrize("tau,dt", [(0.3, .3), (.5, .3), (.9, .3), (2.4, .3), (3.1, .3), (.9, .6)])
def test_stop_target_preserves_existing_clock_and_prefix(tau, dt):
    original = torch.arange(24).float().view(8, 3)
    s = {"future_trajectory": original.clone(), "future_trajectory_mask": torch.ones(8, dtype=torch.bool),
         "_waypoint_dt": dt, "_future_goal": {"time_s": tau, "pose": [1.2, -.4, .3]}}
    stop_at_goal(s)
    after = torch.arange(1, 9).double() * dt >= tau-1e-7
    assert torch.equal(s["future_trajectory"][~after], original[~after])
    assert torch.allclose(s["future_trajectory"][after], torch.tensor([1.2, -.4, .3]).expand(int(after.sum()), 3))
    assert s["future_trajectory_mask"].all()


def test_task_source_marginals_and_epoch_reproducibility():
    pools = {s: list(range(i*20, i*20+20)) for i, s in enumerate(SOURCE_NAMES)}
    tasks = {"ratios": {"tracking": .4, **{t: .15 for t in TASK_NAMES[1:]}},
             "samples_per_epoch": 1000, "seed": 42}
    sources = {"mode": "global_marginals", "ratios": {
        "dobot_tracking": .2, "dobot_roaming": .6, "trackvla_at": .08,
        "trackvla_dt": .06, "trackvla_stt": .06}}
    sampler = OmniTaskSourceSampler(tasks, sources, pools, 100)
    p = sampler.plan(1)
    assert np.bincount(p[:, 0]).tolist() == [400, 150, 150, 150, 150]
    assert np.bincount(p[:, 1]).tolist() == [200, 600, 80, 60, 60]
    assert np.array_equal(p, sampler.plan(1))
    assert not np.array_equal(p, sampler.plan(2))
    sources["ratios"]["dobot_roaming"] = .2
    with pytest.raises(ValueError, match="incompatible"):
        OmniTaskSourceSampler(tasks, sources, pools, 100)


def test_weighted_backward_unweighted_logging_and_mask():
    batch, pred = loss_batch([0, 1], [math.sqrt(2), math.sqrt(3)])
    loss = TaskAwareTrajectoryRegressionLoss(loss_fn="mse", global_loss_scale=10,
                                            loss_weights={"roaming": 4})
    result = loss(batch, pred)
    assert result["total_loss"].item() == pytest.approx(5)
    assert result["optimization_loss"].item() == pytest.approx(140)
    result["optimization_loss"].backward()
    original = result["loss/roaming/trajectory"].item()
    loss.task_weights[1] = 0
    other = loss(batch, pred)
    assert other["loss/roaming/trajectory"].item() == original
    assert other["total_loss"].item() == pytest.approx(5)
    assert other["optimization_loss"].item() == pytest.approx(20)
    batch["future_trajectory_mask"][0, -1] = False
    batch["future_trajectory"][0, -1] = float("nan")
    assert torch.isfinite(loss(batch, pred)["optimization_loss"])


def tokenizer_file(tmp_path):
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "SYSTEM": 10, "COMMAND": 11}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    p = tmp_path/"tokenizer.json"
    tok.save(str(p))
    return str(p)


def prompt_processor(tmp_path):
    return OmniVLNPromptProcessor(tokenizer_path=tokenizer_file(tmp_path),
        system_prompt="SYSTEM", use_origin_prompt=True, vision_token_type="video",
        split_video_by_timestamps=True, image_token_id=62, video_token_id=63,
        vision_start_id=60, vision_end_id=61)


def visual_sample(goal=False, task=0):
    s = dict(rgb=torch.rand(5, 64, 64, 3)*255, instruction_text="COMMAND",
        _num_video_tokens=12, _video_grid_t=3, _tokens_per_video_grid=4,
        _input_hz=2., _temporal_patch_size=2, _video_grid_timestamps=torch.tensor([0., 1., 2.]),
        future_trajectory=torch.zeros(8,3), future_trajectory_mask=torch.ones(8,dtype=torch.bool),
        task_id=task, source_id=0)
    if goal:
        s["goal_rgb"] = torch.rand(64,64,3)*255
        s["instruction_parts"] = [{"text":"COMMAND "}, {"image":"goal_rgb"}, {"text":"."}]
    return s


def small_model():
    from vla.omni.ported.projects.vln.model.qwen3_5.vlm import Qwen35VLM
    from vla.omni.ported.projects.vln.model.head.omtrack_planner_head import OmTrackPlannerHead
    from vla.omni.ported.projects.vln.model.arch.omni_vln_planner import OmniVLNPlannerWithActToken
    text = dict(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, hidden_act="silu",
        rms_norm_eps=1e-6, tie_word_embeddings=True, attention_bias=False, attn_output_gate=True,
        linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_num_key_heads=4, linear_num_value_heads=4, full_attention_interval=4,
        max_position_embeddings=4096, rope_parameters=dict(rope_type="default",
        rope_theta=10000000, partial_rotary_factor=.25, mrope_interleaved=True, mrope_section=[2,1,1]))
    vision = dict(depth=2, hidden_size=64, intermediate_size=128, hidden_act="gelu_pytorch_tanh",
        num_heads=4, in_channels=3, patch_size=16, spatial_merge_size=2, temporal_patch_size=2,
        out_hidden_size=64, num_position_embeddings=256)
    return OmniVLNPlannerWithActToken(
        vision_language_model=Qwen35VLM(text_config=text, vision_config=vision, treat_image_sequence_as_video=True),
        trajectory_pred_head=OmTrackPlannerHead(input_channels=64, hidden_channels=[64]),
        task_loss=TaskAwareTrajectoryRegressionLoss(loss_fn="mse"),
        image_token_id=62, video_token_id=63, use_uint8_input_images=True)


@pytest.mark.parametrize("goal_flags", [(False,False), (True,True), (False,True)])
def test_mixed_visual_forward_backward_and_order(tmp_path, goal_flags):
    processor = prompt_processor(tmp_path)
    samples = [visual_sample(goal, 4 if goal else 0) for goal in goal_flags]
    for s in samples:
        processor(s)
        ids = s["prompt_ids"].tolist()
        assert ids.index(10) < ids.index(63) < ids.index(11)
        if "goal_rgb" in s:
            assert ids.index(11) < ids.index(62)
    batch = omni_planner_collate_fn(samples)
    if "goal_rgb" in batch:
        batch["goal_rgb"].requires_grad_()
    model = small_model()
    output = model(batch)
    assert output["pred_future_trajectory"].shape == (2,8,3)
    losses = model.get_loss_dict(batch, output)
    losses["optimization_loss"].backward()
    assert model.act_token.grad is not None
    assert next(model.vision_encoder.parameters()).grad is not None
    if "goal_rgb" in batch:
        assert batch["goal_rgb"].grad.abs().sum() > 0
    merged = output["mm_merged_prompt_ids"]
    attn = torch.cat([batch["attention_mask"], torch.ones(2,1,dtype=torch.long)], dim=1)
    pos = model._build_position_ids(attn, merged, batch)
    assert pos.shape == (3,2,merged.shape[1])
    for b, goal in enumerate(goal_flags):
        if goal:
            grid = pos[:,b,merged[b]==62]
            assert torch.unique(grid[0]).numel() == 1
            assert torch.unique(grid[1]).numel() == 2
            assert torch.unique(grid[2]).numel() == 2


def create_raw_manifest(tmp_path):
    bag = tmp_path/"bag";cam=bag/"cam1";cam.mkdir(parents=True)
    frames=[];labels=[];base=1_700_000_000.
    for i in range(70):
        ns=round((base+i*.1)*1e9)
        p=cam/f"{ns}.jpg"
        Image.new("RGB",(64,64),(i%255,30,60)).save(p)
        frames.append({"timestamp":ns,"cam1":str(p)})
        labels.append({"timestamp":base+i*.1,"pose":{"x":i*.1,"y":0.,"yaw":.2},
                       "velocity":{"vx":1.,"vy":0.,"omega":0.}})
    path=bag/"training.json";path.write_text(json.dumps({"frames":frames,"labels":labels,"bag_meta":{"scene_desc":"COMMAND"}}))
    root=tmp_path/"dobot"/"roaming";root.mkdir(parents=True)
    (root/"index.jsonl").write_text("\n".join(json.dumps({"raw_format":"dobot",
        "training_json":str(path),"label_index":j,"current_frame_index":j,"task":"roaming"}) for j in (10,11,12))+"\n")
    return str(root)


def test_dataset_retarget_preservation_goal_coverage_and_resume(tmp_path):
    path=create_raw_manifest(tmp_path)
    kwargs=dict(json_path=path,num_input_frames=5,input_hz=2.,img_size=64,waypoint_dt=.6,
                trajectory_dt=.1,history=31,mode="train",seed=42)
    old=DobotOmTrackRGBDataset(**kwargs)
    original=next(iter(old))
    ds=DobotOmniRGBDataset(**kwargs,task_sampling={"ratios":{"roaming":1},"samples_per_epoch":3})
    new=next(iter(ds))
    raw=ds._active_index_record
    base=old._materialize_raw_index(raw)
    expected,mask=old._load_target(base,Path(path))
    assert torch.equal(new["future_trajectory"],expected)
    assert torch.equal(new["future_trajectory_mask"],mask)
    ds=DobotOmniRGBDataset(**kwargs,task_sampling={"ratios":{"point_nav":.5,"image_nav":.5},"samples_per_epoch":100})
    it=iter(ds);first=next(it);state=ds.state_dict()
    expected=next(it)
    recovered=DobotOmniRGBDataset(**kwargs,task_sampling={"ratios":{"point_nav":.5,"image_nav":.5},"samples_per_epoch":100})
    recovered.load_state_dict(state);actual=next(iter(recovered))
    assert actual["goal_metadata"] == expected["goal_metadata"]
    assert torch.equal(actual["future_trajectory"],expected["future_trajectory"])
    goals=[first,expected]+list(it)
    times=[s["goal_metadata"]["time_s"] for s in goals]
    assert min(times)<.7 and max(times)>2.4
    for s in goals:
        goal=s["goal_metadata"];after=torch.arange(1,9).double()*.6 >= goal["time_s"]-1e-7
        assert torch.allclose(s["future_trajectory"][after],
            torch.tensor(goal["pose"],dtype=torch.float32).expand(int(after.sum()),3))
        if s["task_name"]=="image_nav":
            assert s["goal_rgb"].shape==(64,64,3)


def _distributed_loss_worker(rank, init_file, output_dir):
    dist.init_process_group("gloo",init_method="file://"+init_file,rank=rank,world_size=2)
    try:
        tasks=([0,0] if rank==0 else [0,1])
        values=([1.,2.] if rank==0 else [3.,4.])
        batch,pred=loss_batch(tasks,values)
        parameter=torch.tensor(1.,requires_grad=True)
        pred={"pred_future_trajectory":pred["pred_future_trajectory"].detach()*parameter}
        loss=TaskAwareTrajectoryRegressionLoss(loss_fn="mse",global_loss_scale=1)
        result=loss(batch,pred);result["optimization_loss"].backward()
        grad=parameter.grad.clone();dist.all_reduce(grad);grad/=2
        metrics = {'total_loss': 0.}
        for key, statistics in result['_loss_statistics'].items():
            dist.all_reduce(statistics)
            value = float(statistics[0] / statistics[1].clamp_min(1e-12))
            metrics[key] = value
            metrics[key + '/count'] = int(statistics[2])
            metrics['total_loss'] += value
        Path(output_dir,f"rank{rank}.json").write_text(json.dumps({"gradient":grad.item(),**metrics}))
    finally:
        dist.destroy_process_group()


def test_distributed_task_means_and_missing_local_task(tmp_path):
    mp.spawn(_distributed_loss_worker,args=(str(tmp_path/"gloo_init"),str(tmp_path)),nprocs=2,join=True)
    result=json.loads((tmp_path/"rank0.json").read_text())
    # tracking mean=(1+4+9)/3, roaming mean=16; d/d(scale)=2*total
    expected=14/3+16
    assert result["total_loss"]==pytest.approx(expected)
    assert result["gradient"]==pytest.approx(2*expected)
    assert result["loss/tracking/trajectory/count"]==3
    assert result["loss/roaming/trajectory/count"]==1


def test_resume_rebuilds_current_rank_shard(tmp_path, monkeypatch):
    path=create_raw_manifest(tmp_path)
    kwargs=dict(json_path=path,num_input_frames=5,input_hz=2.,img_size=64,
        task_sampling={"ratios":{"point_nav":1.},"samples_per_epoch":16})
    monkeypatch.setenv("WORLD_SIZE","2")
    monkeypatch.setenv("RANK","0")
    rank0=DobotOmniRGBDataset(**kwargs)
    next(iter(rank0));state=rank0.state_dict()
    monkeypatch.setenv("RANK","1")
    rank1=DobotOmniRGBDataset(**kwargs)
    rank1.load_state_dict(state)
    resumed=next(iter(rank1))
    assert rank1._active_draw==3
    reference=DobotOmniRGBDataset(**kwargs)
    iterator=iter(reference);next(iterator)
    expected=next(iterator)
    assert resumed["goal_metadata"]==expected["goal_metadata"]
