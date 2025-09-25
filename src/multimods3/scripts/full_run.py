from collections.abc import Mapping
from datetime import datetime
from functools import partial
import logging
from pathlib import Path
from typing import Any, TypeVar, Protocol, runtime_checkable
import warnings

import torch
from torch.amp import GradScaler
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from multimods3.data.heico_dataset import HeicoDataset, HeicoDatasetArgs, MetadataLocationArgs, videos_from, \
    DataLoaderArgs, loader_from
from multimods3.models.attentive_pooler import AttentiveClassifier, AttentiveClassifierArgs, AttentivePoolerArgs
from multimods3.models.sensor_encoder import SensorsEncoder, SensorsArgs
from multimods3.models.vision_transformer import VisionTransformer, ViTSize, ViTSizeArgs, VideoArgs, ViTArgs
from multimods3.utils.distributed import world_size_and_rank, AllReduce
from multimods3.utils.logs import CSVLogger, MatrixLogger, AverageMeter
from multimods3.utils.misc import frozen, get_nested, prettified, to_path, nested_dict_from, save_with_backup
from multimods3.utils import info
from multimods3.utils.schedulers import WarmupCosineScheduleArgs, WarmupCosineSchedule, CosineWDSchedule, \
    CosineWDScheduleArgs
from multimods3.utils.video.classification import ClipAggregation
from multimods3.utils.video.transforms import EvalVideoTransform, VideoTransformArgs, VideoTransform
from multimods3.utils.weighted_sampler import DistributedWeightedSamplerArgs

logger = logging.getLogger()
C = dict[str, Any]  # A config dict

@runtime_checkable
class HasLoadStateDict(Protocol):
    def load_state_dict(self, state_dict, *args, **kwargs): ...
M = TypeVar("M", bound=HasLoadStateDict)  # A module subclass


def _initialized_device() -> torch.device:
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)  # TODO: Is this actually necessary in any situation here?
    return device


def _environment_state() -> dict[str, str]:
    return {
        "environment": info.environment(),
        "git-commit": info.git_commit(),
        "git-status": info.git_status(),
        "launch-command": info.cmd(),
        "machine": info.machine()}


def _transform(t: torch.Tensor, *, m_mean, r_std) -> torch.Tensor:
    return t.add(m_mean.to(t.device)).multiply_(r_std.to(t.device))


def _dataset_from(*, rank: int, video: C, sampling: C, location: C | str, architecture: C, sensors: C | None,
                  is_optimized_phase: bool) -> HeicoDataset:
    # Create videos
    load_sensors = sensors is not None
    if location != "debug":
        label = location["label"]
        location = MetadataLocationArgs(
            split_name=location["split_name"],
            split_file=to_path(location["split_file"]),
            metadata_file=to_path(location["metadata_file"]),
            base_dir=to_path(location["base_dir"]))
    videos = videos_from(location, load_sensors=load_sensors, label=label)

    # Create video transform
    rgb_mean, rgb_std = video["channel_stats"]["mean"], video["channel_stats"]["std"]
    normalize = tuple(v / 255. for v in rgb_mean), tuple(v / 255. for v in rgb_std)
    if not is_optimized_phase and video["num_views_per_segment"] > 1:
        logger.info("Make EvalVideoTransform (multi-view)")
        video_transform = EvalVideoTransform(
            num_views_per_clip=video["num_views_per_segment"],
            short_side_size=architecture["resolution"],
            normalize=normalize)
    else:
        logger.info("Make VideoTransform (single-view)")
        assert video["num_views_per_segment"] == 1
        video_transform = VideoTransform(VideoTransformArgs(
            crop_size=architecture["resolution"],
            normalize=normalize,
            train=is_optimized_phase))

    # Create sensor data transform
    if load_sensors:
        channels_m_mean = -torch.as_tensor(sensors["channel_stats"]["mean"])
        channels_r_std = 1 / torch.as_tensor(sensors["channel_stats"]["std"])
        sensor_transform = partial(_transform, m_mean=channels_m_mean, r_std=channels_r_std)
    else:
        sensor_transform = None

    args = HeicoDatasetArgs(
        num_clips=video["num_segments"],
        num_frames_per_clip=architecture["frames_per_clip"],
        frame_step=video["frame_step"],
        stride=video["stride"],
        relative_frame_indices=video["relative_frame_indices"],
        weight_by_class=sampling["weight_by_class"],
        skip_classes=sampling["skip_classes"])

    return HeicoDataset(rank, videos, video_transform, sensor_transform, args)


def _loaded(model: M, *, path_or_dict: Path | C, key: str) -> M:
    is_path = isinstance(path_or_dict, Path)
    # Following https://discuss.pytorch.org/t/preferred-way-of-loading-checkpoint-with-ddp-training/187470/3 (20250606)
    ddp_less = model.module if isinstance(model, DistributedDataParallel) else model
    class_name = ddp_less.__class__.__name__
    logger.info(f"Loading {class_name} from {path_or_dict if is_path else 'given dict'} (key={key})")
    checkpoint = torch.load(path_or_dict, map_location="cpu", weights_only=True) if is_path else path_or_dict
    params = get_nested(checkpoint, key)
    epoch = checkpoint.get("epoch", "n/a")
    del checkpoint
    params = {k.replace("module.", "").replace("backbone.", ""): v for k, v in params.items()}
    result = ddp_less.load_state_dict(params, **({"strict": True} if isinstance(ddp_less, torch.nn.Module) else {}))
    print(ddp_less)
    logger.info(f"Loaded {class_name} from epoch {epoch} with result: {'n/a' if result is None else result}")
    return model


def _video_encoder_from(*, args: C, frame_step: int, device: torch.device) -> ClipAggregation:
    architecture, load = args["architecture"], args["pretrain"]
    vit = VisionTransformer(
        size=ViTSize[architecture["model_name"].upper()],
        video_args=VideoArgs(
            img_size=architecture["resolution"],
            patch_size=architecture["patch_size"],
            num_frames=architecture["frames_per_clip"],
            frame_step=frame_step,
            tubelet_size=architecture["tubelet_size"],
            uniform_power=architecture["uniform_power"]))
    if load is not None:
        vit = _loaded(vit, path_or_dict=to_path(load["checkpoint_file"]), key=load["checkpoint_key"])
    assert architecture["frames_per_clip"] > 1  # Safely skip V-JEPA's FrameAggregation branch
    video_encoder = ClipAggregation(
        vit,
        tubelet_size=architecture["tubelet_size"],
        max_frames=-1,  # Not used because `use_pos_embed=False`
        use_pos_embed=False,
        attend_across_segments=True).to(device)
    return video_encoder


def _sensors_encoder_from(*, args: C, features: C, frames_per_clip: int, patch_size:int, embed_dim: int,
                          device: torch.device) -> DistributedDataParallel:
    a, load = args["architecture"], args["pretrain"]
    num_sensors = len(features["sensors"]["channel_stats"]["mean"])
    frame_step = features["video"]["common"]["frame_step"]
    size_args = ViTSizeArgs(depth=a["depth"], num_heads=a["num_heads"], mlp_ratio=a["mlp_ratio"], embed_dim=embed_dim)
    sensors_args = SensorsArgs.derive_from(num_sensors=num_sensors, num_frames=frames_per_clip, frame_step=frame_step)
    dropout_rate = a["dropout_rate"]
    encoder = SensorsEncoder(
        size_args=size_args,
        num_frames=frames_per_clip,
        frame_step=frame_step,
        video_patch_size=patch_size,
        sensors_args=sensors_args,
        vit_args=ViTArgs(drop_rate=dropout_rate, attn_drop_rate=dropout_rate),
    )
    if load is not None:
        encoder = _loaded(encoder, path_or_dict=to_path(load["checkpoint_file"]), key=load["checkpoint_key"])
    encoder = DistributedDataParallel(encoder.to(device), static_graph=True)
    return encoder


def _classifier_decoder_from(*, args: C, num_classes: int, device: torch.device) -> DistributedDataParallel:
    architecture, load = args["architecture"], args["pretrain"]
    size = ViTSize[architecture["model_name"].upper()].value
    decoder = AttentiveClassifier(AttentiveClassifierArgs(
        num_classes=num_classes,
        pooler_args=AttentivePoolerArgs(embed_dim=size.embed_dim, num_heads=size.num_heads)))
    if load is not None:
        decoder = _loaded(decoder, path_or_dict=to_path(load["checkpoint_file"]), key=load["checkpoint_key"])
    decoder = DistributedDataParallel(decoder.to(device), static_graph=True)
    return decoder


def _component_keys_by_state_from(*, args: C, opt_phase: str | None) -> C:  # {"frozen": ["encoders.video", …], …}
    component_keys_by_state = {"frozen": [], "fluid": [], "hot": []}
    for k, v in args.items():
        for kk, vv in v.items():
            component_keys_by_state[({} if vv is None else vv).get(opt_phase, {}).get("state", "frozen")] += [f"{k}.{kk}"]
    return component_keys_by_state


def _apply_states_to(*, model: C, component_keys_by_state: C):
    apply = {  # In-place or dummy ops
        "frozen": lambda m: None if m is None else frozen(m),
        "fluid": lambda m: None if m is None else m.eval(),
        "hot": lambda m: None
    }
    for state in component_keys_by_state.keys():
        component_keys = component_keys_by_state[state]
        logger.info(f"Apply '{state}' state to: " + (", ".join(component_keys) if component_keys else "-"))
        for component_key in component_keys:
            apply[state](get_nested(model, component_key))


def _optimizer_from(*, args: C, modules: Mapping[str, DistributedDataParallel]) -> Optimizer:
    mod_vals = modules.values()
    # Distinguish params with and without weight decay
    wd = lambda name, param: ("bias" not in name) and (len(param.shape) != 1)
    params = lambda with_wd: (p for m in mod_vals for n, p in m.module.named_parameters() if wd(n, p) == with_wd)
    wd_y = {"params": params(with_wd=True)}
    wd_n = {"params": params(with_wd=False), "WD_exclude": True, "weight_decay": 0}
    optimizer_name = args["optimizer"]
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW([wd_y, wd_n])
    else:
        raise RuntimeError(f"Unknown optimizer {optimizer_name}")
    logger.info(f"Created {optimizer_name} optimizer for: " + (", ".join(modules.keys()) if modules else "-"))
    return optimizer


def _lr_scheduler_from(args: C, optimizer: Optimizer, steps_total: int, iters_per_epoch: int) -> WarmupCosineSchedule:
    return WarmupCosineSchedule(optimizer, WarmupCosineScheduleArgs(
        steps_warmup=int(args["warmup"] * iters_per_epoch),
        steps_total=steps_total,
        start_lr=args["learning_rate"]["start"],
        ref_lr=args["learning_rate"]["reference"],
        final_lr=args["learning_rate"]["final"]))


def _wd_scheduler_from(args: C, optimizer: Optimizer, steps_total: int) -> CosineWDSchedule:
    return CosineWDSchedule(optimizer, CosineWDScheduleArgs(
        steps_total=steps_total,
        ref_wd=args["weight_decay"]["reference"],
        final_wd=args["weight_decay"]["final"]))


def _load_checkpoint_and_return_epoch(*, args: C, optimizer: Optimizer | None, scaler: GradScaler | None,
                                      modules: Mapping[str, DistributedDataParallel]) -> int:
    continue_from_latest, file = args["continue_from_latest"], to_path(args["file"])
    file_exists = file.is_file()
    if not file_exists:
        logger.info(f"Checkpoint {file} does not exist. Creating folder (if necessary)")
        file.parent.mkdir(parents=True, exist_ok=True)
    if continue_from_latest and file_exists:
        logger.info(f"Loading checkpoint from {file}")
        checkpoint = torch.load(file, map_location="cpu", weights_only=True)
        if optimizer is not None:
            _loaded(optimizer, path_or_dict=checkpoint, key="optimizer")
        if scaler is not None:
            _loaded(scaler, path_or_dict=checkpoint, key="scaler")
        for key, module in modules.items():
            _loaded(module, path_or_dict=checkpoint, key=key)
        epoch = checkpoint["epoch"]
        logger.info(f"Loaded checkpoint from {file} (epoch: {epoch}")
    else:
        reasons = ["file does not exist"] if not file_exists else []
        reasons += ["loading not indicated"] if not continue_from_latest else []
        logger.info(f"Skipped loading checkpoint from {file} ({', '.join(reasons)})")
        epoch = 0
    return epoch


def _save_checkpoint(*, args: C, do_save: bool, scalars: C, optimizer: Optimizer | None, scaler: GradScaler | None,
                     modules: Mapping[str, DistributedDataParallel]):
    if do_save:
        if modules:
            checkpoint_path = to_path(args["file"])
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            model_checkpoint = nested_dict_from({k: v.state_dict() for k, v in modules.items()})
            remaining_checkpoint = {}
            if optimizer is not None:
                remaining_checkpoint["optimizer"] = optimizer.state_dict()
            if scaler is not None:
                remaining_checkpoint["scaler"] = scaler.state_dict()
            remaining_checkpoint |= scalars
            checkpoint = model_checkpoint | remaining_checkpoint
            save_with_backup(checkpoint, checkpoint_path)
            saved_keys = ", ".join(list(modules.keys()) + list(remaining_checkpoint.keys()))  # Use flat keys for log
            logger.info(f"Saved checkpoint with {saved_keys} at {checkpoint_path}")
        else:
            logger.info(f"Skipped saving checkpoint (no optimized modules provided)")


def _barrier(msg: str | None):
    if msg is not None:
        logger.info(msg, stacklevel=2)  # Adjust `stacklevel` to not log this line
    torch.distributed.barrier()


def _run_one_epoch(*,
        device: torch.device,
        phase: str,
        is_optimized_phase: bool,
        model: C,
        optimized_modules_by_key: Mapping[str, DistributedDataParallel],
        optimizer: Optimizer | None,
        scaler: GradScaler | None,
        schedulers = tuple[WarmupCosineSchedule, CosineWDSchedule] | None,
        loader: DataLoader,
        show_every_iter: int,
        epoch: int,
        num_epochs: int,
        state_change_weight: float
) -> tuple[float, torch.Tensor]:

    loader.sampler.set_epoch(epoch)
    for key, module in optimized_modules_by_key.items():
        logger.info(f"Set train({is_optimized_phase}) on {key}")
        module.train(is_optimized_phase)

    criterion = torch.nn.CrossEntropyLoss()
    top1_meter = AverageMeter()
    confusion = None
    iters_per_epoch = len(loader)
    pad_len_iter = len(f"{iters_per_epoch:,}")
    pad_len_epoch = len(f"{num_epochs:,}")
    assert model["encoders"]["video"].attend_across_segments  # Safely skip outputs+loss for False case

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        torch.cuda.reset_max_memory_allocated()

    for itr, data in enumerate(loader):

        if is_optimized_phase and schedulers is not None:
            for scheduler in schedulers:
                scheduler.step()

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=scaler is not None):

            # Load data and put on GPU (iterate over spatial views of clip, temporal index of clip)
            clips = [[dij.to(device, non_blocking=True) for dij in di] for di in data[0]]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            labels = data[1].to(device)
            batch_size = len(labels)

            # Forward and prediction
            state_change = None
            with torch.no_grad():
                outputs = model["encoders"]["video"](clips, clip_indices)
                if not is_optimized_phase:
                    if (enc := model["encoders"]["sensors"]) is not None:
                        samples = [s_i.to(device, non_blocking=True) for s_i in data[4]["sensors_encoder"]]
                        outputs = [enc(o, samples) for o in outputs]  # Advance all views' states with same samples
                    outputs = [model["decoders"]["classifier"](o) for o in outputs]
            if is_optimized_phase:
                if "encoders.sensors" in optimized_modules_by_key:
                    enc = model["encoders"]["sensors"]
                    out_init = [o_i.clone() for o_i in outputs]
                    samples = [s_i.to(device, non_blocking=True) for s_i in data[4]["sensors_encoder"]]
                    outputs = [enc(o, samples) for o in outputs]  # Advance all views' states with same samples
                    state_change = sum([torch.mean((o_i - o) ** 2) for o_i, o in zip(out_init, outputs)]) / len(out_init)
                else:
                    with torch.no_grad():
                        if (enc := model["encoders"]["sensors"]) is not None:
                            samples = [s_i.to(device, non_blocking=True) for s_i in data[4]["sensors_encoder"]]
                            outputs = [enc(o, samples) for o in outputs]  # Advance all views' states with same samples
                outputs = [model["decoders"]["classifier"](o) for o in outputs]

        loss = sum([criterion(o, labels) for o in outputs]) / len(outputs)
        if state_change is not None:
            loss = (1 - state_change_weight) * loss + state_change_weight * state_change
        with torch.no_grad():
            outputs = sum([torch.nn.functional.softmax(o, dim=1) for o in outputs]) / len(outputs)
            top1_acc = 100. * outputs.max(dim=1).indices.eq(labels).sum() / batch_size
            top1_acc = float(AllReduce.apply(top1_acc))
            top1_meter.update(top1_acc)
            if not is_optimized_phase:
                current_confusion = torch.zeros((outputs.shape[1], outputs.shape[1]), dtype=int, device=outputs.device)
                labels_predicted = outputs.max(dim=1).indices
                one = torch.tensor(1, dtype=int, device=outputs.device)
                current_confusion.index_put_((labels_predicted, labels), one, accumulate=True)
                AllReduce.apply(current_confusion)
                if confusion is None:
                    confusion = current_confusion
                else:
                    confusion += current_confusion

        if is_optimized_phase:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                for module in optimized_modules_by_key.values():
                    torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                for module in optimized_modules_by_key.values():
                    torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        if itr % show_every_iter == 0:
            logger.info('%s [ep %s/%s it %s/%s] acc (⌀%.2f ·%.2f)%% (loss %.3f) [GPU ↑%.3gGiB]'
                             % (phase,
                                format(epoch + 1, ",").rjust(pad_len_epoch), format(num_epochs, ","),
                                format(itr + 1, ",").rjust(pad_len_iter), format(iters_per_epoch, ","),
                                top1_meter.avg, top1_meter.val, loss,
                                torch.cuda.max_memory_allocated() / 1024. ** 3))

    return top1_meter.avg, (None if confusion is None else confusion.cpu().detach())


def main(config: C):
    now = datetime.now()
    checkpoint_state =  {"config": prettified(config), "launch-time": f"{now:%Y-%m-%d %H:%M:%S}"}
    checkpoint_state |= _environment_state()
    device = _initialized_device()
    world_size, rank = world_size_and_rank()
    logger.info(f"On {rank=} ({world_size=})")

    # Unwrap config
    phases = config["phases"]
    data_config = config["data"]
    sampling_config = data_config["sampling"]
    workflow_config = config["workflow"]
    checkpointing_config = workflow_config["checkpointing"]
    phases_config = workflow_config["phases"]
    model_config = config["model"]
    encoders_config = model_config["encoders"]
    decoders_config = model_config["decoders"]

    optimized_phase: str | None = workflow_config["optimize_in_phase"]
    optimized_phase = None if optimized_phase not in phases else optimized_phase  # Defined but not used → None
    frame_step = data_config["features"]["video"]["common"]["frame_step"]

    video_encoder = _video_encoder_from(args=encoders_config["video"], frame_step=frame_step, device=device)
    sensors_encoder = None if (sensors_args := encoders_config["sensors"]) is None else _sensors_encoder_from(
            args=sensors_args,
            features=data_config["features"],
            frames_per_clip=encoders_config["video"]["architecture"]["frames_per_clip"],
            patch_size=encoders_config["video"]["architecture"]["patch_size"],
            embed_dim=video_encoder.embed_dim,
            device=device)
    classifier_decoder = _classifier_decoder_from(
        args=decoders_config["classifier"],
        num_classes=data_config["targets"]["classifier"]["num_classes"],
        device=device)
    if (compile_mode := workflow_config["compile_mode"]) is not None:
        logger.info(f"Compile model components with mode={compile_mode}")
        video_encoder = torch.compile(video_encoder, mode=compile_mode)
        sensors_encoder = None if sensors_encoder is None else torch.compile(sensors_encoder, mode=compile_mode)
        classifier_decoder = torch.compile(classifier_decoder, mode=compile_mode)
    model = {
        "encoders": {"video": video_encoder, "sensors": sensors_encoder},
        "decoders": {"classifier": classifier_decoder}}
    component_keys_by_state = _component_keys_by_state_from(args=model_config, opt_phase=optimized_phase)
    _apply_states_to(model=model, component_keys_by_state=component_keys_by_state)
    optimized_modules_by_key = {k: get_nested(model, k) for k in component_keys_by_state["hot"]}
    if optimized_phase is None:
        optimizer = None
    else:
        optimizer = _optimizer_from(args=phases_config[optimized_phase], modules=optimized_modules_by_key)
    scaler = None if not workflow_config["use_bfloat16"] else GradScaler("cuda")
    latest_epoch = _load_checkpoint_and_return_epoch(
        args=checkpointing_config,
        optimizer=optimizer,
        scaler=scaler,
        modules=optimized_modules_by_key)
    num_epochs = workflow_config["num_epochs"]

    dataset_by_phase = {phase: _dataset_from(
        rank=rank,
        video=data_config["features"]["video"][phase],
        sampling=sampling_config[phase],
        location=data_config["location"][phase],
        architecture=encoders_config["video"]["architecture"],
        sensors=data_config["features"]["sensors"],
        is_optimized_phase=(phase == optimized_phase)) for phase in phases}

    loader_args_by_phase = {phase: DataLoaderArgs(
        batch_size=phases_config[phase]["batch_size"],
        num_workers=phases_config[phase]["num_workers"],
        prefetch_factor=phases_config[phase]["prefetch_factor"]) for phase in phases}

    sampler_args_by_phase = {phase: DistributedWeightedSamplerArgs(
        number_type="relative" if (num := sampling_config[phase]["number"]) is None else num["type"],
        number_value=1.0 if num is None else num["value"],
        num_replicas=world_size,
        rank=rank,
        shuffle=(phase == optimized_phase)) for phase in phases}

    # Init logging
    do_log = (rank == 0)
    log_dir = to_path(workflow_config["logging"]["dir"])
    logger.info(f"Creating logging folder {log_dir} (if necessary)")
    log_prefix = workflow_config["logging"]["prefix"]
    csv_path = log_dir / f"{log_prefix}.{now:%Y-%m-%d-%H-%M-%S}.csv"
    conf_path = log_dir / f"{log_prefix}.{now:%Y-%m-%d-%H-%M-%S}.confusion"
    csv_logger = CSVLogger(csv_path, do_log=do_log, time=None, epoch="d",  **{f"acc-{p}": ".5f" for p in phases})
    conf_logger = MatrixLogger(conf_path, do_log=do_log, fmt=",")

    for epoch in range(latest_epoch, num_epochs):

        _barrier(f"Epoch {epoch + 1}: reached 1st barrier in epoch loop.")

        loader_by_phase = {phase: loader_from(
            dataset_by_phase[phase],
            loader_args_by_phase[phase],
            sampler_args_by_phase [phase]) for phase in phases}
        len_by_phase = ", ".join(f"{len(loader_by_phase[p]):,} iters/epoch ({p})" for p in phases)
        logger.info(f"{len(loader_by_phase)} loader(s) created: " + len_by_phase)
        if optimized_phase is not None:
            opt_config = phases_config[optimized_phase]
            schedule_scale = opt_config["schedule_scale"]
            iters_per_epoch = len(loader_by_phase[optimized_phase])
            steps_current = int(epoch * iters_per_epoch)
            steps_total = int(schedule_scale * num_epochs * iters_per_epoch)
            logger.info("Creating and advancing schedulers")
            lr_scheduler = _lr_scheduler_from(opt_config, optimizer, steps_total, iters_per_epoch).step(steps_current)
            wd_scheduler = _wd_scheduler_from(opt_config, optimizer, steps_total).step(steps_current)
            schedulers = (lr_scheduler, wd_scheduler)
        else:
            schedulers = None

        _barrier(f"Epoch {epoch + 1}: reached 2nd barrier in epoch loop")

        run_epoch = partial(
            _run_one_epoch,
            device=device,
            model=model,
            optimized_modules_by_key=optimized_modules_by_key,
            optimizer=optimizer,
            scaler=scaler,
            schedulers=schedulers,
            show_every_iter=workflow_config["logging"]["show_every_iter"],
            epoch=epoch,
            num_epochs=num_epochs,
            state_change_weight=workflow_config["phases"].get(optimized_phase, {}).get("state_change_weight", None)
        )
        acc_by_phase, conf_by_phase = {} ,{}
        for phase in phases:
            a, c = run_epoch(phase=phase, is_optimized_phase=(phase == optimized_phase), loader=loader_by_phase[phase])
            acc_by_phase[phase] = a
            conf_by_phase[phase] = c
        logger.info(f"Epoch {epoch + 1}: " + ", ".join(f"{acc:.3f}% ({phase})" for phase, acc in acc_by_phase.items()))
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        csv_logger.log(time=f"{now_str}", epoch=epoch + 1, **{f"acc-{p}": a for p, a in acc_by_phase.items()})
        for phase, conf in conf_by_phase.items():
            if phase != optimized_phase:
                conf_logger.log(conf, prepend_line=f"{now_str} (epoch {epoch + 1}): ↓ prediction, → target")

        more_scalars = {
            "launch-time": f"{now:%Y-%m-%d %H:%M:%S}",
            "save-time": now_str,
            "config": prettified(config),
            "epoch": epoch + 1,
            "world-size": world_size,
        }
        _save_checkpoint(args=checkpointing_config, do_save=do_log,
                         scalars = _environment_state() | more_scalars,
                         optimizer=optimizer, scaler=scaler, modules=optimized_modules_by_key)
