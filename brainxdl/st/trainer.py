"""ST-BrainXDL: training loop, validation, and checkpoint management.

Contains:
- init_checkpoint: initialise model / optimisers / schedulers, resume if possible.
- train:           main epoch loop with early stopping.
- validate:        evaluation pass (no gradients).

Sources: extracted from legacy_code/1-ST/train.py
  (loadCheckpoint → init_checkpoint, train, validate).

Changes from legacy:
- Imports use package paths (brainxdl.st.model, brainxdl.common.*).
- save_obj / load_obj / load_rng_state replaced with common equivalents.
- myDataset / padseq / load_data replaced with brainxdl.st.dataset.
- The two optimiser passes have explicit, disjoint gradient boundaries.
"""

import copy
import math
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from brainxdl.common.checkpointing import load_rng_state, save_checkpoint
from brainxdl.st.model import STBrainXDL


def compose_core_loss(
    x_nll,
    z_kld,
    adjacency_nll,
    classification_loss,
    *,
    kl_weight: float,
    mce_weight: float,
    encoder_cls_weight: float = 0.0,
):
    """Compose the encoder loss while preserving legacy behavior at weight zero."""
    return (
        x_nll
        + kl_weight * z_kld
        + mce_weight * adjacency_nll
        + encoder_cls_weight * classification_loss
    )


def _optimizer_parameters(optimizer):
    """Return an optimiser's parameters without relying on model-name matching."""
    return [parameter for group in optimizer.param_groups for parameter in group["params"]]


def _grad_norm(parameters) -> float:
    """Return the finite L2 norm of currently accumulated gradients."""
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().pow(2).sum().item())
    return math.sqrt(squared)


def _gradients_equal(parameters, snapshots) -> bool:
    """Check that a later backward pass did not alter an existing gradient set."""
    for parameter, before in zip(parameters, snapshots):
        after = parameter.grad
        if before is None:
            if after is not None:
                return False
        elif after is None or not torch.equal(after, before):
            return False
    return True


def _optimize_batch_two_stage(
    model,
    batch,
    setting: dict,
    optimizers,
    *,
    current_kl_weight: float,
    bce_weight: float,
    mce_weight: float,
    encoder_cls_weight: float,
    gradient_clip: bool,
    grad_threshold: float,
    audit_gradients: bool = False,
):
    """Apply disjoint representation and classifier gradients for one batch.

    Both optimisers use the same pre-update representation.  The classifier is
    frozen and evaluated without dropout/BatchNorm-stat updates while it supplies
    supervised guidance to the representation.  Its own update then uses a
    detached representation in training mode.  Optimiser steps occur only after
    both gradient sets have been formed, so neither pass reuses stale activations
    after a parameter update.
    """
    if len(optimizers) != 2:
        raise ValueError("ST-BrainXDL training requires core and classifier optimizers")
    multipliers = setting.get("yBCEMultiplier", [1, 1])
    if len(multipliers) != 2:
        raise ValueError("yBCEMultiplier must contain core and classifier values")

    core_optimizer, classifier_optimizer = optimizers
    core_parameters = _optimizer_parameters(core_optimizer)
    classifier_parameters = _optimizer_parameters(classifier_optimizer)
    core_optimizer.zero_grad(set_to_none=True)
    classifier_optimizer.zero_grad(set_to_none=True)

    x_nll, z_kld, adjacency_nll, readout, target = model(
        batch, setting, sample=False
    )

    # The head acts as a fixed differentiable map for supervised representation
    # guidance.  Evaluation mode prevents this auxiliary pass from updating its
    # BatchNorm statistics or applying a second independent dropout mask.
    classifier_was_training = model.classifier.training
    classifier_requires_grad = [parameter.requires_grad for parameter in classifier_parameters]
    model.classifier.eval()
    for parameter in classifier_parameters:
        parameter.requires_grad_(False)
    try:
        representation_prediction_loss, _ = model.classifier(
            readout, target, sample=False
        )
    finally:
        for parameter, requires_grad in zip(
            classifier_parameters, classifier_requires_grad
        ):
            parameter.requires_grad_(requires_grad)
        model.classifier.train(classifier_was_training)

    core_loss = compose_core_loss(
        x_nll,
        z_kld,
        adjacency_nll,
        representation_prediction_loss,
        kl_weight=current_kl_weight,
        mce_weight=mce_weight,
        encoder_cls_weight=(0.0 if setting["DAT"] else encoder_cls_weight),
    )
    if setting["DAT"]:
        core_loss = (
            core_loss
            - float(multipliers[0]) * bce_weight * representation_prediction_loss
        )
    core_loss.backward()

    audit = None
    if audit_gradients:
        audit = {
            "core_grad_norm_after_representation": _grad_norm(core_parameters),
            "classifier_grad_norm_after_representation": _grad_norm(
                classifier_parameters
            ),
        }

    if gradient_clip:
        nn.utils.clip_grad_norm_(
            core_parameters,
            grad_threshold,
            norm_type=2,
            error_if_nonfinite=False,
        )
    if audit_gradients:
        core_gradient_snapshot = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in core_parameters
        ]

    # The detached readout makes this a classifier-only objective.  No core
    # parameter can receive an additional gradient from this backward pass.
    classification_loss, classification_accuracy = model.classifier(
        readout.detach(), target, sample=False
    )
    classifier_loss = float(multipliers[1]) * bce_weight * classification_loss
    classifier_loss.backward()

    if audit_gradients:
        audit.update(
            {
                "core_gradients_unchanged_by_classifier": _gradients_equal(
                    core_parameters, core_gradient_snapshot
                ),
                "classifier_grad_norm_after_classifier": _grad_norm(
                    classifier_parameters
                ),
            }
        )
        if audit["classifier_grad_norm_after_representation"] != 0.0:
            raise AssertionError(
                "Classifier accumulated gradients during the representation-only pass"
            )
        if not audit["core_gradients_unchanged_by_classifier"]:
            raise AssertionError(
                "Core gradients changed during the detached classifier-only pass"
            )
        if audit["classifier_grad_norm_after_classifier"] == 0.0:
            raise AssertionError("Classifier-only loss produced no classifier gradient")

    if gradient_clip:
        nn.utils.clip_grad_norm_(
            classifier_parameters,
            grad_threshold,
            norm_type=2,
            error_if_nonfinite=False,
        )

    # The two disjoint updates are committed together from the same pre-update
    # forward state; each optimiser is stepped exactly once per batch.
    core_optimizer.step()
    classifier_optimizer.step()
    return {
        "x_NLL": x_nll,
        "z_KLD": z_kld,
        "a_NLL": adjacency_nll,
        "y_BCE": classification_loss,
        "y_ACC": classification_accuracy,
        "core_loss": core_loss,
        "classifier_loss": classifier_loss,
        "gradient_audit": audit,
    }


def init_checkpoint(setting: dict, load_path: str, save_path: str):
    """Initialise ST-BrainXDL model, optimisers, and schedulers; resume if checkpoint exists.

    Args:
        setting:   Training config dict (see scripts/st/st_train.py for keys).
        load_path: Path to load an existing checkpoint from.
        save_path: Path to write checkpoints to during training.

    Returns:
        (model, optimizers, schedulers, epoch_start,
         train_losses, val_losses, test_losses)
    """
    epoch_start = 0
    _empty_losses = lambda: {
        "x_NLL": [], "z_KLD": [], "a_NLL": [], "y_BCE": [], "y_ACC": [], "Total": []
    }
    train_losses = _empty_losses()
    val_losses   = _empty_losses()
    test_losses  = _empty_losses()

    # Optionally restore RNG state for exact reproducibility
    if setting.get("rngPATH"):
        load_rng_state(setting["rngPATH"])

    model = STBrainXDL(setting)
    model.to(model.device)

    # Two optimisers: one for the ST-BrainXDL core, one for the classifier.
    # This allows different LR / weight-decay for each component.
    st_brainxdl_params = {
        k: v for k, v in model.named_parameters() if "classifier" not in k
    }
    optimizers = [
        optim.AdamW(
            st_brainxdl_params.values(),
            lr=setting["learnRate"][0],
            weight_decay=setting["l2factor"][0],
        ),
        optim.AdamW(
            model.classifier.parameters(),
            lr=setting["learnRate"][1],
            weight_decay=setting["l2factor"][1],
        ),
    ]

    def _make_scheduler(opt, anneal_type, patience, factor):
        if anneal_type == "StepLR":
            return optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.96)
        if anneal_type == "ReduceLROnPlateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", patience=patience, factor=factor
            )
        if anneal_type is None:
            return None
        raise ValueError(f"Unsupported lr_annealType: '{anneal_type}'")

    schedulers = [
        _make_scheduler(
            optimizers[i],
            setting["lr_annealType"][i],
            setting["lr_annealPatience"][i],
            setting["lr_annealFactor"][i],
        )
        for i in range(2)
    ]

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    if os.path.exists(load_path):
        print(f"[init_checkpoint] Loading from: {load_path}")
        ckpt = torch.load(load_path, map_location=model.device, weights_only=False)
        incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                "[init_checkpoint] WARNING: mismatched keys (expected if architecture changed).\n"
                f"  missing:    {incompatible.missing_keys}\n"
                f"  unexpected: {incompatible.unexpected_keys}"
            )
        if load_path == save_path:
            print("[init_checkpoint] Resuming training state.")
            epoch_start  = ckpt["epoch"] + 1
            train_losses = ckpt["train_losses"]
            val_losses   = ckpt["val_losses"]
            test_losses  = ckpt["test_losses"]
            for i in range(2):
                optimizers[i].load_state_dict(ckpt["optimizer_state_dicts"][i])
                if schedulers[i] is not None and ckpt["scheduler_state_dicts"][i] is not None:
                    schedulers[i].load_state_dict(ckpt["scheduler_state_dicts"][i])
        else:
            print("[init_checkpoint] Checkpoint loaded for fine-tune / inference.")
    else:
        print("[init_checkpoint] No checkpoint found — training from scratch.")
        save_checkpoint(
            save_path,
            {
                "model_state_dict":      copy.deepcopy(model.state_dict()),
                "optimizer_state_dicts": [copy.deepcopy(o.state_dict()) for o in optimizers],
                "scheduler_state_dicts": [
                    copy.deepcopy(s.state_dict()) if s is not None else None
                    for s in schedulers
                ],
                "train_losses":   train_losses,
                "val_losses":     val_losses,
                "test_losses":    test_losses,
                "epoch":          0,
                "training_setting": setting,
                "rng_state":      torch.get_rng_state(),
            },
        )

    return model, optimizers, schedulers, epoch_start, train_losses, val_losses, test_losses


def train(
    model,
    optimizers,
    schedulers,
    setting: dict,
    checkpoint_path: str,
    train_losses: dict,
    val_losses: dict,
    test_losses: dict,
    train_loader,
    val_loader,
    test_loader,
    epoch_start: int = 0,
    num_epochs: int = 100,
    grad_threshold: float = 1.0,
    gradient_clip: bool = True,
    verbose_freq: int = 1,
    verbose: bool = True,
    val_freq: int = 1,
    validation: bool = False,
    testing: bool = False,
    early_stop_patience: int = 30,
    early_stop: bool = True,
    device=None,
):
    """Main ST-BrainXDL training loop.

    Dual-optimiser training: optimiser[0] updates the ST-BrainXDL core (with
    optional supervised or domain-adversarial guidance), while optimiser[1]
    updates the classifier from a detached representation.

    Args:
        model:               Initialised ST-BrainXDL instance.
        optimizers:          List of two AdamW optimisers.
        schedulers:          List of two LR schedulers (or None).
        setting:             Training config dict.
        checkpoint_path:     Path to save epoch checkpoints.
        train_losses/val/test: Dicts with keys x_NLL, z_KLD, a_NLL, y_BCE, y_ACC, Total.
        train_loader/val/test: DataLoaders using padseq collate.
        epoch_start:         Resume epoch index.
        num_epochs:          Total epochs to run.
        grad_threshold:      Gradient clipping max norm.
        gradient_clip:       Enable gradient clipping.
        verbose_freq:        Print every N iterations.
        verbose:             Enable verbose printing.
        val_freq:            Run validation/test every N epochs.
        validation:          Enable validation pass.
        testing:             Enable test pass.
        early_stop_patience: Patience before early stopping.
        early_stop:          Enable early stopping.
        device:              Override device (defaults to setting['device']).

    Returns:
        (model, train_losses, val_losses, test_losses)
    """
    if device is None:
        device = torch.device(setting.get("device", str(getattr(model, "device", "cpu"))))
    model.to(device)
    model.device = device
    sparse_on_cpu = setting.get("sparse_on_cpu", True)
    model.apply_device_split(device, sparse_on_cpu)
    kl_weight = float(setting.get("kl_weight", 1.0))
    kl_start_weight = setting.get("kl_start_weight")
    kl_start_weight = kl_weight if kl_start_weight is None else float(kl_start_weight)
    kl_warmup_epochs = int(setting.get("kl_warmup_epochs", 0))
    bce_weight = float(setting.get("bce_weight", 1.0))
    mce_weight = float(setting.get("mce_weight", 1.0))
    encoder_cls_weight = float(setting.get("encoder_cls_weight", 0.0))
    print(
        f"[Train] device={device}  sparse_on_cpu={sparse_on_cpu}  "
        f"prefer_mps={setting.get('prefer_mps', True)}  kl_weight={kl_weight:g}  "
        f"kl_start_weight={kl_start_weight:g}  kl_warmup_epochs={kl_warmup_epochs}  "
        f"bce_weight={bce_weight:g}  mce_weight={mce_weight:g}  "
        f"encoder_cls_weight={encoder_cls_weight:g}"
    )

    valLoss = testLoss = _zero_loss_dict(model.device)
    best_params      = copy.deepcopy(model.state_dict())
    best_val_score   = math.inf
    best_test_score  = float("nan")
    best_at_epoch    = epoch_start
    patience_count   = 0
    lr_anneal_metric = 0.0
    num_bad_epochs   = [None, None]
    selection_metric = setting.get("selection_metric", "legacy_train_bce_acc")
    patience_metric  = float("inf")
    test_metric      = float("nan")
    start_time       = time.time()

    for epoch in range(epoch_start, num_epochs):
        current_kl_weight = _kl_weight_for_epoch(
            epoch,
            kl_start_weight,
            kl_weight,
            kl_warmup_epochs,
        )
        trainLoss     = _zero_loss_dict(model.device)
        num_iter      = len(train_loader)

        for idx, batch in enumerate(train_loader):
            model.train()
            batch_result = _optimize_batch_two_stage(
                model,
                batch,
                setting,
                optimizers,
                current_kl_weight=current_kl_weight,
                bce_weight=bce_weight,
                mce_weight=mce_weight,
                encoder_cls_weight=encoder_cls_weight,
                gradient_clip=gradient_clip,
                grad_threshold=grad_threshold,
                audit_gradients=bool(setting.get("assert_gradient_flow", False)),
            )
            x_NLL = batch_result["x_NLL"]
            z_KLD = batch_result["z_KLD"]
            a_NLL = batch_result["a_NLL"]
            y_BCE = batch_result["y_BCE"]
            y_ACC = batch_result["y_ACC"]

            with torch.no_grad():
                total_loss = (
                    x_NLL
                    + current_kl_weight * z_KLD
                    + mce_weight * a_NLL
                    + bce_weight * y_BCE
                )
                trainLoss["x_NLL"] += x_NLL   / num_iter
                trainLoss["z_KLD"] += z_KLD   / num_iter
                trainLoss["a_NLL"] += a_NLL   / num_iter
                trainLoss["y_BCE"] += y_BCE   / num_iter
                trainLoss["y_ACC"] += y_ACC   / num_iter
                trainLoss["Total"] += total_loss / num_iter

            if verbose and (idx + 1 == 1 or (idx + 1) % verbose_freq == 0):
                elapsed = time.time() - start_time
                print(
                    f"Elapsed: {elapsed:.0f}s  Epoch: {epoch+1}/{num_epochs}  "
                    f"Iter: {idx+1}/{num_iter}  SeqLen: {len(batch)}  "
                    f"BS: {batch[0].num_graphs}\n"
                    f"  VB={setting['variational']}  DAT={setting['DAT']}  "
                    f"GraphRNN={setting['graphRNN']}  Recurrent={setting['recurrent']}\n"
                    f"  LR: {optimizers[0].param_groups[0]['lr']:.2e},"
                    f" {optimizers[1].param_groups[0]['lr']:.2e}  "
                    f"KL_weight: {current_kl_weight:.4g}  "
                    f"MCE_weight: {mce_weight:.4g}  BCE_weight: {bce_weight:.4g}  "
                    f"Encoder_cls_weight: {encoder_cls_weight:.4g}  "
                    f"BCE_mult: {setting['yBCEMultiplier'][0]:.0e},"
                    f" {setting['yBCEMultiplier'][1]:.0e}\n"
                    f"  Patience: {patience_count}/{early_stop_patience}  "
                    f"Selection={selection_metric}  PatMetric: {patience_metric:.4f}  "
                    f"BestTestAtBestVal: {best_test_score:.4f}  "
                    f"BestEpoch: {best_at_epoch}\n"
                    f"  Train -- "
                    f"x_NLL={x_NLL:.4f}  z_KLD={z_KLD:.4f}  "
                    f"a_NLL={a_NLL:.4f}  y_BCE={y_BCE:.4f}  y_ACC={y_ACC:.4f}  "
                    f"Total={total_loss:.4f}\n"
                    f"  Val   -- y_BCE={valLoss['y_BCE'].item():.4f}  "
                    f"y_ACC={valLoss['y_ACC'].item():.4f}\n"
                    f"  Test  -- y_BCE={testLoss['y_BCE'].item():.4f}  "
                    f"y_ACC={testLoss['y_ACC'].item():.4f}\n---"
                )

        # Append epoch metrics
        for key in train_losses:
            train_losses[key].append(trainLoss[key].cpu())

        if (epoch + 1) % val_freq == 0:
            if validation:
                valLoss = validate(model, setting, val_loader, kl_weight=current_kl_weight)
                for key in val_losses:
                    val_losses[key].append(valLoss[key].cpu())
            if testing:
                if test_loader is None:
                    raise ValueError("testing=True requires a test_loader")
                testLoss = validate(model, setting, test_loader, kl_weight=current_kl_weight)
                for key in test_losses:
                    test_losses[key].append(testLoss[key].cpu())
                test_metric = testLoss["y_ACC"].item()

            if early_stop:
                patience_metric = _selection_score(selection_metric, trainLoss, valLoss)
                if patience_metric <= best_val_score:
                    best_val_score  = patience_metric
                    best_test_score = test_metric
                    best_params     = copy.deepcopy(model.state_dict())
                    best_at_epoch   = epoch + 1
                    patience_count  = 0
                else:
                    patience_count += 1
            else:
                best_params = copy.deepcopy(model.state_dict())

        # LR annealing
        lr_anneal_metric = trainLoss["y_BCE"].item()
        for i, scheduler in enumerate(schedulers):
            if isinstance(scheduler, optim.lr_scheduler.StepLR):
                scheduler.step()
            elif isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(lr_anneal_metric)
                num_bad_epochs[i] = scheduler.state_dict()["num_bad_epochs"]

        save_checkpoint(
            checkpoint_path,
            {
                "model_state_dict":      best_params,
                "optimizer_state_dicts": [copy.deepcopy(o.state_dict()) for o in optimizers],
                "scheduler_state_dicts": [
                    copy.deepcopy(s.state_dict()) if s is not None else None
                    for s in schedulers
                ],
                "train_losses":    train_losses,
                "val_losses":      val_losses,
                "test_losses":     test_losses,
                "epoch":           epoch,
                "training_setting": setting,
                "rng_state":       torch.get_rng_state(),
            },
        )

        if early_stop and patience_count > early_stop_patience:
            print("Early stopped.")
            break

    print("Training done!")
    return model, train_losses, val_losses, test_losses


def _selection_score(selection_metric: str, train_loss: dict, val_loss: dict) -> float:
    """Return a scalar score to minimize for checkpoint selection."""
    if selection_metric == "legacy_train_bce_acc":
        return (train_loss["y_BCE"] + train_loss["y_ACC"]).item()
    if selection_metric == "val_bce":
        return val_loss["y_BCE"].item()
    if selection_metric == "val_total":
        return val_loss["Total"].item()
    if selection_metric == "val_acc":
        return -val_loss["y_ACC"].item()
    raise ValueError(f"Unsupported selection_metric: {selection_metric}")


def validate(model, setting: dict, loader, kl_weight: float | None = None):
    """Evaluate ST-BrainXDL on *loader* without gradient computation.

    Args:
        model:   ST-BrainXDL instance (will be set to eval mode).
        setting: Training config dict.
        loader:  DataLoader (uses padseq collate).

    Returns:
        dict with keys x_NLL, z_KLD, a_NLL, y_BCE, y_ACC, Total.
    """
    num_iter = len(loader)
    loss_dict = _zero_loss_dict(model.device)
    if kl_weight is None:
        kl_weight = float(setting.get("kl_weight", 1.0))
    bce_weight = float(setting.get("bce_weight", 1.0))
    mce_weight = float(setting.get("mce_weight", 1.0))

    for batch in loader:
        model.eval()
        with torch.no_grad():
            x_NLL, z_KLD, a_NLL, Readout, Target = model(batch, setting, sample=False)
            y_BCE, y_ACC = model.classifier(Readout, Target, sample=False)
            total = x_NLL + kl_weight * z_KLD + mce_weight * a_NLL + bce_weight * y_BCE

        with torch.no_grad():
            loss_dict["x_NLL"] += x_NLL  / num_iter
            loss_dict["z_KLD"] += z_KLD  / num_iter
            loss_dict["a_NLL"] += a_NLL  / num_iter
            loss_dict["y_BCE"] += y_BCE  / num_iter
            loss_dict["y_ACC"] += y_ACC  / num_iter
            loss_dict["Total"] += total  / num_iter

    return loss_dict


def _kl_weight_for_epoch(
    epoch: int,
    start_weight: float,
    target_weight: float,
    warmup_epochs: int,
) -> float:
    """Linearly warm up KL weight, preserving fixed-KL behavior by default."""
    if warmup_epochs <= 0:
        return target_weight
    progress = min(max((epoch + 1) / warmup_epochs, 0.0), 1.0)
    return start_weight + progress * (target_weight - start_weight)


def _zero_loss_dict(device):
    """Return a fresh all-zero loss accumulation dict on *device*."""
    keys = ["x_NLL", "z_KLD", "a_NLL", "y_BCE", "y_ACC", "Total"]
    return {k: torch.zeros(1).to(device) for k in keys}
