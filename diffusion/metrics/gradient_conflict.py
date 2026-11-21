import torch
from typing import Callable, Dict, Optional


def _grad_norm_of_scalar_loss(
    loss: torch.Tensor,
    params: list[torch.nn.Parameter],
    retain_graph: bool = True,
) -> torch.Tensor:
    """Compute L2 grad norm of a scalar loss w.r.t. params (allowing unused grads)."""
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        allow_unused=True,
        create_graph=False,
    )

    total_sq = None
    for g in grads:
        if g is None:
            continue
        g_sq = g.detach().float().pow(2).sum()
        total_sq = g_sq if total_sq is None else (total_sq + g_sq)

    if total_sq is None:
        return torch.tensor(0.0, device=loss.device)
    return torch.sqrt(total_sq + 1e-12)


def _pick_motion_loss(loss_dict: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """Pick representative motion reconstruction loss key across loss types."""
    for key in ("motion_loss", "mse_motion", "mae_motion"):
        if key in loss_dict:
            return loss_dict[key]
    return None


def _pick_simple_and_rcxyz_losses(
    loss_dict: Dict[str, torch.Tensor],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Build two grouped losses:
    - simple_loss: motion + translation (MSE/L1 variants unified)
    - geo_loss: rcxyz branch only
    """
    simple_terms = []
    for key in ("motion_loss", "mse_motion", "mae_motion", "translation_loss"):
        if key in loss_dict:
            simple_terms.append(loss_dict[key])

    rcxyz_loss = None
    for key in ("motion_rcxyz", "mse_motion_rcxyz"):
        if key in loss_dict:
            rcxyz_loss = loss_dict[key]
            break

    simple_loss = None
    if len(simple_terms) > 0:
        simple_loss = sum(simple_terms)

    return simple_loss, rcxyz_loss


def _pick_simple_and_structural_losses(
    loss_dict: Dict[str, torch.Tensor],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Build two grouped losses:
    - simple_loss: motion + translation (MSE/L1 variants unified)
    - geo_loss: structural_loss branch only
    """
    simple_terms = []
    for key in ("motion_loss", "mse_motion", "mae_motion", "translation_loss"):
        if key in loss_dict:
            simple_terms.append(loss_dict[key])

    structural_loss = loss_dict.get("structural_loss", None)

    simple_loss = None
    if len(simple_terms) > 0:
        simple_loss = sum(simple_terms)

    return simple_loss, structural_loss


def _pair_conflict_stats(
    simple_loss: torch.Tensor,
    geo_loss: torch.Tensor,
    params: list[torch.nn.Parameter],
) -> Optional[Dict[str, torch.Tensor]]:
    """Return one-pair conflict stats from grouped simple vs geo losses."""
    grads_simple = torch.autograd.grad(
        simple_loss,
        params,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )
    grads_geo = torch.autograd.grad(
        geo_loss,
        params,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )

    dot = None
    simple_sq = None
    geo_sq = None
    for gs, gg in zip(grads_simple, grads_geo):
        if gs is None or gg is None:
            continue
        gs = gs.detach().float()
        gg = gg.detach().float()
        cur_dot = (gs * gg).sum()
        cur_s = gs.pow(2).sum()
        cur_g = gg.pow(2).sum()
        dot = cur_dot if dot is None else (dot + cur_dot)
        simple_sq = cur_s if simple_sq is None else (simple_sq + cur_s)
        geo_sq = cur_g if geo_sq is None else (geo_sq + cur_g)

    if dot is None or simple_sq is None or geo_sq is None:
        return None

    simple_norm = torch.sqrt(simple_sq + 1e-12)
    geo_norm = torch.sqrt(geo_sq + 1e-12)
    cos = dot / ((simple_norm + 1e-12) * (geo_norm + 1e-12))
    neg_mask = (cos < 0).float()

    csi_dot = torch.clamp(-dot, min=0.0)
    csi_proj = neg_mask * (-dot) / (geo_norm + 1e-12)
    gdr = geo_norm / (simple_norm + 1e-12)

    return {
        "cos": cos,
        "ncr": neg_mask,
        "csi_dot": csi_dot,
        "csi_proj": csi_proj,
        "gdr": gdr,
    }


def _pair_conflict_stats_from_grads(
    grads_a: tuple[Optional[torch.Tensor], ...],
    grads_b: tuple[Optional[torch.Tensor], ...],
    device: torch.device,
) -> Optional[Dict[str, torch.Tensor]]:
    """Return one-pair conflict stats from two precomputed grad tuples."""
    dot = None
    a_sq = None
    b_sq = None
    for ga, gb in zip(grads_a, grads_b):
        if ga is None or gb is None:
            continue
        ga = ga.detach().float()
        gb = gb.detach().float()
        cur_dot = (ga * gb).sum()
        cur_a = ga.pow(2).sum()
        cur_b = gb.pow(2).sum()
        dot = cur_dot if dot is None else (dot + cur_dot)
        a_sq = cur_a if a_sq is None else (a_sq + cur_a)
        b_sq = cur_b if b_sq is None else (b_sq + cur_b)

    if dot is None or a_sq is None or b_sq is None:
        return None

    a_norm = torch.sqrt(a_sq + 1e-12)
    b_norm = torch.sqrt(b_sq + 1e-12)
    cos = dot / ((a_norm + 1e-12) * (b_norm + 1e-12))
    neg_mask = (cos < 0).float()

    csi_dot = torch.clamp(-dot, min=0.0)
    csi_proj = neg_mask * (-dot) / (b_norm + 1e-12)
    gdr = b_norm / (a_norm + 1e-12)

    return {
        "cos": cos,
        "ncr": neg_mask,
        "csi_dot": csi_dot,
        "csi_proj": csi_proj,
        "gdr": gdr,
    }


def compute_block_motion_grad_debug_metrics(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    timestep: torch.Tensor,
    fk_func,
    block_size: int,
    compute_block_losses_fn: Callable,
    params: list[torch.nn.Parameter],
    extra_logvar: Optional[torch.Tensor] = None,
    frame_supervision_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute lightweight per-block diagnostics:
    1) max/min loss ratio across blocks (for each available loss term)
    2) max/min gradnorm ratio across blocks (motion representative)
    3) per-block motion gradnorm fraction
    """
    if block_size <= 0:
        return {}

    _, seq_len, _ = pred.shape
    if seq_len % block_size != 0:
        return {}

    num_blocks = seq_len // block_size
    if num_blocks < 2 or len(params) == 0:
        return {}

    motion_grad_norms = []
    block_loss_values: Dict[str, list[float]] = {}
    for b in range(num_blocks):
        s = b * block_size
        e = s + block_size

        block_losses = compute_block_losses_fn(
            pred,
            target,
            timestep,
            fk_func,
            s,
            e,
            extra_logvar=extra_logvar,
            frame_supervision_weight=frame_supervision_weight,
        )

        # keep only aggregated ratio-ready values (do not log per-block raw loss values)
        for name, val in block_losses.items():
            block_loss_values.setdefault(name, []).append(float(val.detach().item()))

        motion_loss = _pick_motion_loss(block_losses)
        if motion_loss is None:
            motion_grad_norms.append(torch.tensor(0.0, device=pred.device))
            continue

        motion_grad_norms.append(
            _grad_norm_of_scalar_loss(
                motion_loss,
                params,
                retain_graph=True,
            )
        )

    norms = torch.stack(motion_grad_norms)
    metrics: Dict[str, torch.Tensor] = {}

    # per-loss imbalance ratio across blocks (max/min), no raw per-block loss logging
    for name, vals in block_loss_values.items():
        if len(vals) >= 2:
            metrics[f"debug_block/max_min_{name}_ratio"] = torch.tensor(
                max(vals) / (min(vals) + 1e-8),
                device=pred.device,
            )

    metrics["debug_block_grad/max_min_motion_loss_gradnorm_ratio"] = (
        norms.max() / (norms.min() + 1e-8)
    )

    total = norms.sum() + 1e-8
    for b in range(num_blocks):
        metrics[f"debug_block_grad/motion_loss_gradnorm_frac_b{b}"] = norms[b] / total

    return metrics


def compute_interblock_motion_conflict_metrics(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    timestep: torch.Tensor,
    fk_func,
    block_size: int,
    compute_block_losses_fn: Callable,
    params: list[torch.nn.Parameter],
    extra_logvar: Optional[torch.Tensor] = None,
    frame_supervision_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Measure gradient conflict across different blocks.

    Procedure:
    1) compute representative motion loss for each block
    2) get per-block gradients w.r.t. shared params
    3) compute pairwise conflict stats (cos, NCR, CSI, GDR)
    """
    metrics: Dict[str, torch.Tensor] = {}
    if block_size <= 0 or len(params) == 0:
        return metrics

    _, seq_len, _ = pred.shape
    if seq_len % block_size != 0:
        return metrics

    num_blocks = seq_len // block_size
    if num_blocks < 2:
        return metrics

    block_grads: list[Optional[tuple[Optional[torch.Tensor], ...]]] = []
    for b in range(num_blocks):
        s = b * block_size
        e = s + block_size
        block_losses = compute_block_losses_fn(
            pred,
            target,
            timestep,
            fk_func,
            s,
            e,
            extra_logvar=extra_logvar,
            frame_supervision_weight=frame_supervision_weight,
        )
        motion_loss = _pick_motion_loss(block_losses)
        if motion_loss is None:
            block_grads.append(None)
            continue

        grads = torch.autograd.grad(
            motion_loss,
            params,
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        block_grads.append(grads)

    pair_stats = []
    pair_count = 0
    for i in range(num_blocks):
        gi = block_grads[i]
        if gi is None:
            continue
        for j in range(i + 1, num_blocks):
            gj = block_grads[j]
            if gj is None:
                continue
            st = _pair_conflict_stats_from_grads(gi, gj, pred.device)
            if st is None:
                continue

            pair_count += 1
            pair_stats.append(st)
            metrics[f"debug_block_conflict/pair_b{i}_b{j}/cos_motion"] = st["cos"]
            metrics[f"debug_block_conflict/pair_b{i}_b{j}/ncr_motion"] = st["ncr"]

    metrics["debug_block_conflict/pair_count"] = torch.tensor(
        float(pair_count), device=pred.device
    )

    if len(pair_stats) > 0:
        metrics["debug_block_conflict/pair_mean/cos_motion"] = torch.stack(
            [x["cos"] for x in pair_stats]
        ).mean()
        metrics["debug_block_conflict/pair_mean/ncr_motion"] = torch.stack(
            [x["ncr"] for x in pair_stats]
        ).mean()
        metrics["debug_block_conflict/pair_mean/csi_dot_motion"] = torch.stack(
            [x["csi_dot"] for x in pair_stats]
        ).mean()
        metrics["debug_block_conflict/pair_mean/csi_proj_motion"] = torch.stack(
            [x["csi_proj"] for x in pair_stats]
        ).mean()
        metrics["debug_block_conflict/pair_mean/gdr_motion"] = torch.stack(
            [x["gdr"] for x in pair_stats]
        ).mean()
        metrics["debug_block_conflict/pair_min/cos_motion"] = torch.stack(
            [x["cos"] for x in pair_stats]
        ).min()

    return metrics


def compute_simple_vs_rcxyz_conflict_metrics(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    timestep: torch.Tensor,
    fk_func,
    block_size: int,
    compute_block_losses_fn: Callable,
    params: list[torch.nn.Parameter],
    extra_logvar: Optional[torch.Tensor] = None,
    frame_supervision_weight: Optional[torch.Tensor] = None,
    boundary_window: int = 4,
    train_sampling_steps: Optional[int] = None,
    num_timestep_buckets: int = 0,
) -> Dict[str, torch.Tensor]:
    return _compute_simple_vs_geo_conflict_metrics(
        pred=pred,
        target=target,
        timestep=timestep,
        fk_func=fk_func,
        block_size=block_size,
        compute_block_losses_fn=compute_block_losses_fn,
        params=params,
        pick_pair_losses_fn=_pick_simple_and_rcxyz_losses,
        pair_suffix="simple_vs_rcxyz",
        metric_prefix="debug_conflict",
        extra_logvar=extra_logvar,
        frame_supervision_weight=frame_supervision_weight,
        boundary_window=boundary_window,
        train_sampling_steps=train_sampling_steps,
        num_timestep_buckets=num_timestep_buckets,
    )


def compute_simple_vs_structural_conflict_metrics(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    timestep: torch.Tensor,
    fk_func,
    block_size: int,
    compute_block_losses_fn: Callable,
    params: list[torch.nn.Parameter],
    extra_logvar: Optional[torch.Tensor] = None,
    frame_supervision_weight: Optional[torch.Tensor] = None,
    boundary_window: int = 4,
    train_sampling_steps: Optional[int] = None,
    num_timestep_buckets: int = 0,
) -> Dict[str, torch.Tensor]:
    return _compute_simple_vs_geo_conflict_metrics(
        pred=pred,
        target=target,
        timestep=timestep,
        fk_func=fk_func,
        block_size=block_size,
        compute_block_losses_fn=compute_block_losses_fn,
        params=params,
        pick_pair_losses_fn=_pick_simple_and_structural_losses,
        pair_suffix="simple_vs_structural",
        metric_prefix="debug_conflict_struct",
        extra_logvar=extra_logvar,
        frame_supervision_weight=frame_supervision_weight,
        boundary_window=boundary_window,
        train_sampling_steps=train_sampling_steps,
        num_timestep_buckets=num_timestep_buckets,
    )


def _compute_simple_vs_geo_conflict_metrics(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    timestep: torch.Tensor,
    fk_func,
    block_size: int,
    compute_block_losses_fn: Callable,
    params: list[torch.nn.Parameter],
    pick_pair_losses_fn: Callable[[Dict[str, torch.Tensor]], tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
    pair_suffix: str,
    metric_prefix: str,
    extra_logvar: Optional[torch.Tensor] = None,
    frame_supervision_weight: Optional[torch.Tensor] = None,
    boundary_window: int = 4,
    train_sampling_steps: Optional[int] = None,
    num_timestep_buckets: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Conflict metrics between grouped simple losses (motion+translation)
    and a geometry target guidance (rcxyz / structural_loss).

    Logged metrics:
    - global: NCR / CSI_dot / CSI_proj / GDR / cosine
    - block_mean: same metrics averaged over per-block pairs
    - boundary_mean (BCI): same metrics averaged over boundary windows
    - bucket: same metrics averaged over timestep buckets (block-level)
    """
    metrics: Dict[str, torch.Tensor] = {}
    if len(params) == 0:
        return metrics

    _, seq_len, _ = pred.shape
    if seq_len < 2:
        return metrics

    # 1) global pair stats
    global_losses = compute_block_losses_fn(
        pred,
        target,
        timestep,
        fk_func,
        0,
        seq_len,
        extra_logvar=extra_logvar,
        frame_supervision_weight=frame_supervision_weight,
    )
    simple_loss, geo_loss = pick_pair_losses_fn(global_losses)
    if simple_loss is None or geo_loss is None:
        return metrics

    global_stats = _pair_conflict_stats(simple_loss, geo_loss, params)
    if global_stats is None:
        return metrics

    metrics[f"{metric_prefix}/global/cos_{pair_suffix}"] = global_stats["cos"]
    metrics[f"{metric_prefix}/global/ncr_{pair_suffix}"] = global_stats["ncr"]
    metrics[f"{metric_prefix}/global/csi_dot_{pair_suffix}"] = global_stats["csi_dot"]
    metrics[f"{metric_prefix}/global/csi_proj_{pair_suffix}"] = global_stats["csi_proj"]
    metrics[f"{metric_prefix}/global/gdr_{pair_suffix}"] = global_stats["gdr"]

    # 2) per-block mean stats
    if block_size > 0 and seq_len % block_size == 0 and (seq_len // block_size) >= 2:
        num_blocks = seq_len // block_size
        block_stats = []
        block_bucket_ids = []

        use_bucket = (
            train_sampling_steps is not None
            and int(train_sampling_steps) > 0
            and int(num_timestep_buckets) > 0
        )
        if use_bucket:
            num_timestep_buckets = int(num_timestep_buckets)
            bucket_width = float(train_sampling_steps) / float(num_timestep_buckets)

        for b in range(num_blocks):
            s = b * block_size
            e = s + block_size
            block_losses = compute_block_losses_fn(
                pred,
                target,
                timestep,
                fk_func,
                s,
                e,
                extra_logvar=extra_logvar,
                frame_supervision_weight=frame_supervision_weight,
            )
            b_simple, b_geo = pick_pair_losses_fn(block_losses)
            if b_simple is None or b_geo is None:
                continue
            st = _pair_conflict_stats(b_simple, b_geo, params)
            if st is not None:
                block_stats.append(st)
                if use_bucket:
                    block_t = float(timestep[:, s:e].detach().float().mean().item())
                    if bucket_width <= 0:
                        bid = 0
                    else:
                        bid = int(block_t / bucket_width)
                    bid = max(0, min(num_timestep_buckets - 1, bid))
                    block_bucket_ids.append(bid)

        if len(block_stats) > 0:
            keys = ("ncr", "csi_dot", "csi_proj", "gdr", "cos")
            for k in keys:
                metrics[f"{metric_prefix}/block_mean/{k}_{pair_suffix}"] = torch.stack(
                    [x[k] for x in block_stats]
                ).mean()

            # 2.1) timestep bucket stats (block-level)
            if use_bucket and len(block_bucket_ids) == len(block_stats):
                for bid in range(num_timestep_buckets):
                    idxs = [i for i, v in enumerate(block_bucket_ids) if v == bid]
                    count = len(idxs)
                    metrics[f"{metric_prefix}/tbucket{bid}/count"] = torch.tensor(
                        float(count), device=pred.device
                    )
                    if count == 0:
                        continue

                    keys = ("ncr", "csi_dot", "csi_proj", "gdr", "cos")
                    for k in keys:
                        metrics[f"{metric_prefix}/tbucket{bid}/{k}_{pair_suffix}"] = torch.stack(
                            [block_stats[i][k] for i in idxs]
                        ).mean()

    # 3) boundary-window mean stats (BCI)
    if (
        boundary_window > 0
        and block_size > 0
        and seq_len % block_size == 0
        and (seq_len // block_size) >= 2
    ):
        num_blocks = seq_len // block_size
        boundary_stats = []
        for b in range(1, num_blocks):
            center = b * block_size
            s = max(0, center - boundary_window)
            e = min(seq_len, center + boundary_window)
            if e - s <= 1:
                continue
            reg_losses = compute_block_losses_fn(
                pred,
                target,
                timestep,
                fk_func,
                s,
                e,
                extra_logvar=extra_logvar,
                frame_supervision_weight=frame_supervision_weight,
            )
            r_simple, r_geo = pick_pair_losses_fn(reg_losses)
            if r_simple is None or r_geo is None:
                continue
            st = _pair_conflict_stats(r_simple, r_geo, params)
            if st is not None:
                boundary_stats.append(st)

        if len(boundary_stats) > 0:
            keys = ("ncr", "csi_dot", "csi_proj", "gdr", "cos")
            for k in keys:
                metrics[f"{metric_prefix}/boundary_mean/{k}_{pair_suffix}"] = torch.stack(
                    [x[k] for x in boundary_stats]
                ).mean()

    return metrics
