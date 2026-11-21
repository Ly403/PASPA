"""
提供的快照函数，用以debug
"""
import hashlib, random, os, json
from collections import OrderedDict
import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def _tensor_digest(t: torch.Tensor) -> str:
    t = t.detach().to("cpu")
    if t.is_floating_point():
        t = t.float()
    return _sha256_bytes(t.contiguous().numpy().tobytes())

def model_state_digest(model) -> str:
    m = model.module if hasattr(model, "module") else model
    h = hashlib.sha256()
    with torch.no_grad():
        sd = m.state_dict()
        for k in sorted(sd.keys()):
            v = sd[k]
            if torch.is_tensor(v):
                x = v.detach().to("cpu")
                if x.is_floating_point():
                    x = x.float()
                h.update(x.contiguous().numpy().tobytes())
            else:
                h.update(str(v).encode())
    return h.hexdigest()[:16]

def batchnorm_digest(model) -> dict:
    m = model.module if hasattr(model, "module") else model
    out = {}
    for name, mod in m.named_modules():
        if isinstance(mod, _BatchNorm):
            out[name] = dict(
                training=bool(mod.training),
                track_running_stats=bool(getattr(mod, "track_running_stats", False)),
                running_mean=_tensor_digest(mod.running_mean) if getattr(mod, "running_mean", None) is not None else None,
                running_var=_tensor_digest(mod.running_var) if getattr(mod, "running_var", None) is not None else None,
                num_batches_tracked=int(mod.num_batches_tracked) if hasattr(mod, "num_batches_tracked") else None,
            )
    return out

def rng_digest() -> dict:
    d = {}
    d["torch"] = _sha256_bytes(torch.get_rng_state().cpu().numpy().tobytes())
    if torch.cuda.is_available():
        all_cuda = b"".join([x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all()])
        d["cuda"] = _sha256_bytes(all_cuda)
    d["py"] = _sha256_bytes(str(random.getstate()).encode())
    d["np"] = _sha256_bytes(str(np.random.get_state()[1][:10]).encode())  # 部分摘要即可
    return d

def snapshot(tag: str, model, logger=None):
    msg = (
        f"[{tag}] training={getattr(model, 'training', None)} "
        f"model_digest={model_state_digest(model)} rng={rng_digest()} "
        f"bn_modules={len(batchnorm_digest(model))}"
    )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)

def _tensor_bytes_dtype_preserving(t: torch.Tensor) -> bytes:
    x = t.detach()
    if x.device.type != "cpu":
        x = x.cpu()
    x = x.contiguous()
    return x.view(torch.uint8).numpy().tobytes()

def _tensor_bytes(t: torch.Tensor) -> bytes:
    """
    Return raw byte representation of a tensor (dtype-preserving).
    Works for bf16/fp16/etc by viewing as uint8.
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(t)}")

    x = t.detach()
    if x.device.type != "cpu":
        x = x.cpu()
    x = x.contiguous()

    # view as uint8 to get raw bytes across all dtypes (including bf16)
    x_u8 = x.view(torch.uint8)
    return x_u8.numpy().tobytes()

def compute_state_dict_snapshot(
    state_dict: "OrderedDict[str, torch.Tensor]",
    algo: str = "sha1",
    per_tensor: bool = True,
) -> dict:
    """
    Compute a deterministic snapshot for a state_dict.

    Returns:
      {
        "algo": "sha1",
        "overall": "<hex>",
        "num_tensors": N,
        "tensors": { "key": {"shape":[...],"dtype":"...","hash":"...","numel":...}, ... }  # optional
      }
    """
    if algo not in ("sha1", "sha256", "blake2b"):
        raise ValueError(f"Unsupported algo={algo}")

    h = getattr(hashlib, algo)()
    tensors_info = {} if per_tensor else None

    # Ensure deterministic order even if caller passed a normal dict
    keys = list(state_dict.keys())
    keys.sort()

    for k in keys:
        v = state_dict[k]
        if not isinstance(v, torch.Tensor):
            # Some state_dict entries can be non-tensor in exotic setups; skip or encode repr
            payload = repr(v).encode("utf-8")
            h.update(k.encode("utf-8") + b"\0" + payload)
            continue

        # incorporate metadata to make collisions harder and debugging easier
        meta = f"{k}|{str(v.dtype)}|{list(v.shape)}|{v.numel()}".encode("utf-8")
        bts = _tensor_bytes(v)

        if per_tensor:
            ht = getattr(hashlib, algo)()
            ht.update(meta)
            ht.update(b"\0")
            ht.update(bts)
            tensors_info[k] = {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "numel": int(v.numel()),
                "hash": ht.hexdigest(),
            }

        h.update(meta)
        h.update(b"\0")
        h.update(bts)

    out = {
        "algo": algo,
        "overall": h.hexdigest(),
        "num_tensors": len(keys),
    }
    if per_tensor:
        out["tensors"] = tensors_info
    return out


@torch.no_grad()
def compute_model_snapshot(
    model: torch.nn.Module,
    accelerator=None,
    use_accelerator_get_state_dict: bool = False,
    include_buffers: bool = True,
    algo: str = "sha1",
    per_tensor: bool = True,
) -> dict:
    """
    model snapshot wrapper:
    - DDP: pass accelerator, uses unwrap_model(model).state_dict()
    - FSDP: set use_accelerator_get_state_dict=True to use accelerator.get_state_dict(model)
    """
    if accelerator is not None:
        base_model = accelerator.unwrap_model(model)
    else:
        base_model = model

    # IMPORTANT:
    # - state_dict() includes buffers by default.
    # - If you truly want params-only, you can filter by base_model.named_parameters() instead.
    if accelerator is not None and use_accelerator_get_state_dict:
        # Recommended path for FSDP FULL_STATE_DICT in Accelerate docs
        sd = accelerator.get_state_dict(model)  # :contentReference[oaicite:2]{index=2}
    else:
        sd = base_model.state_dict()

    if not include_buffers:
        # keep only parameter keys (fast path: build a set)
        param_keys = set(dict(base_model.named_parameters()).keys())
        sd = OrderedDict((k, v) for k, v in sd.items() if k in param_keys)

    return compute_state_dict_snapshot(sd, algo=algo, per_tensor=per_tensor)

def iter_all_buffers_with_persistence(model: torch.nn.Module):
    """
    Yields: (full_name, tensor, persistent: bool)
    Includes non-persistent buffers too.
    """
    m = model.module if hasattr(model, "module") else model

    for module_prefix, mod in m.named_modules():
        # mod._buffers: Dict[str, Optional[Tensor]]
        for buf_name, buf in getattr(mod, "_buffers", {}).items():
            if buf is None:
                continue
            full = f"{module_prefix}.{buf_name}" if module_prefix else buf_name
            non_persistent = buf_name in getattr(mod, "_non_persistent_buffers_set", set())
            persistent = not non_persistent
            yield full, buf, persistent


@torch.no_grad()
def compute_all_buffers_snapshot(model: torch.nn.Module, algo: str = "sha1", per_tensor: bool = True) -> dict:
    if algo not in ("sha1", "sha256", "blake2b"):
        raise ValueError(f"Unsupported algo={algo}")

    h = getattr(hashlib, algo)()
    infos = {} if per_tensor else None

    items = list(iter_all_buffers_with_persistence(model))
    items.sort(key=lambda x: x[0])  # deterministic

    for name, buf, persistent in items:
        meta = f"{name}|{str(buf.dtype)}|{list(buf.shape)}|{int(buf.numel())}|persistent={int(persistent)}".encode("utf-8")
        bts = _tensor_bytes_dtype_preserving(buf)

        if per_tensor:
            ht = getattr(hashlib, algo)()
            ht.update(meta); ht.update(b"\0"); ht.update(bts)
            infos[name] = {
                "shape": list(buf.shape),
                "dtype": str(buf.dtype),
                "numel": int(buf.numel()),
                "persistent": bool(persistent),
                "hash": ht.hexdigest(),
            }

        h.update(meta); h.update(b"\0"); h.update(bts)

    out = {
        "algo": algo,
        "overall": h.hexdigest(),
        "num_buffers": len(items),
    }
    if per_tensor:
        out["buffers"] = infos
    return out

@torch.no_grad()
def compute_full_debug_snapshot(model, accelerator=None, use_fsdp_state=False) -> dict:
    # 1) state_dict snapshot (params + persistent buffers)
    sd_snap = compute_model_snapshot(
        model,
        accelerator=accelerator,
        use_accelerator_get_state_dict=use_fsdp_state,
        include_buffers=True,
        algo="sha1",
        per_tensor=False,
    )

    # 2) all buffers snapshot (persistent + non-persistent)
    base = accelerator.unwrap_model(model) if accelerator is not None else model
    buf_snap = compute_all_buffers_snapshot(base, algo="sha1", per_tensor=False)

    return {
        "state_dict": sd_snap,
        "all_buffers": buf_snap,
    }

def dump_snapshot_json(snapshot: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    
def sampler_fingerprint(dataloader, k=64):
    s = dataloader.sampler
    it = iter(s)
    idx = [next(it) for _ in range(k)]
    h = hashlib.sha1(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:16]
    return h, idx[:10]