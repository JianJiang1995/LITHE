"""Stock diffusers loader on a tiny random FluxTransformer2DModel (CPU, no weights).

Skipped when diffusers / peft are not importable.
"""
import copy

import pytest
import torch

import lithe

diffusers = pytest.importorskip("diffusers", exc_type=ImportError)
pytest.importorskip("peft", exc_type=ImportError)
from diffusers import FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel  # noqa: E402
from diffusers.loaders import FluxLoraLoaderMixin  # noqa: E402


def tiny_flux():
    torch.manual_seed(0)
    return FluxTransformer2DModel(patch_size=1, in_channels=16, num_layers=2, num_single_layers=2,
                                  attention_head_dim=16, num_attention_heads=2, joint_attention_dim=32,
                                  pooled_projection_dim=32, guidance_embeds=True, axes_dims_rope=(4, 4, 8))


def inputs(bsz=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return dict(hidden_states=torch.randn(bsz, 16, 16, generator=g),
                encoder_hidden_states=torch.randn(bsz, 5, 32, generator=g),
                pooled_projections=torch.randn(bsz, 32, generator=g),
                timestep=torch.full((bsz,), 0.5), guidance=torch.full((bsz,), 3.5),
                img_ids=torch.zeros(16, 3), txt_ids=torch.zeros(5, 3), return_dict=False)


class _Loader(FluxLoraLoaderMixin):
    """Just enough of a pipeline for FluxLoraLoaderMixin.load_lora_weights."""

    def __init__(self, transformer):
        self.transformer = transformer
        self.text_encoder = None
        self.hf_device_map = None
        self.components = {"transformer": transformer}


def test_tiny_flux_target_count():
    model = tiny_flux()
    layers = lithe.inject_lithe(model, lithe.FLUX_TARGET_MODULES, rank=4)
    assert len(layers) == 2 * 12 + 2 * 3


def test_decoded_lora_loads_with_stock_loader(tmp_path):
    base = tiny_flux()
    ref = copy.deepcopy(base)
    layers = lithe.inject_lithe(base, lithe.FLUX_TARGET_MODULES, rank=4, alpha=8)
    with torch.no_grad():
        for layer in layers.values():
            layer.lora_A.normal_(0, 0.06)
            layer.lora_B.normal_(0, 0.02)
            layer.requantize()
    base.eval()
    with torch.no_grad():
        y_lithe = base(**inputs())[0]
        y_base = ref(**inputs())[0]
    ad = lithe.adapter_from_layers(layers, rank=4, alpha=8)
    path = tmp_path / "a.lithe"
    lithe.save_lithe(path, ad)
    from safetensors.torch import save_file
    lora_path = tmp_path / "lora.safetensors"
    save_file(lithe.to_lora_state_dict(lithe.load_lithe(path), dtype=torch.float32), str(lora_path))
    _Loader(ref).load_lora_weights(str(lora_path), adapter_name="lithe")
    with torch.no_grad():
        y_peft = ref(**inputs())[0]
    assert (y_lithe - y_base).abs().max() > 1e-3          # the adapter does something
    assert torch.allclose(y_peft, y_lithe, atol=1e-5), (y_peft - y_lithe).abs().max()


def test_flow_matching_training_steps_reduce_nothing_breaks():
    import train_lithe_flux as T

    model = tiny_flux()
    model.requires_grad_(False)
    layers = lithe.inject_lithe(model, lithe.FLUX_TARGET_MODULES, rank=4)
    params = [p for layer in layers.values() for p in (layer.lora_A, layer.lora_B)]
    opt = torch.optim.AdamW(params, lr=1e-3)
    sched = FlowMatchEulerDiscreteScheduler()
    latents = torch.randn(1, 4, 8, 8)
    prompt = torch.randn(1, 5, 32)
    pooled = torch.randn(1, 32)
    model.train()
    for _ in range(3):
        loss = T.flow_matching_loss(model, latents, prompt, pooled, torch.zeros(5, 3), sched)
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert torch.isfinite(loss)
    assert all(layer.lora_B.abs().sum() > 0 for layer in layers.values())


def test_multi_adapter_batch_on_tiny_flux():
    from lithe import serve

    base = tiny_flux()
    ads = []
    for seed in range(2):
        m = copy.deepcopy(base)
        torch.manual_seed(seed)
        layers = lithe.inject_lithe(m, lithe.FLUX_TARGET_MODULES, rank=4)
        with torch.no_grad():
            for layer in layers.values():
                layer.lora_A.normal_(0, 0.06)
                layer.lora_B.normal_(0, 0.02)
        ads.append(lithe.adapter_from_layers(layers, rank=4, alpha=4))
    singles = []
    for ad in ads:
        m = copy.deepcopy(base)
        router, _ = serve.attach_adapters(m, [ad])
        router.set([[(0, 1.0)]])
        with torch.no_grad():
            singles.append(m(**inputs(bsz=1))[0])
    multi = copy.deepcopy(base)
    router, _ = serve.attach_adapters(multi, ads)
    router.set([[(0, 1.0)], [(1, 1.0)]])
    x = inputs(bsz=1)
    x["hidden_states"] = x["hidden_states"].repeat(2, 1, 1)
    x["encoder_hidden_states"] = x["encoder_hidden_states"].repeat(2, 1, 1)
    x["pooled_projections"] = x["pooled_projections"].repeat(2, 1)
    x["timestep"], x["guidance"] = torch.full((2,), 0.5), torch.full((2,), 3.5)
    with torch.no_grad():
        y = multi(**x)[0]
    assert torch.allclose(y[0], singles[0][0], atol=1e-5)
    assert torch.allclose(y[1], singles[1][0], atol=1e-5)
