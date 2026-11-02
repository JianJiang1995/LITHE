import copy

import torch
from torch import nn

import lithe
from lithe import quant


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(96, 160)
        self.attn.to_out = nn.ModuleList([nn.Linear(160, 96)])
        self.other = nn.Linear(96, 96)

    def forward(self, x):
        return self.other(self.attn.to_out[0](self.attn.to_q(x)))


def _trained_net(seed=0, return_base=False):
    torch.manual_seed(seed)
    net = Net()
    base_net = copy.deepcopy(net)
    layers = lithe.inject_lithe(net, ["attn.to_q", "attn.to_out.0"], rank=8, alpha=16)
    with torch.no_grad():
        for layer in layers.values():
            layer.lora_A.normal_(0, 0.06)
            layer.lora_B.normal_(0, 0.003)
            layer.requantize()
    return (net, layers, base_net) if return_base else (net, layers)


def test_injection_targets():
    net, layers = _trained_net()
    assert list(layers) == ["attn.to_q", "attn.to_out.0"]
    assert isinstance(net.other, nn.Linear)
    assert all(not p.requires_grad for layer in layers.values() for p in layer.base.parameters())


def test_flux_target_count():
    # 19 double-stream blocks x 12 + 38 single-stream blocks x 3 (to_q/k/v)
    names = []
    for i in range(19):
        names += [f"transformer_blocks.{i}.{t}" for t in lithe.FLUX_TARGET_MODULES]
    for i in range(38):
        names += [f"single_transformer_blocks.{i}.attn.{t}" for t in ("to_q", "to_k", "to_v")]
        names += [f"single_transformer_blocks.{i}.{t}" for t in ("proj_mlp", "proj_out", "norm.linear")]
    from lithe.module import _match
    assert sum(_match(n, lithe.FLUX_TARGET_MODULES) for n in names) == 342


def test_forward_matches_dense_lora_from_decoded_factors():
    net, layers = _trained_net()
    net.eval()
    x = torch.randn(4, 7, 96)
    ad = lithe.adapter_from_layers(layers, rank=8, alpha=16)
    factors = lithe.decode_factors(ad)
    with torch.no_grad():
        y = net(x)
        h = net.attn.to_q.base(x)
        a, b = factors["attn.to_q"]
        h = h + (x @ a.T @ b.T) * 2.0
        a2, b2 = factors["attn.to_out.0"]
        h2 = net.attn.to_out[0].base(h) + (h @ a2.T @ b2.T) * 2.0
        ref = net.other(h2)
    assert torch.allclose(y, ref, atol=1e-6)
    # decoded factors equal what the module uses
    for name, layer in layers.items():
        a_hat, b_hat = layer.decoded_factors()
        assert torch.equal(a_hat, factors[name][0]) and torch.equal(b_hat, factors[name][1])


def test_lora_state_dict_folds_alpha():
    _, layers = _trained_net()
    ad = lithe.adapter_from_layers(layers, rank=8, alpha=16)
    sd = lithe.to_lora_state_dict(ad, dtype=torch.float32)
    a, b = lithe.decode_factors(ad)["attn.to_q"]
    assert torch.equal(sd["transformer.attn.to_q.lora_A.weight"], a)
    assert torch.equal(sd["transformer.attn.to_q.lora_B.weight"], b * 2.0)


def test_lazy_requantisation_and_ste_grad():
    torch.manual_seed(0)
    net = Net()
    layers = lithe.inject_lithe(net, ["attn.to_q"], rank=4, requantize_every=3)
    layer = layers["attn.to_q"]
    net.train()
    calls = []
    orig = layer.requantize
    layer.requantize = lambda: (calls.append(layer._forward_count), orig())[1]
    x = torch.randn(2, 96)
    for _ in range(7):
        net(x).sum().backward()
    assert calls == [0, 3, 6]
    assert layer.lora_A.grad is not None and layer.lora_A.grad.abs().sum() > 0


def test_load_indices_freezes_module():
    net, layers, base_net = _trained_net(return_base=True)
    ad = lithe.adapter_from_layers(layers, rank=8, alpha=16)
    torch.manual_seed(5)
    fresh = copy.deepcopy(base_net)
    new_layers = lithe.inject_lithe(fresh, ["attn.to_q", "attn.to_out.0"], rank=8, alpha=16)
    for name, layer in new_layers.items():
        layer.load_indices(torch.from_numpy(ad.indices[name]["A"].astype("int64")),
                           torch.from_numpy(ad.indices[name]["B"].astype("int64")))
    fresh.train()  # frozen modules ignore train mode
    x = torch.randn(3, 96)
    with torch.no_grad():
        assert torch.allclose(fresh(x), net.eval()(x), atol=1e-6)


def test_dense_lora_encoding_roundtrip():
    torch.manual_seed(0)
    pa, pb = lithe.twin_pools()
    proj = lithe.codebook.projection_matrix()
    true_a = quant.rvq_decode_blocks(torch.randint(0, 512, (3, 8 * 96 // 64)), pa).reshape(8, 96)
    true_b = quant.rvq_decode_blocks(torch.randint(0, 512, (3, 160 * 8 // 64)), pb).reshape(160, 8)
    sd = {"transformer.attn.to_q.lora_A.weight": true_a, "transformer.attn.to_q.lora_B.weight": true_b}
    ad = lithe.from_dense_lora(sd)
    a, b = lithe.decode_factors(ad)["attn.to_q"]
    # A uses equal scales on all levels, so different index triples can give the
    # same sum; the reconstruction is recovered, the indices need not be.
    assert torch.allclose(a, true_a, atol=1e-7) and torch.equal(b, true_b)
