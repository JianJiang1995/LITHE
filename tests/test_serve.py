import torch
from torch import nn

import lithe
from lithe import serve


def _adapters(n=3, seed=0):
    rng = torch.Generator().manual_seed(seed)
    ads = []
    for _ in range(n):
        modules = [{"name": "fc", "shape_a": [4, 128], "shape_b": [64, 4]}]
        indices = {"fc": {"A": torch.randint(0, 512, (3, 8), generator=rng).numpy().astype("uint16"),
                          "B": torch.randint(0, 512, (3, 4), generator=rng).numpy().astype("uint16")}}
        ads.append(lithe.LitheAdapter(modules=modules, indices=indices, rank=4, alpha=8.0))
    return ads


class One(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        return self.fc(x)


def test_heterogeneous_batch_matches_per_sample():
    torch.manual_seed(0)
    model = One()
    base = nn.Linear(128, 64)
    base.load_state_dict(model.fc.state_dict())
    ads = _adapters()
    router, _ = serve.attach_adapters(model, ads)
    x = torch.randn(5, 10, 128)
    routes = [[(0, 1.0)], [(2, 1.0)], [], [(1, 0.6), (2, 0.4)], [(0, 1.0)]]
    router.set(routes)
    with torch.no_grad():
        y = model(x)
    factors = [lithe.decode_factors(ad)["fc"] for ad in ads]
    for i, route in enumerate(routes):
        ref = base(x[i])
        for a_id, w in route:
            a, b = factors[a_id]
            ref = ref + w * (x[i] @ a.T @ b.T) * 2.0
        assert torch.allclose(y[i], ref, atol=1e-6), i


def test_parse_route():
    assert serve.parse_route("1") == [(1, 1.0)]
    assert serve.parse_route("0:0.6,2:0.4") == [(0, 0.6), (2, 0.4)]
    assert serve.parse_route("none") == []
