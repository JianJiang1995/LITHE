#!/usr/bin/env python
"""Train a LITHE adapter on FLUX.1-dev from a small image folder (DreamBooth style).

    python scripts/train_lithe_flux.py \
        --pretrained ./checkpoints/FLUX.1-dev \
        --instance_data_dir ./data/my_cat \
        --instance_prompt "a photo of a <v_sub_001> cat" \
        --output_dir ./outputs/my_cat

Writes ``<output_dir>/adapter.lithe`` (index file, about 1.8 MB at r = 16) and,
every ``save_every`` steps, ``<output_dir>/checkpoint-<step>.lithe``.

Single GPU, no prior preservation, frozen bf16 backbone, fp32 LoRA proxies.
The prompt embedding is computed once and the text encoders are then freed;
the training images are encoded to latents once.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from PIL.ImageOps import exif_transpose

import lithe

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_config(path, overrides: dict) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for dotted, value in overrides.items():
        if value is None:
            continue
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = value
    return cfg


def load_images(folder, resolution: int, center_crop: bool = True, random_flip: bool = False) -> torch.Tensor:
    """Images in ``folder`` -> (N, 3, res, res) in [-1, 1]."""
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"no images in {folder}")
    out = []
    for p in paths:
        img = exif_transpose(Image.open(p)).convert("RGB")
        scale = resolution / min(img.size)
        img = img.resize((max(resolution, round(img.width * scale)), max(resolution, round(img.height * scale))),
                         Image.BILINEAR)
        if center_crop:
            left, top = (img.width - resolution) // 2, (img.height - resolution) // 2
        else:
            left, top = random.randint(0, img.width - resolution), random.randint(0, img.height - resolution)
        img = img.crop((left, top, left + resolution, top + resolution))
        if random_flip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        out.append(arr)
    return torch.stack(out)


@torch.no_grad()
def encode_latents(vae, images: torch.Tensor, device, dtype) -> torch.Tensor:
    lat = []
    for img in images:
        dist = vae.encode(img[None].to(device, dtype=vae.dtype)).latent_dist
        z = (dist.sample() - vae.config.shift_factor) * vae.config.scaling_factor
        lat.append(z.to(dtype))
    return torch.cat(lat)


def flow_matching_loss(transformer, latents, prompt_embeds, pooled, text_ids, scheduler,
                       guidance_scale: float = 3.5, weighting_scheme: str = "logit_normal",
                       logit_mean: float = 0.0, logit_std: float = 1.0) -> torch.Tensor:
    """Rectified-flow loss of the FLUX diffusers trainer for a batch of packed-able latents."""
    from diffusers import FluxPipeline
    from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

    bsz, ch, h, w = latents.shape
    device, dtype = latents.device, latents.dtype
    u = compute_density_for_timestep_sampling(weighting_scheme=weighting_scheme, batch_size=bsz,
                                              logit_mean=logit_mean, logit_std=logit_std, mode_scale=1.29)
    idx = (u * scheduler.config.num_train_timesteps).long().clamp(max=scheduler.config.num_train_timesteps - 1)
    timesteps = scheduler.timesteps[idx].to(device)
    sigmas = scheduler.sigmas[idx].to(device=device, dtype=dtype).view(-1, 1, 1, 1)
    noise = torch.randn_like(latents)
    noisy = (1.0 - sigmas) * latents + sigmas * noise
    packed = FluxPipeline._pack_latents(noisy, bsz, ch, h, w)
    img_ids = FluxPipeline._prepare_latent_image_ids(bsz, h // 2, w // 2, device, dtype)
    guidance = None
    if transformer.config.guidance_embeds:
        guidance = torch.full((bsz,), guidance_scale, device=device, dtype=torch.float32)
    pred = transformer(hidden_states=packed, timestep=timesteps / 1000, guidance=guidance,
                       pooled_projections=pooled.expand(bsz, -1), encoder_hidden_states=prompt_embeds.expand(bsz, -1, -1),
                       txt_ids=text_ids, img_ids=img_ids, return_dict=False)[0]
    vae_scale = 8
    pred = FluxPipeline._unpack_latents(pred, height=h * vae_scale, width=w * vae_scale, vae_scale_factor=vae_scale)
    weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
    target = noise - latents
    return (weighting.float() * (pred.float() - target.float()) ** 2).reshape(bsz, -1).mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "flux_lithe_default.yaml"))
    ap.add_argument("--pretrained", default=os.environ.get("FLUX_MODEL_PATH"),
                    help="HF repo id or local folder (default: $FLUX_MODEL_PATH, else config)")
    ap.add_argument("--instance_data_dir", required=True)
    ap.add_argument("--instance_prompt", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_train_steps", type=int)
    ap.add_argument("--resolution", type=int)
    ap.add_argument("--gradient_accumulation_steps", type=int)
    ap.add_argument("--learning_rate", type=float)
    ap.add_argument("--rank", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--save_every", type=int)
    ap.add_argument("--gradient_checkpointing", action="store_true", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = load_config(args.config, {
        "model.pretrained": args.pretrained,
        "train.max_train_steps": args.max_train_steps, "train.resolution": args.resolution,
        "train.gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train.learning_rate": args.learning_rate, "lora.rank": args.rank, "train.seed": args.seed,
        "train.save_every": args.save_every, "train.gradient_checkpointing": args.gradient_checkpointing,
    })
    if args.rank is not None:
        cfg["lora"]["alpha"] = args.rank
    mc, lc, tc = cfg["model"], cfg["lora"], cfg["train"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_config.json").write_text(json.dumps(
        {**cfg, "instance_prompt": args.instance_prompt}, indent=2))

    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline, FluxTransformer2DModel

    torch.manual_seed(tc["seed"])
    random.seed(tc["seed"])
    np.random.seed(tc["seed"])
    device, dtype = torch.device(args.device), DTYPES[mc["dtype"]]
    path = mc["pretrained"]

    # 1) prompt embedding once, then free the text encoders
    text_pipe = FluxPipeline.from_pretrained(path, transformer=None, vae=None, torch_dtype=dtype).to(device)
    with torch.no_grad():
        prompt_embeds, pooled, text_ids = text_pipe.encode_prompt(
            prompt=args.instance_prompt, prompt_2=None, device=device,
            max_sequence_length=tc["max_sequence_length"])
    del text_pipe
    torch.cuda.empty_cache()

    # 2) latents once, then free the VAE
    images = load_images(args.instance_data_dir, tc["resolution"], tc["center_crop"], tc["random_flip"])
    vae = AutoencoderKL.from_pretrained(path, subfolder="vae", torch_dtype=dtype).to(device)
    latents = encode_latents(vae, images, device, dtype)
    del vae
    torch.cuda.empty_cache()
    print(f"[lithe] {latents.shape[0]} images -> latents {tuple(latents.shape[1:])}")

    # 3) frozen transformer + LITHE wrappers
    transformer = FluxTransformer2DModel.from_pretrained(path, subfolder="transformer", torch_dtype=dtype).to(device)
    transformer.requires_grad_(False)
    if tc["gradient_checkpointing"]:
        transformer.enable_gradient_checkpointing()
    layers = lithe.inject_lithe(transformer, mc["target_modules"], rank=lc["rank"], alpha=lc["alpha"],
                                requantize_every=cfg["lithe"]["requantize_every"],
                                drift_tau=cfg["lithe"]["drift_tau"])
    params = [p for layer in layers.values() for p in (layer.lora_A, layer.lora_B)]
    n_params = sum(p.numel() for p in params)
    print(f"[lithe] wrapped {len(layers)} linears, {n_params / 1e6:.2f} M proxy parameters")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")
    opt = torch.optim.AdamW(params, lr=tc["learning_rate"], betas=(tc["adam_beta1"], tc["adam_beta2"]),
                            eps=tc["adam_epsilon"], weight_decay=tc["adam_weight_decay"])
    if tc["lr_scheduler"] != "constant":
        raise ValueError("only the constant schedule of the paper recipe is implemented")

    def save(name: str):
        ad = lithe.adapter_from_layers(layers, rank=lc["rank"], alpha=lc["alpha"], base_model=str(mc["pretrained"]),
                                       metadata={"instance_prompt": args.instance_prompt,
                                                 "steps": step + 1, "resolution": tc["resolution"]})
        size = lithe.save_lithe(out_dir / name, ad)
        print(f"[lithe] wrote {out_dir / name} ({size / 1e6:.2f} MB, {ad.num_indices} indices)")

    transformer.train()
    ga, n_img = tc["gradient_accumulation_steps"], latents.shape[0]
    t0 = time.time()
    for step in range(tc["max_train_steps"]):
        total = 0.0
        for _ in range(ga):
            batch = latents[torch.randint(0, n_img, (tc["train_batch_size"],))]
            loss = flow_matching_loss(transformer, batch, prompt_embeds, pooled, text_ids, scheduler,
                                      tc["guidance_scale"], tc["weighting_scheme"], tc["logit_mean"], tc["logit_std"])
            (loss / ga).backward()
            total += loss.item() / ga
        torch.nn.utils.clip_grad_norm_(params, tc["max_grad_norm"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0 or step == tc["max_train_steps"] - 1:
            rate = (step + 1) / (time.time() - t0)
            print(f"[lithe] step {step + 1}/{tc['max_train_steps']} loss {total:.4f} ({rate:.2f} it/s)", flush=True)
        if tc["save_every"] and (step + 1) % tc["save_every"] == 0 and step + 1 < tc["max_train_steps"]:
            save(f"checkpoint-{step + 1}.lithe")
    save("adapter.lithe")


if __name__ == "__main__":
    main()
