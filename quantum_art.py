# quantum_painter.py
# 1-qubit H->measure 샷으로 컬러 이미지를 생성 (다채로운 팔레트/옵션 지원)
# pip install qiskit qiskit-aer pillow numpy

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont  #  ImageDraw, ImageFont 추가

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

@dataclass
class PaintConfig:
    width: int = 32
    height: int = 32
    bits_per_channel: int = 8     # 1,2,4,8 권장 (8이면 24bit RGB)
    palette: str = "direct"       # "direct"=직접 0..255, "rainbow", "blue_orange"
    theta: float | None = None    # None=H(0.5), 값 주면 Ry(theta): p(1)=sin^2(theta/2)
    block: int = 1                # >1이면 block×block을 같은 색(지역 상관성/얽힘 분위기)
    seed: int | None = 7          # 시뮬레이터 시드(재현)
    out_path: str = "quantum_art.bmp"

def _bits_to_int(bits: np.ndarray) -> np.ndarray:
    weights = (1 << np.arange(bits.shape[1], dtype=np.uint32))
    return (bits * weights).sum(axis=1, dtype=np.uint32)

def _apply_palette(rgb_0_255: np.ndarray, palette: str) -> np.ndarray:
    if palette == "direct":
        return rgb_0_255
    arr = rgb_0_255.astype(np.float32) / 255.0
    r, g, b = arr[:,0], arr[:,1], arr[:,2]
    if palette == "rainbow":
        h = (0.6*r + 0.3*g + 0.1*b) % 1.0
        s = 0.6 + 0.4*(0.5*r + 0.5*g)
        v = 0.7 + 0.3*(0.5*b + 0.5*r)
        i = (h*6).astype(int) % 6
        f = h*6 - i
        p = v*(1-s)
        q = v*(1-f*s)
        t = v*(1-(1-f)*s)
        out = np.zeros_like(arr)
        mask = (i==0); out[mask] = np.stack([v[mask], t[mask], p[mask]], axis=1)
        mask = (i==1); out[mask] = np.stack([q[mask], v[mask], p[mask]], axis=1)
        mask = (i==2); out[mask] = np.stack([p[mask], v[mask], t[mask]], axis=1)
        mask = (i==3); out[mask] = np.stack([p[mask], q[mask], v[mask]], axis=1)
        mask = (i==4); out[mask] = np.stack([t[mask], p[mask], v[mask]], axis=1)
        mask = (i==5); out[mask] = np.stack([v[mask], p[mask], q[mask]], axis=1)
        return (np.clip(out, 0, 1)*255).astype(np.uint8)
    if palette == "blue_orange":
        x = 0.3*r + 0.4*g + 0.3*b
        c1 = np.stack([0.1*np.ones_like(x), 0.2*np.ones_like(x), 0.7*np.ones_like(x)], axis=1)
        c2 = np.stack([0.95*np.ones_like(x), 0.6*np.ones_like(x), 0.15*np.ones_like(x)], axis=1)
        out = c1*(1-x[:,None]) + c2*(x[:,None])
        return (np.clip(out, 0, 1)*255).astype(np.uint8)
    return rgb_0_255

def _generate_bits_with_one_qubit(total_bits: int, theta: float | None, seed: int | None, batch: int = 500000) -> np.ndarray:
    sim = AerSimulator()
    out = np.empty(total_bits, dtype=np.uint8)
    done = 0
    while done < total_bits:
        n = min(batch, total_bits - done)
        qc = QuantumCircuit(1,1)
        if theta is None:
            qc.h(0)
        else:
            qc.ry(theta, 0)
        qc.measure(0,0)
        job = sim.run(qc, shots=n, memory=True, seed_simulator=seed)
        res = job.result()
        mem = res.get_memory()
        out[done:done+n] = (np.frombuffer("".join(mem).encode(), dtype='S1') == b'1').astype(np.uint8)
        done += n
    return out

def quantum_paint(cfg: PaintConfig) -> Path:
    W, H = cfg.width, cfg.height
    B = cfg.bits_per_channel
    if B not in (1,2,4,8):
        raise ValueError("bits_per_channel must be one of 1,2,4,8")
    if cfg.block < 1:
        cfg.block = 1

    BW = math.ceil(W / cfg.block)
    BH = math.ceil(H / cfg.block)
    n_blocks = BW * BH
    bits_per_block = 3 * B
    total_bits = n_blocks * bits_per_block

    meas = _generate_bits_with_one_qubit(total_bits, cfg.theta, cfg.seed)
    meas = meas.reshape(n_blocks, bits_per_block)
    meas = meas.reshape(n_blocks, 3, B)

    vals = np.zeros((n_blocks, 3), dtype=np.uint8)
    for ch in range(3):
        v = _bits_to_int(meas[:, ch, :])
        if B < 8:
            vmax = (1 << B) - 1
            v = (v.astype(np.float32) * (255.0 / vmax)).astype(np.uint8)
        vals[:, ch] = v.astype(np.uint8)

    vals = _apply_palette(vals, cfg.palette)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    idx = 0
    for by in range(BH):
        for bx in range(BW):
            y0, y1 = by*cfg.block, min((by+1)*cfg.block, H)
            x0, x1 = bx*cfg.block, min((bx+1)*cfg.block, W)
            img[y0:y1, x0:x1, :] = vals[idx]
            idx += 1

    out = Path(cfg.out_path)
    Image.fromarray(img).save(out)
    return out

#  워터마크 함수 추가
def add_watermark_bottom_right(
    img_path: str,
    text: str = "- AYJ QUANTUM ART -",
    margin: int = 30,
    font_size: int | None = None,
    font_path: str | None = None,
    fill=(255,255,255,230),
    shadow=(0,0,0,160),
    shadow_offset=(3,3),
    pad_box=10,
    box_fill=(0,0,0,100)
):
    im = Image.open(img_path).convert("RGBA")
    W, H = im.size

    if font_size is None:
        font_size = max(20, W // 28)
    try:
        font = ImageFont.truetype(font_path or "arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    overlay = Image.new("RGBA", im.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    bbox = draw.textbbox((0,0), text, font=font)
    tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x = W - tw - margin
    y = H - th - margin

    if pad_box > 0:
        box = (x - pad_box, y - pad_box, x + tw + pad_box, y + th + pad_box)
        draw.rounded_rectangle(box, radius=pad_box, fill=box_fill)

    if shadow:
        draw.text((x+shadow_offset[0], y+shadow_offset[1]), text, font=font, fill=shadow)
    draw.text((x, y), text, font=font, fill=fill)

    out = Image.alpha_composite(im, overlay).convert("RGB")
    out.save(img_path)
    print(f" Watermark added → {img_path}")

# 메인 실행부
if __name__ == "__main__":
    cfg = PaintConfig(
        width=1024, height=1024,
        bits_per_channel=8,
        palette="rainbow",
        theta=None,
        block=1,
        seed=7,
        out_path="quantum_art_1024x1024.bmp"
    )
    out = quantum_paint(cfg)
    print("Saved:", out.resolve())

    # 워터마크 자동 추가
    add_watermark_bottom_right(str(out))
