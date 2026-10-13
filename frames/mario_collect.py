# /// script
# requires-python = "==3.11.*"
# dependencies = ["gym-super-mario-bros", "numpy<2", "pillow"]
# ///
"""usage: uv run mario_collect.py [out=$DM_DATA/frames/mario] [per stage=500]
mario crops for training: a scripted player (walk right, random jumps, jump when stuck) on several stages.
saves the 'ahead' window (same crop as d1-plays mario.py --ahead 20 56, 4x nearest) every 4 frames."""
import json, sys, warnings
from pathlib import Path

import gym_super_mario_bros
import numpy as np
from nes_py.wrappers import JoypadSpace
from PIL import Image

warnings.filterwarnings("ignore")
AHEAD, SCALE = (20, 56), 4
import os
out = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("DM_DATA", Path(__file__).resolve().parents[1] / "data"), "frames", "mario"))
out.mkdir(parents=True, exist_ok=True)
per_stage = int(sys.argv[2]) if len(sys.argv) > 2 else 500
rng = np.random.default_rng(0)
rows = []
for stage in ["1-1", "1-2", "1-3", "2-1", "3-1", "4-1"]:
    n = 0
    while n < per_stage:
        env = JoypadSpace(gym_super_mario_bros.make(f"SuperMarioBros-{stage}-v0", apply_api_compatibility=True, render_mode="rgb_array"),
                          [["right"], ["right", "A"]])
        nes = env
        while not hasattr(nes, "ram"):
            nes = nes.env
        frame, _ = env.reset()
        xs, t, done = [], 0, False
        while not done and n < per_stage:
            mx = int(nes.ram[0x3AD])
            if t % 4 == 0 and rng.random() < 0.5:
                x0 = int(np.clip(mx + AHEAD[0], 0, 255 - AHEAD[1]))
                c = Image.fromarray(frame).crop((x0, 140, x0 + AHEAD[1], 232))
                c = c.resize((c.width * SCALE, c.height * SCALE), Image.NEAREST)
                name = f"{stage}_{len(rows):05d}.png"
                c.save(out / name)
                rows.append({"file": name, "stage": stage})
                n += 1
            stuck = len(xs) > 30 and xs[-1] == xs[-30]
            a = 1 if (stuck or rng.random() < 0.03) else 0
            for _ in range(40 if a else 4):
                frame, _, term, trunc, info = env.step(a)
                t += 1
                if term or trunc or info["life"] < 2:
                    done = True
                    break
            xs.append(info["x_pos"])
            if t > 3000:
                break
        env.close()
(out / "rows.json").write_text(json.dumps(rows))
print(len(rows), "crops")
