# /// script
# requires-python = ">=3.10"
# dependencies = ["vizdoom", "numpy", "pillow"]
# ///
"""doom frames with ground truth from the engine's labels buffer (vizdoom defend_the_center).
a mix of an aimbot and random turning, so both empty and busy views show up. saves the 128 px center strip the
doom agent looks at, plus labels.json: center = a living monster's center is inside the strip, crosshair = under the crosshair.
usage: uv run doom_collect.py [n=800] [seed=5000] [out=$DM_DATA/frames/doom]"""
import json, os, sys
from pathlib import Path
import numpy as np
import vizdoom as vzd
from PIL import Image

n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
out = Path(sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.environ.get("DM_DATA", Path(__file__).resolve().parents[1] / "data"), "frames", "doom"))
out.mkdir(parents=True, exist_ok=True)
NOT_MONSTERS = {"DoomPlayer", "BulletPuff", "Blood", "Clip"}
ACTIONS = {"left": [1, 0, 0], "right": [0, 1, 0], "shoot": [0, 0, 1]}


def game(s):
    g = vzd.DoomGame()
    g.load_config(str(Path(vzd.scenarios_path) / "defend_the_center.cfg"))
    g.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.set_window_visible(False)
    g.set_render_hud(True)
    g.set_labels_buffer_enabled(True)
    g.set_seed(s)
    g.set_available_buttons([vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK])
    g.init()
    return g


def monsters(labels, x0, x1):
    return [l for l in labels if l.object_name not in NOT_MONSTERS and x0 <= l.x + l.width / 2 < x1]


rng = np.random.default_rng(seed)
rows, ep = [], 0
while len(rows) < n:
    g = game(seed + ep); g.new_episode(); ep += 1
    while not g.is_episode_finished() and len(rows) < n:
        s = g.get_state()
        mons = monsters(s.labels, 0, 640)
        if rng.random() < 0.5 and mons:
            c = min(mons, key=lambda l: abs(l.x + l.width / 2 - 320)); cx = c.x + c.width / 2
            act = "shoot" if abs(cx - 320) < 12 else ("left" if cx < 320 else "right")
        else:
            act = rng.choice(list(ACTIONS))
        if rng.random() < 0.35:
            name = f"{len(rows):04d}.png"
            Image.fromarray(np.ascontiguousarray(s.screen_buffer[:400, 256:384])).save(out / name)
            rows.append({"file": name, "center": len(monsters(s.labels, 256, 384)) > 0, "crosshair": any(l.x <= 320 < l.x + l.width for l in mons)})
        g.make_action(ACTIONS[act], 4)
    g.close()
(out / "labels.json").write_text(json.dumps(rows))
print(len(rows), "frames ->", out)
