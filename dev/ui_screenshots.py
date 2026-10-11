"""
dev/ui_screenshots.py -- drive the real Tkinter app on a virtual display and save screenshots (how the UI was verified).

Needs (Fedora):  sudo dnf install xorg-x11-server-Xvfb ImageMagick      (xvfb-run + the `import` command)
Run from the PROJECT ROOT:

  xvfb-run -a -s "-screen 0 1600x1000x24" python dev/ui_screenshots.py                       # auto mode
  xvfb-run -a -s "-screen 0 1600x1000x24" python dev/ui_screenshots.py --manual --pods R1-L-B4 R1-R-B4
  xvfb-run -a -s "-screen 0 1366x768x24"  python dev/ui_screenshots.py --size 1366x768       # laptop size

Writes dev/shots/shot_00.png, shot_01.png ... (one every --every seconds). Open them to look at the UI.
It uses the project's real dynamo_table.json / cycle file: back them up or pass --scratch to run in a temp copy.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--pods", nargs="*", default=["R2-M-F4", "R2-R-F1"], help="pod ids to run")
ap.add_argument("--manual", action="store_true", help="start in manual-approval mode")
ap.add_argument("--seconds", type=float, default=14, help="how long to run")
ap.add_argument("--every", type=float, default=1.5, help="seconds between screenshots")
ap.add_argument("--size", default="1500x940", help="window size WxH")
ap.add_argument("--out", default=os.path.join(ROOT, "dev", "shots"))
ap.add_argument("--scratch", action="store_true", help="run against a temporary copy of the data files")
a = ap.parse_args()

os.chdir(ROOT)
sys.path.insert(0, ROOT)
if a.scratch:                       # app_ui chdirs to its own folder, so give it a throwaway copy of the project data
    tmp = tempfile.mkdtemp()
    for f in ("dynamo_table.json", "pods_manifest.json"):
        shutil.copy(os.path.join(ROOT, f), tmp)
    print("note: --scratch only protects dynamo_table.json; crop ages are saved next to it in the project folder")
os.makedirs(a.out, exist_ok=True)

import app_ui                       # noqa: E402
from pod_registry import PODS       # noqa: E402

app = app_ui.App()
app.geometry(a.size + "+0+0")
chosen = [p for p in PODS if p["pod_id"] in a.pods]
app._selected_pods = lambda: chosen


def shot(i):
    app.update_idletasks()
    app.update()
    path = os.path.join(a.out, f"shot_{i:02d}.png")
    try:
        subprocess.run(["import", "-window", "root", path], check=True)
    except (OSError, subprocess.CalledProcessError):
        print("could not run ImageMagick `import` - install ImageMagick"); app.destroy()


def safe(fn):
    def w():
        try:
            fn()
        except Exception:
            traceback.print_exc()
    return w


def begin():
    if a.manual:
        app.toggle_auto()
    app.var_delay.set(600)
    app.start()
    app.after(300, lambda: setattr(app.engine.orders.policy, "sim_step_seconds", 0.9))


app.after(300, safe(begin))
n = int(a.seconds / a.every)
for i in range(n):
    app.after(int(1500 + i * a.every * 1000), safe(lambda i=i: shot(i)))


def finish():
    print(f"saved {n} screenshots to {a.out}")
    if app.engine:
        app.engine.close()
    app.destroy()


app.after(int(1500 + a.seconds * 1000 + 800), safe(finish))
app.mainloop()
