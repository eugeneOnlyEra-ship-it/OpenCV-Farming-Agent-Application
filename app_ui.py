"""
app_ui.py  --  OpenCV-Farming-Agent: interactive monitor

A real desktop application (Tkinter, ships with Python; the only
dependencies are still opencv-python + numpy). Instead of drawing labels
and scores INTO the OpenCV image, the photo is shown plainly and every
reading is rendered as proper UI next to it, following the layout
sketch:

    +--------------------------- Pod 1 ----------------------------+
    |  [ image ]   Disease detection scores  (bars per class)      |
    |  [ image ]   Growth detection scores   (bars per class)      |
    +----------------------------------------------------------------+
    Pod 2 is the mirror image (scores on the inside, photos outside).

Detection boxes are drawn by the UI on top of the photo (toggleable,
hover for the label), they are not baked into the pixels.

Things you can do while it runs
  Run        Start / Pause / Resume / Step / Stop        (Space, N)
  Per pod    Hold a lane on its pod, Skip the pod, Re-scan the image on
             screen, step back through that pod's earlier images
  Queue      Pick any pod and "Run next", or drop it from the run
  Tuning     Speed, model sensitivity, disease / growth action
             thresholds -- live, no restart; shown decisions re-evaluate
             instantly
  Inspect    Click a photo for a large view with both models' boxes;
             click a class row to highlight just that class's boxes
  Export     "Save log now" writes the CSV/JSON run log at any moment

Run:   python3 app_ui.py
"""

import base64
import functools
import json
import os
import queue
import sys
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import messagebox, ttk

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                      # the pipeline uses ./models, ./logs, ./dynamo_table.json
sys.path.insert(0, HERE)

import perception                                   # noqa: E402
from farm_engine import (ACTIONS, FarmEngine, Settings, class_scores,   # noqa: E402
                         healthy_class)
from pod_registry import PODS, STATION_LABELS       # noqa: E402
from worker_planner import ALL_AGENTS, PRIORITY_LABEL   # noqa: E402

# ---------------------------------------------------------------------------
# Look & feel
# ---------------------------------------------------------------------------
C = {
    "bg": "#0f131a", "card": "#171d27", "panel": "#1e2633", "panel2": "#252f3f",
    "line": "#2d394c", "text": "#e8edf4", "muted": "#8e9bb0", "dim": "#5d6b82",
    "growth": "#38c8ea", "disease": "#ff7a45", "good": "#43b05c", "warn": "#e6b422",
    "bad": "#e0463c", "harvest": "#f0932b", "accent": "#4c8dff",
}
ACTION_STYLE = {   # same four colours the OpenCV banners used
    "log_healthy": ("HEALTHY · LOGGED", "#3caa3c"),
    "flag_for_harvest": ("READY TO HARVEST", "#e6960a"),
    "schedule_frequent_monitoring": ("MONITOR CLOSELY", "#c9a90a"),
    "flag_for_treatment": ("NEEDS TREATMENT", "#dc2828"),
}
STATE_STYLE = {"idle": ("IDLE", C["dim"]), "running": ("RUNNING", C["good"]), "paused": ("PAUSED", C["warn"]),
               "finished": ("FINISHED", C["accent"]), "stopping": ("STOPPING…", C["warn"]),
               "stopped": ("STOPPED", C["dim"])}
SHORT_ACTION = {"log_healthy": "HEALTHY", "flag_for_harvest": "HARVEST",
                "schedule_frequent_monitoring": "MONITOR", "flag_for_treatment": "TREAT"}
MAX_BOXES = 25
MAX_LABELS = 6          # only the most confident few get a text label; hover shows any box


def pick_font(root):
    have = set(tkfont.families(root))
    for f in ("Segoe UI", "SF Pro Text", "Helvetica Neue", "Ubuntu", "Noto Sans", "DejaVu Sans", "Arial"):
        if f in have:
            return f
    return "TkDefaultFont"


def stage_label(name):
    """'S4_5' -> 'Stage S4_5', but 'stage 04- Harvest stage' / 'Harvest' stay as they are."""
    return f"Stage {name}" if len(name) <= 4 and name[:1] == "S" else name


def pretty(name):
    n = name.replace("_", " ")
    for pre in ("Cabbage-",):
        n = n.replace(pre, "")
    return n.replace(" on lettuce", "").strip()


@functools.lru_cache(maxsize=48)
def load_bgr(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_photo(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    try:
        return tk.PhotoImage(width=w, height=h, data=b"P6 %d %d 255\n" % (w, h) + rgb.tobytes(), format="PPM")
    except tk.TclError:
        ok, buf = cv2.imencode(".png", bgr)
        return tk.PhotoImage(data=base64.b64encode(buf.tobytes()))


# ---------------------------------------------------------------------------
# Small widgets
# ---------------------------------------------------------------------------
class Btn(tk.Label):
    """Label-as-button: renders identically on Windows/macOS/Linux (native
    tk.Button ignores colours on macOS)."""
    KINDS = {"primary": ("#2f6fe0", "#4c8dff"), "ok": ("#2c8a46", "#3fae5a"), "warn": ("#b8860b", "#d9a21b"),
             "danger": ("#b83a31", "#d4493f"), "ghost": (C["panel2"], "#2f3b50")}

    def __init__(self, parent, text, command, kind="ghost", font=None, padx=12, pady=6):
        self.base, self.hover = self.KINDS[kind]
        super().__init__(parent, text=text, bg=self.base, fg="white", font=font, padx=padx, pady=pady, cursor="hand2")
        self.command, self.enabled = command, True
        self.bind("<Enter>", lambda e: self.enabled and self.config(bg=self.hover))
        self.bind("<Leave>", lambda e: self.config(bg=self.base if self.enabled else C["panel"]))
        self.bind("<ButtonRelease-1>", self._click)

    def _click(self, e):
        if self.enabled and 0 <= e.x <= self.winfo_width() and 0 <= e.y <= self.winfo_height():
            self.command()

    def set_enabled(self, on):
        self.enabled = on
        self.config(bg=self.base if on else C["panel"], fg="white" if on else C["dim"],
                    cursor="hand2" if on else "arrow")

    def set_text(self, t):
        self.config(text=t)

    def set_kind(self, kind):
        self.base, self.hover = self.KINDS[kind]
        if self.enabled:
            self.config(bg=self.base)


class Chip(tk.Label):
    def __init__(self, parent, font, **kw):
        super().__init__(parent, font=font, padx=9, pady=2, **kw)

    def set(self, text, color, fg="white"):
        self.config(text=text, bg=color, fg=fg)


# ---------------------------------------------------------------------------
# ImageView: plain photo + UI-drawn boxes
# ---------------------------------------------------------------------------
class ImageView(tk.Canvas):
    def __init__(self, parent, fonts, on_click=None):
        super().__init__(parent, bg="#0b0e13", highlightthickness=1, highlightbackground=C["line"], cursor="hand2")
        self.f = fonts
        self.bgr, self.layers, self.badge = None, [], ""
        self.show_boxes, self.show_labels, self.highlight, self.busy = True, True, None, False
        self._photo, self._photo_key, self._hits = None, None, []
        self._after = None
        self.on_click = on_click
        self.bind("<Configure>", lambda e: self._schedule())
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda e: self.delete("tip"))
        self.bind("<Button-1>", lambda e: self.on_click and self.on_click())

    def set_content(self, bgr, layers, badge="", busy=False):
        self.bgr, self.layers, self.badge, self.busy = bgr, layers, badge, busy
        self.redraw()

    def _schedule(self):
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(40, self.redraw)

    def redraw(self):
        self._after = None
        self.delete("all")
        self._hits = []
        cw, ch = max(self.winfo_width(), 2), max(self.winfo_height(), 2)
        if self.bgr is None:
            self.create_text(cw / 2, ch / 2, text="no image", fill=C["dim"], font=self.f["small"])
            return
        h, w = self.bgr.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        ox, oy = (cw - nw) // 2, (ch - nh) // 2
        key = (id(self.bgr), nw, nh)
        if key != self._photo_key:
            self._photo = to_photo(cv2.resize(self.bgr, (nw, nh), interpolation=cv2.INTER_AREA))
            self._photo_key = key
        self.create_image(ox, oy, image=self._photo, anchor="nw")

        if self.show_boxes:
            hidden = 0
            for layer in self.layers:
                if not layer.get("visible", True):
                    continue
                dets = sorted(layer["dets"], key=lambda d: -d["confidence"])
                hidden += max(0, len(dets) - MAX_BOXES)
                for n_drawn, d in enumerate(dets[:MAX_BOXES]):
                    x, y, bw, bh = d["box"]
                    x1, y1 = ox + max(0, x) * scale, oy + max(0, y) * scale
                    x2, y2 = min(ox + nw, ox + (x + bw) * scale), min(oy + nh, oy + (y + bh) * scale)
                    dim = self.highlight is not None and d["class_name"] != self.highlight
                    base = C["good"] if layer.get("healthy") == d["class_name"] else layer["color"]
                    col = "#6b7384" if dim else base
                    self.create_rectangle(x1, y1, x2, y2, outline=col, width=1 if dim else (3 if self.highlight else 2),
                                          dash=(3, 3) if dim else ())
                    label = f"{pretty(d['class_name'])} {d['confidence']:.2f}"
                    self._hits.append((x1, y1, x2, y2, label, base))
                    if self.show_labels and not dim and (n_drawn < MAX_LABELS or self.highlight):
                        tw = self.f["tiny"].measure(label) + 8
                        lx = min(max(ox, x1), ox + nw - tw)
                        ly = max(oy, y1 - 15)
                        self.create_rectangle(lx, ly, lx + tw, ly + 15, fill=col, outline="")
                        self.create_text(lx + 4, ly + 7, text=label, anchor="w", fill="#0b0e13", font=self.f["tiny"])
            if hidden:
                self.create_text(ox + 6, oy + nh - 8, text=f"+{hidden} lower-confidence boxes not drawn",
                                 anchor="sw", fill="#cfd6e2", font=self.f["tiny"])
        if self.badge:
            tw = self.f["tiny_b"].measure(self.badge) + 12
            bx, by = ox + nw - 6 - tw, oy + nh - 6
            self.create_rectangle(bx, by - 18, bx + tw, by, fill="#000000", outline="", stipple="gray50")
            self.create_text(bx + 6, by - 9, text=self.badge, anchor="w", fill="white", font=self.f["tiny_b"])
        if self.busy:
            self.create_rectangle(ox, oy, ox + nw, oy + nh, fill="#000000", outline="", stipple="gray50")
            self.create_text(cw / 2, ch / 2, text="ANALYSING…", fill="white", font=self.f["head"])

    def _motion(self, e):
        self.delete("tip")
        best = None
        for x1, y1, x2, y2, label, col in self._hits:
            if x1 <= e.x <= x2 and y1 <= e.y <= y2:
                area = (x2 - x1) * (y2 - y1)
                if best is None or area < best[0]:
                    best = (area, label, col)
        if best:
            tw = self.f["small"].measure(best[1]) + 12
            tx = min(e.x + 12, max(0, self.winfo_width() - tw - 2))
            ty = max(2, e.y - 28)
            self.create_rectangle(tx, ty, tx + tw, ty + 20, fill="#05070a", outline=best[2], tags="tip")
            self.create_text(tx + 6, ty + 10, text=best[1], anchor="w", fill="white", font=self.f["small"], tags="tip")


# ---------------------------------------------------------------------------
# ScorePanel: "Disease detection scores" / "Growth detection scores"
# ---------------------------------------------------------------------------
class ScorePanel(tk.Frame):
    def __init__(self, parent, fonts, title, accent, on_row_click):
        super().__init__(parent, bg=C["panel"], highlightthickness=1, highlightbackground=C["line"])
        self.f, self.accent, self.on_row_click = fonts, accent, on_row_click
        tk.Label(self, text=title.upper(), bg=C["panel"], fg=accent, font=self.f["tiny_b"], anchor="w"
                 ).pack(fill="x", padx=10, pady=(8, 0))
        self.headline = tk.Label(self, text="—", bg=C["panel"], fg=C["text"], font=self.f["head"], anchor="w",
                                 justify="left")
        self.headline.pack(fill="x", padx=10)
        self.bind("<Configure>", lambda e: self.headline.config(wraplength=max(120, e.width - 22)))
        self.sub = tk.Label(self, text="", bg=C["panel"], fg=C["muted"], font=self.f["small"], anchor="w")
        self.sub.pack(fill="x", padx=10)
        self.canvas = tk.Canvas(self, bg=C["panel"], highlightthickness=0, height=60)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(4, 0))
        self.verified = tk.Label(self, text="", bg=C["panel"], fg=C["muted"], font=self.f["small"], anchor="w")
        self.verified.pack(fill="x", padx=10, pady=(0, 6))
        self.rows, self.healthy, self.threshold, self.highlight = [], None, 0.5, None
        self._row_hit, self._after, self._compact = [], None, False
        self.bind("<Configure>", self._on_panel_resize, add="+")
        self.canvas.bind("<Configure>", lambda e: self._schedule())
        self.canvas.bind("<Button-1>", self._click)

    def _on_panel_resize(self, e):
        """Short panels (4 pods on screen) drop the secondary lines so the
        score bars keep the room; a bigger window brings them back."""
        compact = e.height < 215
        if compact != self._compact:
            self._compact = compact
            if compact:
                self.sub.pack_forget()
                self.verified.pack_forget()
            else:
                self.sub.pack(fill="x", padx=10, before=self.canvas)
                self.verified.pack(fill="x", padx=10, pady=(0, 6))

    def update_content(self, rows, headline, sub, verified, healthy=None, threshold=0.5, highlight=None):
        self.headline.config(text=headline[0], fg=headline[1])
        self.sub.config(text=sub)
        self.verified.config(text=verified[0], fg=verified[1])
        self.rows, self.healthy, self.threshold, self.highlight = rows, healthy, threshold, highlight
        self.redraw()

    def clear(self, text="waiting…"):
        self.headline.config(text=text, fg=C["dim"])
        self.sub.config(text="")
        self.verified.config(text="")
        self.rows = []
        self.redraw()

    def _schedule(self):
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(40, self.redraw)

    def _click(self, e):
        for y1, y2, name in self._row_hit:
            if y1 <= e.y <= y2:
                self.on_row_click(name)
                return

    def redraw(self):
        self._after = None
        c = self.canvas
        c.delete("all")
        self._row_hit = []
        if not self.rows:
            return
        W, H = max(c.winfo_width(), 50), max(c.winfo_height(), 20)
        rows = sorted(self.rows, key=lambda r: (-r[1], 0))
        row_h = 22 if H >= 22 * len(rows) else max(15, H // max(1, len(rows)))
        fit = max(3, H // row_h)
        shown, rest = rows[:fit], rows[fit:]
        name_w = max(70, int(W * 0.47))
        bx0, bx1 = name_w + 4, W - 40
        font = self.f["small"] if row_h >= 18 else self.f["tiny"]
        for i, (name, conf, count) in enumerate(shown):
            y = i * row_h
            mid = y + row_h / 2
            self._row_hit.append((y, y + row_h, name))
            sel = self.highlight == name
            if sel:
                c.create_rectangle(0, y, W, y + row_h, fill=C["panel2"], outline="")
            label = pretty(name)
            if font.measure(label) > name_w - 4:
                while font.measure(label + "…") > name_w - 4 and len(label) > 3:
                    label = label[:-1]
                label = label.rstrip() + "…"
            col_txt = C["text"] if conf > 0 else C["dim"]
            c.create_text(2, mid, text=label, anchor="w", fill=col_txt, font=font)
            c.create_rectangle(bx0, mid - 4, bx1, mid + 4, fill=C["line"], outline="")
            if conf > 0:
                is_healthy = self.healthy is not None and name == self.healthy
                color = C["good"] if is_healthy else self.accent
                c.create_rectangle(bx0, mid - 4, bx0 + (bx1 - bx0) * min(conf, 1.0), mid + 4, fill=color, outline="")
            tx = bx0 + (bx1 - bx0) * self.threshold
            c.create_line(tx, mid - 7, tx, mid + 7, fill="#cdd5e1")
            c.create_text(W - 2, mid, text=f"{conf:.2f}" if conf > 0 else "–", anchor="e", fill=col_txt, font=font)
        if rest:
            c.create_text(2, len(shown) * row_h + 2, anchor="nw", fill=C["dim"], font=self.f["tiny"],
                          text=f"+{len(rest)} more classes (all 0.00)")


# ---------------------------------------------------------------------------
# Timeline: ◀ ● ● ○ ○ ▶  LIVE
# ---------------------------------------------------------------------------
class Timeline(tk.Canvas):
    def __init__(self, parent, fonts, on_select):
        super().__init__(parent, bg=C["card"], height=26, highlightthickness=0)
        self.f, self.on_select = fonts, on_select
        self._items, self._state = [], None
        self.bind("<Button-1>", self._click)
        self.bind("<Configure>", lambda e: self.draw(self._state))

    def draw(self, state):
        """state = dict(entries=[(color, is_rescan)], pending=int, view=int|None, live_idx=int)"""
        self.delete("all")
        self._items = []
        self._state = state
        if not state:
            return
        W = max(self.winfo_width(), 100)
        entries, pending, view = state["entries"], state["pending"], state["view"]
        cur = len(entries) - 1 if view is None else view
        x = W - 6
        # LIVE button (right aligned)
        live_on = view is None
        tw = 44
        self.create_rectangle(x - tw, 3, x, 23, fill=C["good"] if live_on else C["panel2"], outline="")
        self.create_text(x - tw / 2, 13, text="● LIVE", fill="white", font=self.f["tiny_b"])
        self._items.append((x - tw, x, ("live", None)))
        x -= tw + 8
        self.create_text(x - 6, 13, text="▶", fill=C["text"], font=self.f["small"])
        self._items.append((x - 16, x, ("next", None)))
        x -= 20
        for i in range(pending - 1, -1, -1):
            self.create_oval(x - 10, 8, x - 2, 16, outline=C["dim"])
            x -= 14
        for i in range(len(entries) - 1, -1, -1):
            color, rescan = entries[i]
            sel = i == cur
            if rescan:
                self.create_polygon(x - 6, 5, x - 1, 12, x - 6, 19, x - 11, 12, fill=color, outline="white" if sel else "")
            else:
                self.create_oval(x - 12, 6, x - 0, 18, fill=color, outline="white" if sel else "", width=2)
            self._items.append((x - 14, x + 1, ("idx", i)))
            x -= 15
        self.create_text(x - 6, 13, text="◀", fill=C["text"], font=self.f["small"])
        self._items.append((x - 16, x, ("prev", None)))

    def _click(self, e):
        st = self._state
        if not st:
            return
        n = len(st["entries"])
        cur = n - 1 if st["view"] is None else st["view"]
        for x1, x2, (kind, val) in self._items:
            if x1 <= e.x <= x2:
                if kind == "live":
                    self.on_select(None)
                elif kind == "idx":
                    self.on_select(val)
                elif kind == "prev":
                    self.on_select(max(0, cur - 1))
                elif kind == "next":
                    self.on_select(cur + 1 if cur + 1 < n - 1 else None)
                return


# ---------------------------------------------------------------------------
# WorkerPanel: replaces a score panel while a robot works on that aspect of the pod
# ---------------------------------------------------------------------------
class WorkerPanel(tk.Frame):
    """Same spot as 'Disease detection scores' / 'Growth detection scores'. After the scores have been shown, the
    robot's job on that aspect takes the panel over: every step with its live state, the agent, a progress bar,
    and (when needed) Approve / Reject. '⇄ scores' flips back to the scores while the job runs."""

    def __init__(self, parent, fonts, app, kind, on_flip):
        super().__init__(parent, bg=C["panel"], highlightthickness=2, highlightbackground=C["line"])
        self.f, self.app, self.kind = fonts, app, kind
        self.shown, self._sig, self._btn_sig, self._frac, self._compact = False, None, None, 0.0, False
        head = tk.Frame(self, bg=C["panel"])
        head.pack(fill="x", padx=10, pady=(7, 0))
        self.title = tk.Label(head, text="WORKER", bg=C["panel"], fg=C["growth"], font=fonts["tiny_b"], anchor="w")
        self.title.pack(side="left")
        flip = tk.Label(head, text="⇄ scores", bg=C["panel2"], fg=C["muted"], font=fonts["tiny_b"], padx=6,
                        cursor="hand2")
        flip.pack(side="right")
        flip.bind("<Button-1>", lambda e: on_flip(kind))
        self.headline = tk.Label(self, text="", bg=C["panel"], fg=C["text"], font=fonts["small_b"], anchor="w",
                                 justify="left", wraplength=230)
        self.headline.pack(fill="x", padx=10)
        self.bar = tk.Canvas(self, height=6, bg=C["line"], highlightthickness=0)
        self.bar.pack(side="bottom", fill="x", padx=10, pady=(2, 6))
        self.btns = tk.Frame(self, bg=C["panel"])
        self.btns.pack(side="bottom", fill="x", padx=10)
        self.sub = tk.Label(self, text="", bg=C["panel"], fg=C["muted"], font=fonts["small"], anchor="w",
                            justify="left", wraplength=230)
        self.sub.pack(fill="x", padx=10)
        self.steps = tk.Text(self, height=3, bg=C["panel"], fg=C["text"], font=fonts["small"], bd=0,
                             highlightthickness=0, wrap="word", state="disabled", padx=2, pady=2, cursor="arrow")
        self.steps.pack(fill="both", expand=True, padx=8, pady=(2, 2))
        for tag, col in (("done", C["good"]), ("todo", C["muted"]), ("act", C["text"])):
            self.steps.tag_configure(tag, foreground=col)
        self.steps.tag_configure("now", foreground=C["growth"], background=C["panel2"], font=fonts["small_b"])
        self.bind("<Configure>", self._on_resize)
        self.bar.bind("<Configure>", lambda e: self._draw_bar())

    def _on_resize(self, e):
        for w in (self.headline, self.sub):
            w.config(wraplength=max(120, e.width - 24))
        compact = e.height < 190
        if compact != self._compact:
            self._compact = compact
            if compact:
                self.sub.pack_forget()
            else:
                self.sub.pack(fill="x", padx=10, before=self.steps)

    def _draw_bar(self):
        self.bar.delete("all")
        W = max(self.bar.winfo_width(), 10)
        if self._frac > 0:
            self.bar.create_rectangle(0, 0, W * min(1.0, self._frac), 6, fill=self._bar_col, outline="")

    _bar_col = C["growth"]

    def show(self, o):
        eng = self.app.engine
        sim = eng.orders.policy.simulate_robots if eng else True
        sig = (o.order_id, o.status, o.progress, o.stale, sim)
        if sig == self._sig:
            return
        self._sig = sig
        n, st = len(o.steps), o.status
        col = TASK_COLOR.get(o.task, C["growth"])
        self.config(highlightbackground=col if st in ("in_progress", "dispatched", "approved") else C["line"])
        self.title.config(text=f"{o.role.upper()} · {TASK_LABEL.get(o.task, o.task)}", fg=col)
        self.headline.config(text=o.title)
        sub, sc = {
            "planning": ("🧠 The AI planner is checking this before it goes to the robot…", C["accent"]),
            "awaiting_approval": ("⏳ Waiting for your approval — " + (o.approval_note or "approve to send it to the robot"),
                                  C["warn"]),
            "approved": (f"→ sent to {o.agent}, starting…", C["accent"]),
            "dispatched": (f"→ sent to {o.agent}, starting…" + ("" if sim else " (waiting for a real agent)"),
                           C["accent"]),
            "in_progress": (f"▶ {o.agent} · step {min(o.progress + 1, n)}/{n}", C["growth"]),
            "completed": (f"✓ Done by {o.agent}", C["good"]),
            "rejected": ("✖ Rejected — nothing was done", C["muted"]),
            "cancelled": ("Superseded by a newer order", C["muted"]),
        }.get(st, (st, C["muted"]))
        if o.stale:
            sub += "   ⚠ a newer reading disagrees"
        self.sub.config(text=sub, fg=sc)

        t = self.steps
        t.config(state="normal")
        t.delete("1.0", "end")
        for step in o.steps:
            seq = step["seq"]
            if st == "completed" or seq <= o.progress:
                mark, tag = "✓", "done"
            elif st == "in_progress" and seq == o.progress + 1:
                mark, tag = "▶", "now"
            else:
                mark, tag = "○", "todo"
            t.insert("end", f"{mark} {seq}. {step['action']}", ("now",) if tag == "now" else ("act",))
            t.insert("end", f" — {step['text']}\n", (tag,))
        t.config(state="disabled")
        if st == "in_progress":
            t.see(f"{min(o.progress + 1, n)}.0")

        self._frac = 1.0 if st == "completed" else (o.progress / n if n else 0.0)
        self._bar_col = C["good"] if st == "completed" else col
        self._draw_bar()

        btn_sig = (o.order_id, st, sim)
        if btn_sig != self._btn_sig:
            self._btn_sig = btn_sig
            for w in self.btns.winfo_children():
                w.destroy()
            oid = o.order_id
            book = lambda: self.app.engine.orders          # noqa: E731
            if st == "awaiting_approval":
                Btn(self.btns, "✔ Approve", lambda: book().approve(oid), "ok", self.f["tiny_b"], 10, 2).pack(
                    side="left", padx=(0, 6), pady=(3, 3))
                Btn(self.btns, "✖ Reject", lambda: book().reject(oid), "danger", self.f["tiny_b"], 10, 2).pack(
                    side="left", pady=(3, 3))
            elif st in ("approved", "dispatched", "in_progress") and not sim:
                Btn(self.btns, "Mark done", lambda: book().complete(oid, note="marked done by human"), "ghost",
                    self.f["tiny_b"], 10, 2).pack(side="left", pady=(3, 3))


# ---------------------------------------------------------------------------
# PodCard: one slot, laid out like the sketch
# ---------------------------------------------------------------------------
class PodCard(tk.Frame):
    def __init__(self, parent, app, slot_id, mirrored):
        super().__init__(parent, bg=C["card"], highlightthickness=2, highlightbackground=C["line"])
        self.app, self.slot_id, self.mirrored, self.f = app, slot_id, mirrored, app.fonts
        self.view, self.highlight = None, None

        head = tk.Frame(self, bg=C["card"])
        head.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(head, text=f"POD {slot_id + 1}", bg=C["card"], fg=C["muted"], font=self.f["tiny_b"]
                 ).pack(side="left", padx=(0, 8))
        self.pod_label = tk.Label(head, text="—", bg=C["card"], fg=C["text"], font=self.f["title"])
        self.pod_label.pack(side="left")
        self.meta_label = tk.Label(head, text="", bg=C["card"], fg=C["muted"], font=self.f["small"])
        self.meta_label.pack(side="left", padx=10)
        self.action_chip = Chip(head, self.f["small_b"])
        self.action_chip.pack(side="right")
        self.flag_chip = Chip(head, self.f["tiny_b"])
        self.flag_chip.pack(side="right", padx=6)

        foot = tk.Frame(self, bg=C["card"])
        foot.pack(side="bottom", fill="x", padx=12, pady=(4, 10))     # packed before body: never clipped
        self.action_chip.pack(side="right", before=self.pod_label)      # repack first: never squeezed out
        self.flag_chip.pack(side="right", padx=6, before=self.pod_label)
        body = tk.Frame(self, bg=C["card"])
        body.pack(fill="both", expand=True, padx=12)
        body.grid_columnconfigure(0 if not mirrored else 1, weight=11, uniform="c")
        body.grid_columnconfigure(1 if not mirrored else 0, weight=9, uniform="c")
        body.grid_rowconfigure(0, weight=1, uniform="r")
        body.grid_rowconfigure(1, weight=1, uniform="r")
        img_col, sc_col = (0, 1) if not mirrored else (1, 0)

        self.disease_img = ImageView(body, self.f, on_click=self.zoom)
        self.growth_img = ImageView(body, self.f, on_click=self.zoom)
        self.disease_img.grid(row=0, column=img_col, sticky="nsew", pady=(0, 6), padx=(0, 6) if not mirrored else (6, 0))
        self.growth_img.grid(row=1, column=img_col, sticky="nsew", pady=(0, 2), padx=(0, 6) if not mirrored else (6, 0))
        self.disease_panel = ScorePanel(body, self.f, "Disease detection scores", C["disease"],
                                        lambda n: self.toggle_highlight("disease", n))
        self.growth_panel = ScorePanel(body, self.f, "Growth detection scores", C["growth"],
                                       lambda n: self.toggle_highlight("growth", n))
        self.disease_panel.grid(row=0, column=sc_col, sticky="nsew", pady=(0, 6))
        self.growth_panel.grid(row=1, column=sc_col, sticky="nsew", pady=(0, 2))
        self.disease_worker = WorkerPanel(body, self.f, app, "disease", self.flip_scores)
        self.growth_worker = WorkerPanel(body, self.f, app, "growth", self.flip_scores)
        self.disease_worker.grid(row=0, column=sc_col, sticky="nsew", pady=(0, 6))
        self.growth_worker.grid(row=1, column=sc_col, sticky="nsew", pady=(0, 2))
        self.disease_worker.grid_remove()
        self.growth_worker.grid_remove()
        self._force = {"disease": None, "growth": None}     # '⇄ scores' pins the scores back for one order
        self._scored_at, self._result_key = 0.0, None

        self.reason = tk.Label(foot, text="", bg=C["card"], fg=C["text"], font=self.f["small"], anchor="w",
                               justify="left", wraplength=500)
        self.reason.pack(fill="x")
        self.trend = tk.Label(foot, text="", bg=C["card"], fg=C["muted"], font=self.f["small"], anchor="w",
                              justify="left", wraplength=500)
        self.trend.pack(fill="x")
        self.order_label = tk.Label(foot, text="", bg=C["card"], fg=C["dim"], font=self.f["small_b"], anchor="w",
                                    cursor="hand2")
        self.order_label.pack(fill="x", pady=(2, 0))
        self._order_ref = None
        self.order_label.bind("<Button-1>", lambda e: app.open_orders(self._order_ref))
        self.foot_wrap = [self.reason, self.trend]
        foot.bind("<Configure>", lambda e: [w.config(wraplength=max(200, e.width - 4)) for w in self.foot_wrap])

        ctl = tk.Frame(foot, bg=C["card"])
        ctl.pack(fill="x", pady=(6, 0))
        self.hold_btn = Btn(ctl, "⏸ Hold pod", lambda: app.hold(slot_id), "ghost", self.f["small"], 9, 3)
        self.skip_btn = Btn(ctl, "⏭ Skip pod", lambda: app.skip(slot_id), "ghost", self.f["small"], 9, 3)
        self.scan_btn = Btn(ctl, "↻ Re-scan", lambda: app.rescan(slot_id), "ghost", self.f["small"], 9, 3)
        for b in (self.hold_btn, self.skip_btn, self.scan_btn):
            b.pack(side="left", padx=(0, 6))
        self.timeline = Timeline(ctl, self.f, lambda idx: app.set_view(slot_id, idx))
        self.timeline.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.render(None)

    # --- interaction -----------------------------------------------------
    def toggle_highlight(self, kind, name):
        self.highlight = None if self.highlight == (kind, name) else (kind, name)
        self._apply_highlight()

    def _apply_highlight(self):
        for kind, img, panel in (("disease", self.disease_img, self.disease_panel),
                                 ("growth", self.growth_img, self.growth_panel)):
            hl = self.highlight[1] if self.highlight and self.highlight[0] == kind else None
            img.highlight = hl
            panel.highlight = hl
            img.redraw()
            panel.redraw()

    def zoom(self):
        res = self._current_result()
        if res:
            ZoomWindow(self.app, res)

    def _current_result(self):
        v = self.view
        if not v or not v["results"]:
            return None
        idx = v["view_index"] if v["view_index"] is not None else len(v["results"]) - 1
        return v["results"][idx]

    def flip_scores(self, kind):
        worker = self.disease_worker if kind == "disease" else self.growth_worker
        oid = worker._sig[0] if worker._sig else None
        self._force[kind] = None if self._force[kind] == oid else oid
        self.refresh_worker_panels()

    def refresh_worker_panels(self):
        """After a pod's scores have been on screen for a moment, the robot job that belongs to the disease /
        growth aspect takes over that panel (progress in detail); afterwards the scores come back."""
        eng, v = self.app.engine, self.view
        want = {"disease": None, "growth": None}
        if (eng and v and v["pod"] is not None and v["view_index"] is None and v["results"]
                and v["status"] != "inspecting" and time.time() - self._scored_at >= SCORE_VIEW_SECONDS):
            want = eng.orders.pod_panel_orders(v["pod"]["pod_id"], WORKER_DONE_SECONDS)
        for kind, scores, worker in (("disease", self.disease_panel, self.disease_worker),
                                     ("growth", self.growth_panel, self.growth_worker)):
            o = want[kind]
            if o is not None and o.status in OPEN_ORDER_STATES and self._force[kind] == o.order_id:
                o = None                                    # you pinned the scores for this job
            if o is None:
                if worker.shown:
                    worker.grid_remove()
                    scores.grid()
                    worker.shown, worker._sig, worker._btn_sig = False, None, None
            else:
                if not worker.shown:
                    scores.grid_remove()
                    worker.grid()
                    worker.shown = True
                worker.show(o)

    def refresh_orders(self):
        """Cheap update of just the robot-task line (called on every order event)."""
        pod = self.view["pod"] if self.view else None
        book = self.app.engine.orders if self.app.engine else None
        rows = book.for_pod(pod["pod_id"]) if (pod and book) else []
        self._order_ref = rows[0][0] if rows else None
        if not pod:
            self.order_label.config(text="")
        elif not rows:
            self.order_label.config(text="🤖 no open robot task for this pod", fg=C["dim"])
        else:
            oid, agent, task, status, prog, n, stale, _, action = rows[0]
            more = f"  (+{len(rows) - 1} more)" if len(rows) > 1 else ""
            if status == "in_progress":       # live: progress bar + the step being carried out right now
                state = f"{'▰' * prog}{'▱' * (n - prog)}  step {min(prog + 1, n)}/{n}: {action}"
            else:
                state = order_status_text(status, prog, n, stale, task)
            self.order_label.config(text=f"🤖 {agent} · {task} {oid} — {state}{more}   ▸ open",
                                    fg=C["growth"] if status == "in_progress" else ORDER_STATUS[status][1])

    # --- rendering -------------------------------------------------------
    def render(self, view):
        self.view = view
        app, eng = self.app, self.app.engine
        has_engine = eng is not None
        if not view or view["pod"] is None:
            self.pod_label.config(text="—")
            self.meta_label.config(text="")
            self.action_chip.set("WAITING" if has_engine else "READY", C["panel2"], C["muted"])
            self.flag_chip.config(text="", bg=C["card"])
            idle_txt = "waiting for a pod…" if has_engine else "press ▶ Start"
            self.disease_img.set_content(None, [])
            self.growth_img.set_content(None, [])
            self.disease_panel.clear(idle_txt)
            self.growth_panel.clear(idle_txt)
            self.reason.config(text="")
            self.trend.config(text="")
            self.config(highlightbackground=C["line"])
            self.timeline.draw(None)
            for b in (self.hold_btn, self.skip_btn, self.scan_btn):
                b.set_enabled(False)
            self.hold_btn.set_text("⏸ Hold pod")
            self._result_key = None
            self.refresh_orders()
            self.refresh_worker_panels()
            return

        pod = view["pod"]
        self.pod_label.config(text=pod["pod_id"])
        self.meta_label.config(text=f"{pod['crop_type']} · rail {pod['rail']} · station "
                                    f"{STATION_LABELS[pod['station']]} · {pod['side']}")
        res = self._current_result()
        key = (res.pod_id, res.step_index, res.kind, res.timestamp) if res is not None else None
        if key != self._result_key:                       # a new reading: scores first, robot panel after a moment
            self._result_key = key
            if key is not None:
                self._scored_at = time.time()
            self._force = {"disease": None, "growth": None}
        busy = view["status"] == "inspecting"
        reviewing = view["view_index"] is not None

        # buttons
        self.hold_btn.set_text("▶ Release" if view["hold"] else "⏸ Hold pod")
        self.hold_btn.set_enabled(True)
        self.skip_btn.set_enabled(True)
        self.scan_btn.set_enabled(res is not None and not busy)

        # timeline
        entries = []
        for r in view["results"]:
            entries.append((ACTION_STYLE[eng.redecide(r.classification)["action"]][1] if has_engine else C["dim"],
                            r.kind == "rescan"))
        logged = sum(1 for r in view["results"] if r.kind == "scan")
        pending = max(0, (view["n_steps"] - view["first_step"]) - logged)
        self.timeline.draw({"entries": entries, "pending": pending, "view": view["view_index"]})

        # flag chip
        if view["hold"]:
            self.flag_chip.set("HELD", C["warn"], "#111")
        elif reviewing:
            self.flag_chip.set("REVIEWING", C["accent"])
        elif res is not None and res.kind == "rescan":
            self.flag_chip.set("RE-SCAN · NOT LOGGED", C["panel2"], C["muted"])
        elif view["status"] == "robot":
            self.flag_chip.set("🤖 ROBOT WORKING", C["growth"], "#111")
        elif view["status"] == "skipped":
            self.flag_chip.set("SKIPPED", C["panel2"], C["muted"])
        elif view["done"]:
            self.flag_chip.set("DONE", C["panel2"], C["muted"])
        else:
            self.flag_chip.config(text="", bg=C["card"])

        if res is None:
            self.action_chip.set("INSPECTING…", C["panel2"], C["muted"])
            self.disease_img.set_content(load_bgr(pod["image_path"] if not pod.get("trajectory")
                                                  else pod["trajectory"][view["first_step"]]["image_path"]),
                                         [], "", busy=True)
            self.growth_img.set_content(self.disease_img.bgr, [], "", busy=True)
            self.disease_panel.clear("analysing…")
            self.growth_panel.clear("analysing…")
            self.reason.config(text="")
            self.trend.config(text="")
            self.config(highlightbackground=C["line"])
            self.refresh_orders()
            self.refresh_worker_panels()
            return

        st = eng.settings
        cls = res.classification
        decision = eng.redecide(cls)
        label, color = ACTION_STYLE[decision["action"]]
        adjusted = decision["action"] != res.decision["action"]
        self.action_chip.set(label + (" *" if adjusted else ""), color)
        self.config(highlightbackground=color)

        img = load_bgr(res.image_path)
        badge = f"IMAGE {res.step_index + 1}/{res.n_steps}" + ("  ·  RE-SCAN" if res.kind == "rescan" else "")
        self.disease_img.set_content(img, [{"dets": cls["disease_detections"], "color": C["disease"],
                                                      "healthy": healthy_class(res.crop_type)}], badge, busy)
        self.growth_img.set_content(img, [{"dets": cls["growth_detections"], "color": C["growth"]}], badge, busy)

        # ---- disease panel
        d_rows = class_scores(cls, "disease")
        if cls["disease_flag"]:
            dh = (f"⚠ {pretty(cls['disease_name'])}  {cls['disease_confidence']:.0%}", C["disease"])
        elif cls["disease_detections"]:
            dh = (f"✓ Healthy  {cls['disease_confidence']:.0%}", C["good"])
        else:
            dh = ("No detection", C["dim"])
        ver = res.verified
        if ver["aspect"] == "disease":
            dv = ("Verified: %s  %s" % (pretty(ver["expected"]), "✓ match" if ver["ok"] else "✗ mismatch"),
                  C["good"] if ver["ok"] else C["bad"])
        else:
            dv = ("Not verified for this photo", C["dim"])
        self.disease_panel.update_content(
            d_rows, dh, f"{len(cls['disease_detections'])} detections · act at ≥{st.disease_threshold:.2f}", dv,
            healthy=healthy_class(res.crop_type), threshold=st.disease_threshold,
            highlight=self.highlight[1] if self.highlight and self.highlight[0] == "disease" else None)

        # ---- growth panel
        g_rows = class_scores(cls, "growth")
        if cls["growth_stage"]:
            gh = (f"{stage_label(cls['growth_stage'])}  {cls['growth_confidence']:.0%}", C["growth"])
        else:
            gh = ("No stage detected", C["dim"])
        if ver["aspect"] == "growth":
            gv = ("Verified: %s  %s" % (pretty(ver["expected"]), "✓ match" if ver["ok"] else "✗ mismatch"),
                  C["good"] if ver["ok"] else C["bad"])
        else:
            gv = ("Not verified for this photo", C["dim"])
        self.growth_panel.update_content(
            g_rows, gh, f"{len(cls['growth_detections'])} detections · trust at ≥{st.growth_threshold:.2f}", gv,
            threshold=st.growth_threshold,
            highlight=self.highlight[1] if self.highlight and self.highlight[0] == "growth" else None)

        reason = decision["reason"]
        if adjusted:
            reason += f"   [* re-evaluated with your thresholds; logged as {res.decision['action']}]"
        self.reason.config(text="→ " + reason)
        meta = f"{res.latency_s:.2f}s · {res.worker} · sensitivity {res.conf_threshold:.2f}"
        trend = res.trend_note if res.kind == "scan" else "re-scan: not written to history or log"
        self.trend.config(text=f"{trend}   ·   {meta}")
        self.refresh_orders()
        self.refresh_worker_panels()


# ---------------------------------------------------------------------------
# Large inspection window
# ---------------------------------------------------------------------------
class ZoomWindow(tk.Toplevel):
    def __init__(self, app, res):
        super().__init__(app, bg=C["bg"])
        self.title(f"{res.pod_id} · image {res.step_index + 1}/{res.n_steps}")
        self.geometry("1180x760")
        self.bind("<Escape>", lambda e: self.destroy())
        f, cls, st = app.fonts, res.classification, app.engine.settings
        bar = tk.Frame(self, bg=C["bg"])
        bar.pack(fill="x", padx=12, pady=8)
        self.layers = [{"dets": cls["disease_detections"], "color": C["disease"], "visible": True,
                        "healthy": healthy_class(res.crop_type)},
                       {"dets": cls["growth_detections"], "color": C["growth"], "visible": True}]
        self.vars = []
        for i, (txt, col) in enumerate((("Disease boxes", C["disease"]), ("Growth boxes", C["growth"]), ("Labels", C["text"]))):
            v = tk.BooleanVar(value=True)
            self.vars.append(v)
            tk.Checkbutton(bar, text=txt, variable=v, command=self.apply, bg=C["bg"], fg=col, selectcolor=C["panel"],
                           activebackground=C["bg"], activeforeground=col, font=f["small_b"]).pack(side="left", padx=8)
        tk.Label(bar, text="hover a box for its label · Esc closes", bg=C["bg"], fg=C["dim"], font=f["small"]
                 ).pack(side="right")
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=1, minsize=320)
        main.grid_rowconfigure(0, weight=1)
        self.view = ImageView(main, f)
        self.view.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        side = tk.Frame(main, bg=C["bg"])
        side.grid(row=0, column=1, sticky="nsew")
        side.grid_rowconfigure((0, 1), weight=1)
        side.grid_columnconfigure(0, weight=1)
        dp = ScorePanel(side, f, "Disease detection scores", C["disease"], lambda n: None)
        gp = ScorePanel(side, f, "Growth detection scores", C["growth"], lambda n: None)
        dp.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        gp.grid(row=1, column=0, sticky="nsew")
        dh = (f"⚠ {pretty(cls['disease_name'])} {cls['disease_confidence']:.0%}", C["disease"]) if cls["disease_flag"] \
            else (("✓ Healthy", C["good"]) if cls["disease_detections"] else ("No detection", C["dim"]))
        gh = (f"{stage_label(cls['growth_stage'])} {cls['growth_confidence']:.0%}", C["growth"]) if cls["growth_stage"] \
            else ("No stage detected", C["dim"])
        dp.update_content(class_scores(cls, "disease"), dh, f"{len(cls['disease_detections'])} detections", ("", C["dim"]),
                          healthy_class(res.crop_type), st.disease_threshold)
        gp.update_content(class_scores(cls, "growth"), gh, f"{len(cls['growth_detections'])} detections", ("", C["dim"]),
                          None, st.growth_threshold)
        self.view.set_content(load_bgr(res.image_path), self.layers, f"{res.pod_id}  IMAGE {res.step_index + 1}/{res.n_steps}")

    def apply(self):
        self.layers[0]["visible"], self.layers[1]["visible"] = self.vars[0].get(), self.vars[1].get()
        self.view.show_labels = self.vars[2].get()
        self.view.redraw()


# ---------------------------------------------------------------------------
# Robot orders: instructions for the camera + worker agents
# ---------------------------------------------------------------------------
ORDER_STATUS = {   # label, colour
    "planning": ("AI planner thinking…", C["accent"]),
    "awaiting_approval": ("needs your approval", C["warn"]),
    "approved": ("approved", C["good"]),
    "dispatched": ("dispatched", C["accent"]),
    "in_progress": ("in progress", C["growth"]),
    "completed": ("done", C["good"]),
    "rejected": ("rejected", C["dim"]),
    "cancelled": ("superseded", C["dim"]),
    "scheduled": ("scheduled", C["muted"]),
}
TASK_COLOR = {"TREAT": "#dc2828", "HARVEST": "#e6960a", "RECAPTURE": C["accent"], "VERIFY": C["muted"],
              "REVIEW": "#a855f7", "REPLANT": C["good"]}
TASK_LABEL = {"TREAT": "TREATMENT", "HARVEST": "HARVEST", "REPLANT": "REPLANTING", "RECAPTURE": "RE-CAPTURE",
              "VERIFY": "TREATMENT CHECK"}
SCORE_VIEW_SECONDS = 1.8     # the scores stay up this long before the robot's progress takes their panel over
WORKER_DONE_SECONDS = 3.0    # a finished robot job stays visible this long
OPEN_ORDER_STATES = {"planning", "awaiting_approval", "approved", "dispatched", "in_progress"}
VERDICT_COLOR = {"agreed": C["good"], "downgraded": C["warn"], "rejected": C["muted"], "fallback": C["muted"]}
LOOP_GOOD = {"resolved_healthy", "confirmed_disease", "confirmed_harvest", "treatment_worked"}


def order_status_text(status, progress=0, n_steps=0, stale=False, task=""):
    if status == "in_progress":
        txt = f"step {min(progress + 1, n_steps)}/{n_steps}"
    elif task == "REVIEW" and status == "awaiting_approval":
        txt = "needs your review"
    else:
        txt = ORDER_STATUS[status][0]
    return txt + ("  ⚠ stale" if stale else "")


class OrdersWindow(tk.Toplevel):
    """Everything the four robot agents are being told to do, why, and what a
    human still has to approve. Live-updating."""

    def __init__(self, app):
        super().__init__(app, bg=C["bg"])
        self.app, f = app, app.fonts
        self.title("Robot orders — camera & worker agents")
        self.geometry("1320x900+50+30")
        self.minsize(1000, 620)
        self._ver, self._selected, self._book_id, self._orders = -1, None, None, {}

        # ---- the four agents
        strip = tk.Frame(self, bg=C["bg"])
        strip.pack(fill="x", padx=12, pady=(10, 6))
        self.tiles = {}
        for i, agent in enumerate(ALL_AGENTS):
            role, side = agent.split("_")
            box = tk.Frame(strip, bg=C["panel"], highlightthickness=1, highlightbackground=C["line"])
            box.grid(row=0, column=i, sticky="ew", padx=4)
            strip.grid_columnconfigure(i, weight=1, uniform="ag")
            tk.Label(box, text=f"{side.upper()} SIDE · {role.upper()} AGENT", bg=C["panel"],
                     fg=C["accent"] if role == "camera" else C["harvest"], font=f["tiny_b"], anchor="w"
                     ).pack(fill="x", padx=10, pady=(8, 0))
            status = tk.Label(box, text="idle", bg=C["panel"], fg=C["text"], font=f["small_b"], anchor="w")
            status.pack(fill="x", padx=10)
            bar = tk.Canvas(box, height=6, bg=C["line"], highlightthickness=0)
            bar.pack(fill="x", padx=10, pady=(4, 8))
            self.tiles[agent] = (status, bar)

        # ---- toolbar
        tb = tk.Frame(self, bg=C["bg"])
        tb.pack(fill="x", padx=12, pady=(2, 6))
        self.approve_btn = Btn(tb, "✔ Approve", self.approve, "ok", f["small_b"], 12, 5)
        self.reject_btn = Btn(tb, "✖ Reject", self.reject, "danger", f["small_b"], 12, 5)
        self.all_btn = Btn(tb, "✔✔ Approve all pending", self.approve_all, "primary", f["small_b"], 12, 5)
        self.done_btn = Btn(tb, "Mark done", self.mark_done, "ghost", f["small_b"], 12, 5)
        self.copy_btn = Btn(tb, "Copy JSON", self.copy_json, "ghost", f["small_b"], 12, 5)
        for b in (self.approve_btn, self.reject_btn, self.all_btn, self.done_btn, self.copy_btn):
            b.pack(side="left", padx=(0, 6))
        pol = app.engine.orders.policy
        self.sim_var = tk.BooleanVar(value=pol.simulate_robots)
        self.loop_var = tk.BooleanVar(value=pol.close_the_loop)
        self.auto_btn = Btn(tb, "", app.toggle_auto, "ok", f["small_b"], 12, 5)
        self.auto_btn.pack(side="right", padx=(8, 0))
        for txt, var, cmd in (("Close the loop", self.loop_var, self.toggle_loop),
                              ("Simulate robots", self.sim_var, self.toggle_sim)):
            tk.Checkbutton(tb, text=txt, variable=var, command=cmd, bg=C["bg"], fg=C["text"], selectcolor=C["panel2"],
                           activebackground=C["bg"], activeforeground=C["text"], font=f["small"]
                           ).pack(side="right", padx=8)

        # ---- list + detail
        main = tk.Frame(self, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=12, pady=(0, 6))
        main.grid_columnconfigure(0, weight=5, uniform="m")
        main.grid_columnconfigure(1, weight=6, uniform="m")
        main.grid_rowconfigure(0, weight=1)
        left = tk.Frame(main, bg=C["panel"], highlightthickness=1, highlightbackground=C["line"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.tree = ttk.Treeview(left, columns=("pod", "agent", "task", "prio", "status"), selectmode="browse")
        for col, text, w in (("#0", "Order", 96), ("pod", "Pod", 74), ("agent", "Agent", 104), ("task", "Task", 96),
                             ("prio", "Priority", 70), ("status", "Status", 150)):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, stretch=(col == "status"))
        sb = ttk.Scrollbar(left, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True, padx=(6, 0), pady=6)
        for st, (_, col) in ORDER_STATUS.items():
            self.tree.tag_configure(st, foreground=col)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        right = tk.Frame(main, bg=C["panel"], highlightthickness=1, highlightbackground=C["line"])
        right.grid(row=0, column=1, sticky="nsew")
        self.d_title = tk.Label(right, text="Select an order", bg=C["panel"], fg=C["text"], font=f["title"],
                                anchor="w", justify="left")
        self.d_title.pack(fill="x", padx=12, pady=(10, 2))
        chips = tk.Frame(right, bg=C["panel"])
        chips.pack(fill="x", padx=12)
        self.c_status, self.c_task, self.c_prio = Chip(chips, f["tiny_b"]), Chip(chips, f["tiny_b"]), Chip(chips, f["tiny_b"])
        for c in (self.c_task, self.c_status, self.c_prio):
            c.pack(side="left", padx=(0, 6))
        self.d_where = tk.Label(right, text="", bg=C["panel"], fg=C["muted"], font=f["small"], anchor="w")
        self.d_where.pack(fill="x", padx=12, pady=(4, 0))
        self.d_why = tk.Label(right, text="", bg=C["panel"], fg=C["text"], font=f["small"], anchor="w", justify="left")
        self.d_why.pack(fill="x", padx=12, pady=(2, 0))
        self.d_gate = tk.Label(right, text="", bg=C["panel"], fg=C["warn"], font=f["small_b"], anchor="w", justify="left")
        self.d_gate.pack(fill="x", padx=12, pady=(2, 4))
        right.bind("<Configure>", lambda e: [w.config(wraplength=max(200, e.width - 28))
                                             for w in (self.d_title, self.d_why, self.d_gate, self.d_human, self.d_ai,
                                                       self.d_outcome)])
        self.d_ai = tk.Label(right, text="", bg=C["panel"], fg=C["muted"], font=f["small"], anchor="w", justify="left")
        self.d_ai.pack(fill="x", padx=12, pady=(0, 4))
        imgs = tk.Frame(right, bg=C["panel"])
        imgs.pack(fill="x", padx=12, pady=(0, 6))
        imgs.grid_columnconfigure(0, weight=1, uniform="im")
        imgs.grid_columnconfigure(1, weight=1, uniform="im")
        self.imgs = imgs
        self.img = ImageView(imgs, f)
        self.img.config(height=190)
        self.img.grid(row=0, column=0, sticky="ew")
        self.img_after = ImageView(imgs, f)
        self.img_after.config(height=190)
        self.img_after.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.img_after.grid_remove()
        self.d_outcome = tk.Label(right, text="", bg=C["panel"], fg=C["muted"], font=f["small_b"], anchor="w",
                                  justify="left")
        self.d_outcome.pack(fill="x", padx=12, pady=(0, 4))
        tk.Label(right, text="INSTRUCTIONS FOR THE AGENT", bg=C["panel"], fg=C["muted"], font=f["tiny_b"], anchor="w"
                 ).pack(fill="x", padx=12)
        self.steps = tk.Text(right, height=7, bg=C["panel2"], fg=C["text"], font=f["small"], bd=0, highlightthickness=0,
                             wrap="word", state="disabled", padx=8, pady=6)
        self.steps.pack(fill="both", expand=True, padx=12, pady=(2, 6))
        for tag, col in (("done", C["good"]), ("now", C["growth"]), ("todo", C["muted"]), ("act", C["text"])):
            self.steps.tag_configure(tag, foreground=col)
        self.d_human = tk.Label(right, text="", bg=C["panel"], fg=C["muted"], font=f["small"], anchor="w", justify="left")
        self.d_human.pack(fill="x", padx=12, pady=(0, 10))

        self.foot = tk.Label(self, text="", bg=C["bg"], fg=C["dim"], font=f["tiny"], anchor="w")
        self.foot.pack(fill="x", padx=14, pady=(0, 8))
        self.after(250, self._poll)

    # --- selection / actions -------------------------------------------------
    def select(self, order_id):
        if self.tree.exists(order_id):
            self.tree.selection_set(order_id)
            self.tree.see(order_id)

    def _on_select(self, _e=None):
        sel = self.tree.selection()
        self._selected = sel[0] if sel else None
        self._show()

    def _book(self):
        return self.app.engine.orders if self.app.engine else None

    def approve(self):
        b = self._book()
        if b and self._selected:
            b.approve(self._selected)

    def reject(self):
        b = self._book()
        if b and self._selected:
            b.reject(self._selected)

    def approve_all(self):
        b = self._book()
        if b:
            n = b.approve_all_pending()
            self.app._log_line(f"approved {n} order(s)", "ok")

    def mark_done(self):
        b = self._book()
        if b and self._selected:
            b.complete(self._selected, note="marked done by human")

    def copy_json(self):
        o = self._orders.get(self._selected)
        if o:
            self.clipboard_clear()
            self.clipboard_append(json.dumps(o.to_dict(), indent=2))
            self.app._log_line(f"{o.order_id} copied to clipboard as JSON", "ok")

    def toggle_loop(self):
        b = self._book()
        if b:
            b.policy.close_the_loop = self.loop_var.get()

    def toggle_sim(self):
        b = self._book()
        if b:
            b.set_simulation(self.sim_var.get())

    # --- live refresh -----------------------------------------------------------
    def _poll(self):
        if not self.winfo_exists():
            return
        b = self._book()
        if b is not None:
            if id(b) != self._book_id:          # a new run started: start clean
                self._book_id, self._ver, self._selected = id(b), -1, None
                self.tree.delete(*self.tree.get_children())
                self.sim_var.set(b.policy.simulate_robots)
                self.loop_var.set(b.policy.close_the_loop)
                self.foot.config(text=f"Orders, outbox and trace are saved in  {b.out_dir}/")
            if b.version != self._ver:
                self._ver = b.version
                self._refresh_list(b)
            self._update_fleet(b)
            self._update_buttons(b)
        self.after(250, self._poll)

    def _refresh_list(self, book):
        orders = book.list_orders()
        self._orders = {o.order_id: o for o in orders}
        have = set(self.tree.get_children())
        for i, o in enumerate(orders):
            vals = (o.pod_id, o.agent, o.task, PRIORITY_LABEL[o.priority],
                    order_status_text(o.status, o.progress, len(o.steps), o.stale, o.task))
            if o.order_id in have:
                self.tree.item(o.order_id, values=vals, tags=(o.status,))
            else:
                self.tree.insert("", "end", iid=o.order_id, text=o.order_id, values=vals, tags=(o.status,))
            self.tree.move(o.order_id, "", i)
        for gone in have - set(self._orders):
            self.tree.delete(gone)
        self._show()

    def _show(self):
        o = self._orders.get(self._selected)
        if not o:
            self.d_title.config(text="Select an order")
            for c in (self.c_status, self.c_task, self.c_prio):
                c.config(text="", bg=C["panel"])
            self.d_where.config(text="")
            self.d_why.config(text="Instructions for the camera and worker agents appear here as images are analysed.")
            self.d_gate.config(text="")
            self.d_human.config(text="")
            self.d_ai.config(text="")
            self.d_outcome.config(text="")
            self.img_after.grid_remove()
            self.img.set_content(None, [])
            self._set_steps([])
            return
        label, col = ORDER_STATUS[o.status]
        self.d_title.config(text=f"{o.order_id} · {o.title}")
        self.c_task.set(o.task, TASK_COLOR[o.task])
        self.c_status.set(order_status_text(o.status, o.progress, len(o.steps), o.stale, o.task).upper(), col,
                          "#111" if o.status in ("awaiting_approval",) else "white")
        self.c_prio.set(f"{PRIORITY_LABEL[o.priority]} PRIORITY", C["panel2"], C["muted"])
        self.d_where.config(text=f"{o.agent}  ·  {o.location}  ·  " + (f"visit {o.evidence['visit']}" if o.evidence["visit"] else "follow-up frame")
                                 + (f"  ·  seen {o.occurrences}×" if o.occurrences > 1 else "")
                                 + (f"  ·  follow-up of {o.follow_up_of}" if o.follow_up_of else ""))
        why = "Why: " + o.rationale
        if o.stale:
            why += "\n⚠ A newer reading contradicts this order — review before approving."
        self.d_why.config(text=why)
        self.d_gate.config(text=("✋ " + o.approval_note) if o.approval_note and o.status == "awaiting_approval" else "")
        w, h = o.evidence["image_size"]
        dets = [{"box": (t["box_norm"][0] * w, t["box_norm"][1] * h, t["box_norm"][2] * w, t["box_norm"][3] * h),
                 "class_name": t["class"], "confidence": t["confidence"]} for t in o.targets]
        try:
            self.img.set_content(load_bgr(o.evidence["image"]),
                                 [{"dets": dets, "color": TASK_COLOR[o.task]}], f"{len(dets)} TARGET REGION(S)")
        except FileNotFoundError:
            self.img.set_content(None, [])
        if o.ai:
            prop = o.ai.get("proposal")
            txt = f"🧠 AI planner ({o.ai['backend']}): {o.ai['verdict'].upper()} — {o.ai['why']}"
            if prop and prop.get("reason"):
                txt += f"\n     it said: “{prop['reason']}”  [wanted {prop['task']}, urgency {prop['urgency']}]"
            self.d_ai.config(text=txt, fg=VERDICT_COLOR.get(o.ai["verdict"], C["muted"]))
        else:
            self.d_ai.config(text="")
        oc = o.outcome
        if oc:
            try:
                aw, ah = oc["after_size"]
                adets = [{"box": (t["box_norm"][0] * aw, t["box_norm"][1] * ah, t["box_norm"][2] * aw,
                                  t["box_norm"][3] * ah), "class_name": t["class"], "confidence": t["confidence"]}
                         for t in oc["after_boxes"]]
                self.img_after.set_content(load_bgr(oc["frame"]), [{"dets": adets, "color": C["accent"]}],
                                           "NEW FRAME" + (" · SIMULATED" if oc["simulated"] else ""))
                self.img_after.grid()
            except FileNotFoundError:
                self.img_after.grid_remove()
            self.d_outcome.config(
                text=f"🔁 LOOP RESULT: {oc['headline']}\n     new frame: {oc['frame_note']}"
                     + ("   (SIMULATED frame — see README)" if oc["simulated"] else ""),
                fg=C["good"] if oc["result"] in LOOP_GOOD else C["warn"])
        else:
            self.img_after.grid_remove()
            self.d_outcome.config(text="")
        self._set_steps([(s["seq"], s["action"], s["text"]) for s in o.steps], o.progress, o.status)
        self.d_human.config(text=("FOR THE HUMAN:\n• " + "\n• ".join(o.human_notes)) if o.human_notes else "")

    def _set_steps(self, steps, progress=0, status=""):
        t = self.steps
        t.config(state="normal")
        t.delete("1.0", "end")
        for seq, action, text in steps:
            if status == "completed" or seq <= progress:
                mark, tag = "✓", "done"
            elif status == "in_progress" and seq == progress + 1:
                mark, tag = "▶", "now"
            else:
                mark, tag = "○", "todo"
            t.insert("end", f"{mark} {seq}. ", tag)
            t.insert("end", f"{action}  ", ("act",))
            t.insert("end", text + "\n", tag if tag != "todo" else "todo")
        t.config(state="disabled")

    def _update_fleet(self, book):
        fleet = book.fleet
        waiting = {a: sum(1 for o in self._orders.values() if o.agent == a and o.status == "dispatched")
                   for a in ALL_AGENTS}
        for agent, (label, bar) in self.tiles.items():
            bar.delete("all")
            if fleet is None:
                label.config(text=f"external agent · {waiting[agent]} waiting" if waiting[agent] else "external agent · no orders",
                             fg=C["muted"])
                continue
            st = fleet.state[agent]
            q = fleet.queue_len(agent)
            if st["status"] == "busy":
                label.config(text=f"{st['order']} · step {st['step'] + 1}/{st['n']}" + (f"  (+{q} queued)" if q else ""),
                             fg=C["growth"])
                W = max(bar.winfo_width(), 10)
                bar.create_rectangle(0, 0, W * (st["step"] + 1) / max(1, st["n"]), 6, fill=C["growth"], outline="")
            else:
                label.config(text="idle" + (f"  ({q} queued)" if q else ""), fg=C["muted"])

    def _update_buttons(self, book):
        o = self._orders.get(self._selected)
        st = o.status if o else ""
        self.approve_btn.set_text("▶ Run now" if st == "scheduled" else
                                  ("✔ Mark reviewed" if o and o.task == "REVIEW" else "✔ Approve"))
        self.approve_btn.set_enabled(st in ("awaiting_approval", "scheduled"))
        self.reject_btn.set_enabled(st in ("awaiting_approval", "scheduled"))
        self.done_btn.set_enabled(st in ("dispatched", "in_progress", "approved") and not book.policy.simulate_robots)
        self.copy_btn.set_enabled(o is not None)
        self.app.style_auto_button(self.auto_btn)
        n = book.pending_robot_count()
        self.all_btn.set_text(f"✔✔ Approve all robot jobs ({n})")
        self.all_btn.set_enabled(n > 0)


# ---------------------------------------------------------------------------
# FleetStrip: the four robot agents, always visible in the main window
# ---------------------------------------------------------------------------
class FleetStrip(tk.Frame):
    """Live view of what each agent is doing, so you can watch automatic mode without opening Robot orders.
    Click a tile to jump to that order."""

    def __init__(self, parent, app):
        super().__init__(parent, bg=C["bg"])
        self.app, f = app, app.fonts
        self.f = f
        self.tiles, self._ref = {}, {}
        for i, agent in enumerate(ALL_AGENTS):
            role, side = agent.split("_")
            box = tk.Frame(self, bg=C["panel"], highlightthickness=1, highlightbackground=C["line"], cursor="hand2")
            box.grid(row=0, column=i, sticky="ew", padx=3)
            self.grid_columnconfigure(i, weight=1, uniform="fleet")
            head = tk.Label(box, text=f"{side.upper()} SIDE · {role.upper()}", bg=C["panel"], anchor="w",
                            fg=C["accent"] if role == "camera" else C["harvest"], font=f["tiny_b"])
            head.pack(fill="x", padx=8, pady=(5, 0))
            status = tk.Label(box, text="idle", bg=C["panel"], fg=C["muted"], anchor="w", font=f["small"])
            status.pack(fill="x", padx=8)
            bar = tk.Canvas(box, height=5, bg=C["line"], highlightthickness=0)
            bar.pack(fill="x", padx=8, pady=(2, 6))
            self.tiles[agent] = (box, status, bar)
            for w in (box, head, status, bar):
                w.bind("<Button-1>", lambda e, a=agent: self.app.open_orders(self._ref.get(a)))

    def update_status(self, st):
        for agent, (box, label, bar) in self.tiles.items():
            bar.delete("all")
            info = st.get(agent) if st else None
            self._ref[agent] = None
            if not info:
                label.config(text="idle", fg=C["dim"])
                box.config(highlightbackground=C["line"])
                continue
            act, extra = info["active"], []
            if info["queued"]:
                extra.append(f"+{info['queued']}")
            if info["awaiting"]:
                extra.append(f"⏳{info['awaiting']}")
            tail = ("   " + "  ·  ".join(extra)) if extra else ""
            if act:
                self._ref[agent] = act["order_id"]
                label.config(text=f"{act['order_id']} {act['task']} {act['pod']} · {act['step']}/{act['n']} "
                                  f"{act['action']}{tail}", fg=C["growth"])
                box.config(highlightbackground=C["growth"])
                W = max(bar.winfo_width(), 10)
                bar.create_rectangle(0, 0, W * act["step"] / max(1, act["n"]), 5, fill=C["growth"], outline="")
            elif info["awaiting"] or info["queued"]:
                label.config(text=("waiting" + tail).strip(), fg=C["warn"] if info["awaiting"] else C["muted"])
                box.config(highlightbackground=C["warn"] if info["awaiting"] else C["line"])
            elif info["last_done"]:
                d = info["last_done"]
                self._ref[agent] = d["order_id"]
                label.config(text=f"✓ done: {d['order_id']} {d['task']} {d['pod']}", fg=C["good"])
                box.config(highlightbackground=C["line"])
            else:
                label.config(text="idle", fg=C["dim"])
                box.config(highlightbackground=C["line"])


# ---------------------------------------------------------------------------
# The application window
# ---------------------------------------------------------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OpenCV Farming Agent — Interactive Monitor")
        self.geometry("1560x940")
        self.minsize(1180, 720)
        self.configure(bg=C["bg"])
        fam = pick_font(self)
        self.fonts = {
            "title": tkfont.Font(family=fam, size=13, weight="bold"), "head": tkfont.Font(family=fam, size=12, weight="bold"),
            "small": tkfont.Font(family=fam, size=9), "small_b": tkfont.Font(family=fam, size=9, weight="bold"),
            "tiny": tkfont.Font(family=fam, size=8), "tiny_b": tkfont.Font(family=fam, size=8, weight="bold"),
            "big": tkfont.Font(family=fam, size=15, weight="bold"), "norm": tkfont.Font(family=fam, size=10),
            "norm_b": tkfont.Font(family=fam, size=10, weight="bold"),
        }
        self._style()
        self.engine = None
        self._orders_win = None
        self.auto_approve = True          # master switch: robot orders approved + executed automatically
        self.cards = []
        self.settings = Settings()
        self.state = "idle"
        self._build_toolbar()
        self._build_summary()
        self.fleet_strip = FleetStrip(self, self)
        self.fleet_strip.pack(fill="x", padx=9, pady=(2, 2))
        self._build_body()
        self._build_cards()
        self._fill_queue_preview()
        self._set_state("idle")
        self.bind("<space>", lambda e: self._key(self.toggle_pause))
        self.bind("<n>", lambda e: self._key(self.step))
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(80, self._pump)
        self.after(250, self._tick)

    # --- styling ---------------------------------------------------------
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Treeview", background=C["panel"], fieldbackground=C["panel"], foreground=C["text"],
                    rowheight=22, borderwidth=0, font=self.fonts["small"])
        s.configure("Treeview.Heading", background=C["panel2"], foreground=C["muted"], borderwidth=0,
                    font=self.fonts["small_b"])
        s.map("Treeview", background=[("selected", C["accent"])], foreground=[("selected", "white")])
        s.configure("Horizontal.TScale", background=C["panel"], troughcolor=C["line"])
        s.configure("Horizontal.TProgressbar", background=C["good"], troughcolor=C["line"], borderwidth=0)
        s.configure("TCombobox", fieldbackground=C["panel2"], background=C["panel2"], foreground=C["text"],
                    arrowcolor=C["text"], bordercolor=C["line"])
        s.map("TCombobox", fieldbackground=[("readonly", C["panel2"])], foreground=[("readonly", C["text"])])
        s.configure("TSpinbox", fieldbackground=C["panel2"], background=C["panel2"], foreground=C["text"],
                    arrowcolor=C["text"], bordercolor=C["line"])
        self.option_add("*TCombobox*Listbox.background", C["panel2"])
        self.option_add("*TCombobox*Listbox.foreground", C["text"])

    # --- toolbar ---------------------------------------------------------
    def _build_toolbar(self):
        f = self.fonts
        bar = tk.Frame(self, bg=C["panel"], highlightthickness=1, highlightbackground=C["line"])
        bar.pack(fill="x")
        tk.Label(bar, text="🌱 Farm Agent", bg=C["panel"], fg=C["text"], font=f["big"]).pack(side="left", padx=(14, 10), pady=8)
        self.state_chip = Chip(bar, f["small_b"])
        self.state_chip.pack(side="left", padx=(0, 14))
        self.start_btn = Btn(bar, "▶  Start", self.start, "ok", f["norm_b"])
        self.pause_btn = Btn(bar, "⏸  Pause", self.toggle_pause, "warn", f["norm_b"])
        self.step_btn = Btn(bar, "⏭  Step", self.step, "ghost", f["norm_b"])
        self.stop_btn = Btn(bar, "■  Stop", self.stop, "danger", f["norm_b"])
        self.save_btn = Btn(bar, "💾 Save log now", self.save_log, "ghost", f["norm"])
        for b in (self.start_btn, self.pause_btn, self.step_btn, self.stop_btn):
            b.pack(side="left", padx=3)
        self.save_btn.pack(side="left", padx=(14, 3))
        self.orders_btn = Btn(bar, "🤖 Robot orders", self.open_orders, "ghost", f["norm_b"])
        self.orders_btn.pack(side="left", padx=3)
        self.auto_btn = Btn(bar, "", self.toggle_auto, "ok", f["norm_b"])
        self.auto_btn.pack(side="left", padx=3)
        self.style_auto_button(self.auto_btn)

    # --- summary strip ---------------------------------------------------
    def _build_summary(self):
        f = self.fonts
        strip = tk.Frame(self, bg=C["bg"])
        strip.pack(fill="x", padx=12, pady=(10, 4))
        left = tk.Frame(strip, bg=C["bg"])
        left.pack(side="left", fill="x", expand=True)
        self.progress_text = tk.Label(left, text="no run yet", bg=C["bg"], fg=C["text"], font=f["norm_b"], anchor="w")
        self.progress_text.pack(fill="x")
        self.progress = ttk.Progressbar(left, maximum=100, length=300)
        self.progress.pack(fill="x", pady=(3, 0), padx=(0, 16))
        right = tk.Frame(strip, bg=C["bg"])
        right.pack(side="right")
        self.action_chips = {}
        for a in ACTIONS:
            label, col = ACTION_STYLE[a]
            chip = Chip(right, f["small_b"])
            chip.set(f"{SHORT_ACTION[a]}  0", col)
            chip.pack(side="left", padx=3)
            self.action_chips[a] = chip
        self.acc_chip = Chip(right, f["small_b"])
        self.acc_chip.set("accuracy  –", C["panel2"], C["muted"])
        self.acc_chip.pack(side="left", padx=(10, 0))

    # --- body: sidebar + cards + log ------------------------------------
    def _build_body(self):
        f = self.fonts
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=12, pady=(4, 10))
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        side = tk.Frame(body, bg=C["bg"], width=300)
        side.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 10))
        side.grid_propagate(False)
        side.pack_propagate(False)

        setup = tk.Frame(side, bg=C["panel"], highlightthickness=1, highlightbackground=C["line"])
        setup.pack(fill="x", pady=(0, 8))
        tk.Label(setup, text="RUN SETUP", bg=C["panel"], fg=C["muted"], font=f["tiny_b"], anchor="w"
                 ).grid(row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(8, 2))

        def lbl(text, r, c):
            tk.Label(setup, text=text, bg=C["panel"], fg=C["muted"], font=f["small"]).grid(
                row=r, column=c, sticky="w", padx=(10, 4), pady=2)
        lbl("Crop", 1, 0)
        self.crop_var = tk.StringVar(value="all")
        self.crop_cb = ttk.Combobox(setup, textvariable=self.crop_var, values=["all", "cabbage", "lettuce", "mushroom"],
                                    width=9, state="readonly")
        self.crop_cb.grid(row=1, column=1, columnspan=3, sticky="ew", padx=(0, 10), pady=2)
        self.crop_cb.bind("<<ComboboxSelected>>", lambda e: self._fill_queue_preview())
        lbl("Pods (0=all)", 2, 0)
        self.pods_var = tk.StringVar(value="0")
        self.pods_sp = ttk.Spinbox(setup, from_=0, to=72, width=3, textvariable=self.pods_var,
                                   command=self._fill_queue_preview)
        self.pods_sp.grid(row=2, column=1, sticky="w", pady=2)
        lbl("On screen", 2, 2)
        self.slots_var = tk.StringVar(value="2")
        self.slots_sp = ttk.Spinbox(setup, from_=1, to=4, width=3, textvariable=self.slots_var, state="readonly",
                                    command=self._build_cards)
        self.slots_sp.grid(row=2, column=3, sticky="w", padx=(0, 8), pady=2)
        self.reset_var = tk.BooleanVar(value=False)
        self.reset_cb = tk.Checkbutton(setup, text="Reset history", variable=self.reset_var, bg=C["panel"],
                                       fg=C["muted"], selectcolor=C["panel2"], activebackground=C["panel"],
                                       activeforeground=C["text"], font=f["small"])
        self.reset_cb.grid(row=3, column=0, columnspan=4, sticky="w", padx=(6, 0), pady=(2, 8))

        qbox = tk.Frame(side, bg=C["panel"], highlightthickness=1, highlightbackground=C["line"])
        qbox.pack(fill="both", expand=True)
        tk.Label(qbox, text="POD QUEUE", bg=C["panel"], fg=C["muted"], font=f["tiny_b"], anchor="w"
                 ).pack(fill="x", padx=10, pady=(8, 2))
        self.tree = ttk.Treeview(qbox, columns=("crop", "status"), height=3, selectmode="browse")
        self.tree.heading("#0", text="Pod")
        self.tree.heading("crop", text="Crop")
        self.tree.heading("status", text="Status")
        self.tree.column("#0", width=76, stretch=False)
        self.tree.column("crop", width=82, stretch=False)
        self.tree.column("status", width=100)
        sb = ttk.Scrollbar(qbox, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 8))
        sb.pack(side="left", fill="y", pady=(0, 8), padx=(0, 4))
        for tag, col in (("running", C["growth"]), ("done", C["good"]), ("skipped", C["dim"]), ("queued", C["text"])):
            self.tree.tag_configure(tag, foreground=col)
        self.tree.bind("<Double-1>", lambda e: self.queue_run_next())
        qbtn = tk.Frame(side, bg=C["bg"])
        qbtn.pack(fill="x", pady=(6, 8))
        self.next_btn = Btn(qbtn, "⤒ Run next", self.queue_run_next, "primary", f["small_b"], 10, 5)
        self.drop_btn = Btn(qbtn, "✕ Drop from run", self.queue_drop, "ghost", f["small_b"], 10, 5)
        self.next_btn.pack(side="left", padx=(0, 6))
        self.drop_btn.pack(side="left")

        plan = tk.Frame(side, bg=C["panel"], highlightthickness=1, highlightbackground=C["line"])
        plan.pack(fill="x", pady=(0, 8))
        tk.Label(plan, text="AI PLANNER (OPTIONAL)", bg=C["panel"], fg=C["muted"], font=f["tiny_b"], anchor="w"
                 ).pack(fill="x", padx=10, pady=(8, 2))
        self.planner_var = tk.StringVar(value="Rules only")
        self.planner_cb = ttk.Combobox(plan, textvariable=self.planner_var, state="readonly",
                                       values=["Rules only", "Scripted demo (not an LLM)", "Local LLM via Ollama"])
        self.planner_cb.pack(fill="x", padx=10)
        self.planner_cb.bind("<<ComboboxSelected>>", lambda e: self._planner_changed())
        mrow = tk.Frame(plan, bg=C["panel"])
        mrow.pack(fill="x", padx=10, pady=(4, 0))
        self.model_var = tk.StringVar(value="llama3.2:3b")
        self.model_entry = tk.Entry(mrow, textvariable=self.model_var, width=16, bg=C["panel2"], fg=C["text"],
                                    insertbackground=C["text"], relief="flat", font=f["small"])
        self.model_entry.pack(side="left", fill="x", expand=True)
        self.model_entry.bind("<Return>", lambda e: self._planner_changed())
        self.model_entry.bind("<FocusOut>", lambda e: self._planner_changed())
        self.pimg_var = tk.BooleanVar(value=False)
        tk.Checkbutton(mrow, text="send photo", variable=self.pimg_var, command=self._planner_changed, bg=C["panel"],
                       fg=C["text"], selectcolor=C["panel2"], activebackground=C["panel"],
                       activeforeground=C["text"], font=f["small"]).pack(side="left", padx=(6, 0))
        self.planner_note = tk.Label(plan, text="Rule table decides. An AI planner can only ask for a better look.",
                                     bg=C["panel"], fg=C["dim"], font=f["tiny"], justify="left", anchor="w",
                                     wraplength=270)
        self.planner_note.pack(fill="x", padx=10, pady=(3, 8))
        self.model_entry.config(state="disabled")

        tune = tk.Frame(side, bg=C["panel"], highlightthickness=1, highlightbackground=C["line"])
        tune.pack(fill="x")
        tk.Label(tune, text="LIVE TUNING", bg=C["panel"], fg=C["muted"], font=f["tiny_b"], anchor="w"
                 ).pack(fill="x", padx=10, pady=(8, 0))
        self.var_delay = tk.DoubleVar(value=self.settings.delay_ms)
        self.var_conf = tk.DoubleVar(value=self.settings.conf_threshold)
        self.var_dis = tk.DoubleVar(value=self.settings.disease_threshold)
        self.var_gro = tk.DoubleVar(value=self.settings.growth_threshold)
        self._slider(tune, "Delay between images", self.var_delay, 0, 3000, "{:.0f} ms")
        self._slider(tune, "Model sensitivity (box floor)", self.var_conf, 0.10, 0.80, "{:.2f}")
        self._slider(tune, "Disease action threshold", self.var_dis, 0.10, 0.95, "{:.2f}")
        self._slider(tune, "Growth trust threshold", self.var_gro, 0.10, 0.95, "{:.2f}")
        tk.Label(tune, text="Sensitivity applies to the next scan / re-scan.\nThresholds re-evaluate shown decisions now.",
                 bg=C["panel"], fg=C["dim"], font=f["tiny"], justify="left", anchor="w").pack(fill="x", padx=10, pady=(2, 4))
        row = tk.Frame(tune, bg=C["panel"])
        row.pack(fill="x", padx=10, pady=(0, 8))
        self.boxes_var = tk.BooleanVar(value=True)
        self.labels_var = tk.BooleanVar(value=True)
        for txt, var in (("Boxes", self.boxes_var), ("Box labels", self.labels_var)):
            tk.Checkbutton(row, text=txt, variable=var, command=self._apply_overlay, bg=C["panel"], fg=C["text"],
                           selectcolor=C["panel2"], activebackground=C["panel"], activeforeground=C["text"],
                           font=f["small"]).pack(side="left", padx=(0, 8))
        Btn(row, "Defaults", self._defaults, "ghost", f["tiny_b"], 7, 2).pack(side="right")
        for v in (self.var_delay, self.var_conf, self.var_dis, self.var_gro):
            v.trace_add("write", lambda *a: self._settings_changed())

        self.cards_frame = tk.Frame(body, bg=C["bg"])
        self.cards_frame.grid(row=0, column=1, sticky="nsew")

        logbox = tk.Frame(body, bg=C["panel"], highlightthickness=1, highlightbackground=C["line"])
        logbox.grid(row=1, column=1, sticky="ew", pady=(8, 0))
        self.log = tk.Text(logbox, height=5, bg=C["panel"], fg=C["text"], font=f["small"], bd=0, highlightthickness=0,
                           state="disabled", wrap="none", padx=8, pady=6)
        lsb = ttk.Scrollbar(logbox, command=self.log.yview)
        self.log.configure(yscrollcommand=lsb.set)
        lsb.pack(side="right", fill="y")
        self.log.pack(fill="both", expand=True)
        for tag, col in (("info", C["muted"]), ("warn", C["warn"]), ("error", C["bad"]), ("ok", C["good"]), ("time", C["dim"])):
            self.log.tag_configure(tag, foreground=col)

    def _slider(self, parent, text, var, lo, hi, fmt):
        f = self.fonts
        row = tk.Frame(parent, bg=C["panel"])
        row.pack(fill="x", padx=10, pady=(6, 0))
        tk.Label(row, text=text, bg=C["panel"], fg=C["text"], font=f["small"], anchor="w").pack(side="left")
        val = tk.Label(row, text=fmt.format(var.get()), bg=C["panel"], fg=C["accent"], font=f["small_b"])
        val.pack(side="right")
        var.trace_add("write", lambda *a: val.config(text=fmt.format(var.get())))
        ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal").pack(fill="x", padx=10)

    # --- cards -----------------------------------------------------------
    def _build_cards(self):
        n = int(self.slots_var.get())
        for c in self.cards:
            c.destroy()
        self.cards = []
        cols = 1 if n == 1 else 2
        rows = (n + cols - 1) // cols
        for i in range(4):
            self.cards_frame.grid_columnconfigure(i, weight=0)
            self.cards_frame.grid_rowconfigure(i, weight=0)
        for c in range(cols):
            self.cards_frame.grid_columnconfigure(c, weight=1, uniform="cc")
        for r in range(rows):
            self.cards_frame.grid_rowconfigure(r, weight=1, uniform="rr")
        for i in range(n):
            card = PodCard(self.cards_frame, self, i, mirrored=(i % cols == 1))
            card.grid(row=i // cols, column=i % cols, sticky="nsew", padx=5, pady=5)
            self.cards.append(card)
        self._apply_overlay()

    def _apply_overlay(self):
        for c in self.cards:
            for img in (c.disease_img, c.growth_img):
                img.show_boxes, img.show_labels = self.boxes_var.get(), self.labels_var.get()
                img.redraw()

    # --- queue sidebar ---------------------------------------------------
    def _selected_pods(self):
        crop = self.crop_var.get()
        pods = [p for p in PODS if crop == "all" or p["crop_type"] == crop]
        try:
            lim = int(self.pods_var.get())
        except ValueError:
            lim = 0
        return pods[:lim] if lim > 0 else pods

    def _fill_queue_preview(self):
        if self.engine is not None and self.state not in ("idle", "stopped"):
            return
        self.tree.delete(*self.tree.get_children())
        for p in self._selected_pods():
            self.tree.insert("", "end", iid=p["pod_id"], text=p["pod_id"], values=(p["crop_type"], "queued"), tags=("queued",))

    def _refresh_queue(self):
        eng = self.engine
        if not eng:
            return
        for p in eng.pods:
            pid = p["pod_id"]
            if not self.tree.exists(pid):
                continue
            st = eng.pod_status[pid]
            if st == "queued":
                text = f"queued #{eng.queue_position(pid)}"
            elif st == "running":
                text = f"▶ slot {eng.pod_slot.get(pid, 0) + 1}"
            else:
                text = "done ✓" if st == "done" else "skipped"
            self.tree.item(pid, values=(p["crop_type"], text), tags=(st,))

    def queue_run_next(self):
        sel = self.tree.selection()
        if self.engine and sel and self.state in ("running", "paused"):
            if not self.engine.run_next(sel[0]):
                self._log_line("only queued pods can be moved", "warn")

    def queue_drop(self):
        sel = self.tree.selection()
        if self.engine and sel and self.state in ("running", "paused"):
            if not self.engine.skip_queued(sel[0]):
                self._log_line("only queued pods can be dropped (use ⏭ Skip pod on a running one)", "warn")

    # --- AI planner ------------------------------------------------------
    def _planner_kind(self):
        return {"Rules only": "rules", "Scripted demo (not an LLM)": "scripted",
                "Local LLM via Ollama": "ollama"}[self.planner_var.get()]

    def _planner_changed(self):
        kind = self._planner_kind()
        self.model_entry.config(state="normal" if kind == "ollama" else "disabled")
        notes = {"rules": "Rule table decides. An AI planner can only ask for a better look.",
                 "scripted": "Hand-written stand-in for testing the wiring. It is NOT an LLM.",
                 "ollama": "Needs Ollama running locally (ollama.com). Runs offline; if it is down, rules are kept."}
        self.planner_note.config(text=notes[kind])
        if self.engine and self.state in ("running", "paused", "finished"):
            self.engine.set_planner(kind, self.model_var.get().strip() or "llama3.2:3b", self.pimg_var.get())

    # --- tuning ----------------------------------------------------------
    def _settings_changed(self):
        s = self.settings
        s.delay_ms = int(self.var_delay.get())
        s.conf_threshold = round(self.var_conf.get(), 2)
        s.disease_threshold = round(self.var_dis.get(), 2)
        s.growth_threshold = round(self.var_gro.get(), 2)
        if getattr(self, "_rerender", None):
            self.after_cancel(self._rerender)
        self._rerender = self.after(80, self._render_all)

    def _defaults(self):
        d = Settings()
        self.var_delay.set(d.delay_ms)
        self.var_conf.set(d.conf_threshold)
        self.var_dis.set(d.disease_threshold)
        self.var_gro.set(d.growth_threshold)

    def _render_all(self):
        self._rerender = None
        if self.engine:
            for i, c in enumerate(self.cards):
                c.render(self.engine.slot_view(i))

    # --- run controls ----------------------------------------------------
    def start(self):
        if self.state in ("running", "paused", "stopping"):
            return
        if self.engine is not None:
            self.engine.close()
        pods = self._selected_pods()
        if not pods:
            return
        if self.reset_var.get() and not messagebox.askyesno(
                "Reset history", "This deletes dynamo_table.json, so every pod restarts at image 1.\nContinue?"):
            return
        n = int(self.slots_var.get())
        self._build_cards()
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")
        self.status("loading models…")
        self.update_idletasks()
        perception.warm_up()
        self.engine = FarmEngine(pods, n_slots=n, settings=self.settings, reset_history=self.reset_var.get())
        self.reset_var.set(False)
        self.tree.delete(*self.tree.get_children())
        for p in pods:
            self.tree.insert("", "end", iid=p["pod_id"], text=p["pod_id"], values=(p["crop_type"], "queued"), tags=("queued",))
        self.engine.orders.policy.auto_approve_all = self.auto_approve
        self.engine.start()
        if self._planner_kind() != "rules":
            self.engine.set_planner(self._planner_kind(), self.model_var.get().strip() or "llama3.2:3b",
                                    self.pimg_var.get())
        self._refresh_queue()

    def toggle_pause(self):
        if not self.engine:
            return
        if self.state == "running":
            self.engine.pause()
        elif self.state == "paused":
            self.engine.resume()

    def step(self):
        if self.engine and self.state == "paused":
            self.engine.step_once()

    def stop(self):
        if self.engine and self.state in ("running", "paused", "finished"):
            self.engine.stop()

    def save_log(self):
        if not self.engine or not self.engine.logger.rows:
            self._log_line("nothing to save yet", "warn")
            return
        info = self.engine._flush()
        self._log_line(f"log saved: {info['csv']}", "ok")

    def style_auto_button(self, btn):
        if self.auto_approve:
            btn.set_text("⚡ Auto-approve: ON")
            btn.set_kind("ok")
        else:
            btn.set_text("✋ Manual approval")
            btn.set_kind("warn")

    def toggle_auto(self):
        """Master switch. ON: every robot order is approved and executed automatically. OFF: orders created from
        now on wait for you. (Escalations to a human are always manual.)"""
        self.auto_approve = not self.auto_approve
        self.style_auto_button(self.auto_btn)
        if self.engine:
            self.engine.orders.set_auto_approve_all(self.auto_approve)
        self._log_line("auto-approve ON: new robot orders are approved and run automatically" if self.auto_approve
                       else "MANUAL approval: new robot orders will wait for you in 🤖 Robot orders",
                       "ok" if self.auto_approve else "warn")

    def open_orders(self, select=None):
        if not self.engine:
            self._log_line("start a run first - robot orders are created from the analysed images", "warn")
            return
        w = self._orders_win
        if w is None or not w.winfo_exists():
            self._orders_win = w = OrdersWindow(self)
        w.deiconify()
        w.lift()
        if select:
            w.after(300, lambda: w.select(select))

    # per-slot controls
    def hold(self, i):
        self.engine and self.engine.toggle_hold(i)

    def skip(self, i):
        self.engine and self.engine.skip_pod(i)

    def rescan(self, i):
        self.engine and self.engine.rescan(i)

    def set_view(self, i, idx):
        self.engine and self.engine.set_view(i, idx)

    def _key(self, fn):
        if isinstance(self.focus_get(), (ttk.Spinbox, ttk.Combobox, tk.Entry)):
            return
        fn()

    # --- state / rendering loop -----------------------------------------
    def _set_state(self, state):
        self.state = state
        text, col = STATE_STYLE[state]
        self.state_chip.set(text, col, "#111" if state in ("paused", "stopping") else "white")
        active = state in ("running", "paused")
        self.start_btn.set_enabled(state in ("idle", "finished", "stopped"))
        self.start_btn.set_text("▶  New run" if state in ("finished", "stopped") else "▶  Start")
        self.pause_btn.set_enabled(active)
        self.pause_btn.set_text("▶  Resume" if state == "paused" else "⏸  Pause")
        self.step_btn.set_enabled(state == "paused")
        self.stop_btn.set_enabled(active or state == "finished")
        cfg_on = "normal" if not active and state != "stopping" else "disabled"
        self.crop_cb.config(state="readonly" if cfg_on == "normal" else "disabled")
        self.pods_sp.config(state=cfg_on)
        self.slots_sp.config(state="readonly" if cfg_on == "normal" else "disabled")
        self.reset_cb.config(state=cfg_on)
        for b in (self.next_btn, self.drop_btn):
            b.set_enabled(active)

    def status(self, text):
        self.progress_text.config(text=text)

    def _log_line(self, text, level="info", t=None):
        self.log.config(state="normal")
        self.log.insert("end", (t or time.strftime("%H:%M:%S")) + "  ", "time")
        self.log.insert("end", text + "\n", level)
        self.log.see("end")
        self.log.config(state="disabled")

    def _pump(self):
        eng = self.engine
        if eng:
            slots, queue_dirty, orders_dirty = set(), False, False
            try:
                while True:
                    kind, payload = eng.events.get_nowait()
                    if kind == "slot":
                        slots.add(payload)
                    elif kind == "queue":
                        queue_dirty = True
                    elif kind == "orders":
                        orders_dirty = True
                    elif kind == "log":
                        self._log_line(payload[2], payload[1], payload[0])
                    elif kind == "state":
                        self._set_state(payload)
                    elif kind == "finished":
                        self._announce(payload, "Run complete")
                    elif kind == "stopped":
                        self._set_state("stopped")
                        self._announce(payload, "Run stopped")
            except queue.Empty:
                pass
            for i in slots:
                if i < len(self.cards):
                    self.cards[i].render(eng.slot_view(i))
            if queue_dirty:
                self._refresh_queue()
            if orders_dirty:
                for c in self.cards:
                    c.refresh_orders()
                    c.refresh_worker_panels()
        self.after(80, self._pump)

    def _announce(self, info, title):
        s = info["summary"]
        self._log_line(f"{title}: growth accuracy {s['growth_stage_accuracy']}, disease accuracy "
                       f"{s['disease_flag_accuracy']}", "ok")
        if info.get("csv"):
            self._log_line(f"log: {info['csv']}", "ok")

    def _tick(self):
        eng = self.engine
        if eng:
            st = eng.stats()
            pct = 100 * st["pods_done"] / st["total"] if st["total"] else 0
            self.progress.config(value=pct)
            m, s = divmod(int(st["elapsed"]), 60)
            self.progress_text.config(
                text=f"{st['pods_done']}/{st['total']} pods  ·  {st['steps']} scans  ·  {m:02d}:{s:02d} elapsed  ·  "
                     f"{st['avg_latency']:.2f}s avg inference")
            for a, chip in self.action_chips.items():
                chip.config(text=f"{SHORT_ACTION[a]}  {st['actions'][a]}")
            g, d = st["growth"], st["disease"]
            self.acc_chip.set(f"growth {g[0]}/{g[1]}   disease {d[0]}/{d[1]}", C["panel2"], C["text"])
            self.fleet_strip.update_status(eng.orders.fleet_status())
            for c in self.cards:
                c.refresh_worker_panels()
            pending = eng.orders.pending_count()
            self.orders_btn.set_text("🤖 Robot orders" + (f" · {pending} to approve" if pending else ""))
            self.orders_btn.set_kind("warn" if pending else "ghost")
        self.after(250, self._tick)

    def on_close(self):
        if self.engine and self.state in ("running", "paused", "finished"):
            if self.state != "finished" and not messagebox.askyesno(
                    "Quit", "A run is in progress. Stop it, save the log and quit?"):
                return
        if self.engine:
            self.status("saving log and shutting down…")
            self.update_idletasks()
            self.engine.close()
        self.destroy()


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
