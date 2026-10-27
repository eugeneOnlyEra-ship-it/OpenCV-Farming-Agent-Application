"""check_ui_tabs.py -- the tabbed layout of app_ui.py (Operations / Log / Settings).

Builds the real window, checks the widgets the rest of the code relies on still exist, that the settings widgets live
on the Settings tab, that warnings reach the ticker and put an alert on the Log tab, and that the score-detail
switch works. Needs a display; without one it prints SKIP (on a headless box:  xvfb-run python dev/check_ui_tabs.py).

  python dev/check_ui_tabs.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import tkinter as tk
try:
    tk.Tk().destroy()
except tk.TclError:
    print("SKIP: no display available"); sys.exit(0)
import app_ui

app = app_ui.App()
app.update()
tabs = [app.notebook.tab(i, "text") for i in range(app.notebook.index("end"))]
assert tabs == ["Operations", "Calibration", "Pod tags", "Cloud", "Log", "Settings"], tabs

def parent_tab(widget):
    """which notebook tab a widget lives on"""
    w = widget
    while w is not None:
        for i, tab in enumerate(app.notebook.tabs()):
            if str(w) == tab:
                return app.notebook.tab(i, "text")
        w = w.master
    return None
on = {n: parent_tab(getattr(app, n)) for n in
      ("tree", "next_btn", "cards_frame", "ticker", "log", "save_btn", "crop_cb", "pods_sp", "slots_sp", "reset_cb",
       "planner_cb", "model_entry", "pybullet_check", "detail_cb", "cal_btn", "cal_text")}
expect = {"tree": "Operations", "next_btn": "Operations", "cards_frame": "Operations", "ticker": "Operations",
          "log": "Log", "save_btn": "Log", "crop_cb": "Settings", "pods_sp": "Settings", "slots_sp": "Settings",
          "reset_cb": "Settings", "planner_cb": "Settings", "model_entry": "Settings", "pybullet_check": "Settings",
          "detail_cb": "Settings", "cal_btn": "Calibration", "cal_text": "Calibration"}
assert on == expect, {k: (on[k], expect[k]) for k in on if on[k] != expect[k]}

app._log_line("all fine", "info")
assert "all fine" in app.ticker.cget("text") and app.notebook.tab(app._log_tab_index, "text") == "Log"
app._log_line("something went wrong", "error")
assert "something went wrong" in app.ticker.cget("text") and "⚠ 1" in app.notebook.tab(app._log_tab_index, "text")
app.notebook.select(app._log_tab_index); app.update()
assert app.notebook.tab(app._log_tab_index, "text") == "Log", "alert should clear when the Log tab is opened"
app.notebook.select(0)

assert all(not c.full_detail for c in app.cards), "default should be headline only"
app.detail_var.set("Full scores (every class)"); app._apply_detail()
assert all(c.full_detail and c.trend.winfo_manager() == "pack" for c in app.cards)
app.detail_var.set("Headline result only"); app._apply_detail()
assert all(not c.full_detail and c.trend.winfo_manager() == "" for c in app.cards)
app.slots_var.set("4"); app._build_cards(); app.update()
assert len(app.cards) == 4 and all(not c.full_detail for c in app.cards), "new cards must inherit the setting"
app._apply_sim_settings()                          # must never raise, with or without PyBullet
assert app.log.cget("wrap") == "word", "log lines must wrap in the visible pane"
assert app.cal_text.cget("wrap") == "word", "calibration output must wrap in the visible pane"
assert app.cloud_panel.detail.cget("wrap") == "word", "cloud detail output must wrap in the visible pane"
assert app.crop_cb.winfo_exists() and app.crop_cb.master.master is not None
app._set_state("running"); app._set_state("idle")
print("OK: tabs", tabs)
app.destroy()
