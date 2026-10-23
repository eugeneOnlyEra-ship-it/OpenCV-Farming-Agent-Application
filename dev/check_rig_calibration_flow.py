"""check_rig_calibration_flow.py -- test the VISIBLE two-camera calibration without PyBullet.

Runs gantry_robot._g_calibrate_rig / _g_obtain_rig (the generator that holds the chessboard up in front of a
camera head's two lenses, photographs it with both, and calibrates with cv2.calibrateMultiviewExtended) against a
fake PyBullet whose getCameraImage renders the board in software. Checks the flow, the events, the GUI drawings
(A/B labels, baseline line, progress text) and the numbers. It does NOT check PyBullet's own renderer.

  python dev/check_rig_calibration_flow.py
"""
import os, sys, types, math
import numpy as np, cv2
from unittest.mock import MagicMock
sys.modules.setdefault('pybullet', MagicMock()); sys.modules.setdefault('pybullet_data', MagicMock())
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pybullet_sim import gantry_robot as gr, stereo_rig as sr, rig_calibration as rc
from dev import check_stereo_rig as cs
from dev.check_pod_markers import look_at, projection, W, H
SS=8
class FakeP:
    GEOM_BOX=3; JOINT_FIXED=4; ER_TINY_RENDERER=1; ER_NO_SEGMENTATION_MASK=2; ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX=3
    def __init__(s): s.board=None; s.texts=[]; s.lines=[]; s.removed=[]; s.n=0
    def computeProjectionMatrixFOV(s,**k): return [0]*16
    def computeViewMatrix(s,eye,tgt,up): return look_at(eye,tgt).flatten(order="F").tolist()
    def createVisualShape(s,*a,**k): return 1
    def createMultiBody(s,**k): s.n+=1; return 50
    def resetBasePositionAndOrientation(s,b,pos,q,**k): s.board=(np.array(pos),q)
    def getCameraImage(s,w,h,viewMatrix=None,**k):
        V=np.asarray(viewMatrix).reshape(4,4,order="F"); pos,q=s.board
        x,y,z,w_=q; R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w_),2*(x*z+y*w_)],[2*(x*y+z*w_),1-2*(x*x+z*z),2*(y*z-x*w_)],[2*(x*z-y*w_),2*(y*z+x*w_),1-2*(x*x+y*y)]])
        g=cs.render_board(pos,R,V,projection(),SS); rgba=np.dstack([g,g,g,np.full_like(g,255)])
        return (w,h,rgba.flatten().tolist(),None,None)
    def addUserDebugText(s,text,pos,**k): s.texts.append((text,k.get("replaceItemUniqueId",-1))); return 77
    def addUserDebugLine(s,a,b,**k): s.lines.append((a,b,k)); return 78
    def removeBody(s,b,**k): s.removed.append(b)
    def getDebugVisualizerCamera(s,**k):
        V=look_at([1,-2,1.5],[0,0,0.5]); P_=projection()
        return (1000,700,V.flatten(order="F").tolist(),P_.flatten(order="F").tolist())
    def loadTexture(s,path,**k): return 9
    def changeVisualShape(s,*a,**k): pass
    def changeTexture(s,*a,**k): pass
gr.p=FakeP()
robot=gr.GantryRobot.__new__(gr.GantryRobot)
robot._gui_enabled=True; robot.robot_id=1; robot.client=0; robot.live_view=None; robot.rigs={}
events=[]; robot._emit=lambda kind,pod=None,**k: events.append((kind,k))
agent=types.SimpleNamespace(side="front",toward=-1,stage_joints=(3,4),key="front_camera")
eye=[0.0,0.665,0.545]; target=[0.0,0.16,0.325]
gr.RIG_CALIBRATION_DWELL_S=0.01
sr.save_rig=lambda rig,path: None      # don't write into logs/ from a test
gen=robot._g_obtain_rig(agent,eye,target); steps=0; rig=None
try:
    while True: next(gen); steps+=1
except StopIteration as e: rig=e.value
t=rig["truth_comparison"]
print(f"visible calibration: {rig['frames_used']}/{gr.RIG_CALIBRATION_POSES} frames, rms {rig['rms_px']:.3f} px, baseline {rig['baseline_m']*1000:.2f} mm, trans err {t['translation_error_mm']:.2f} mm, rot err {t['rotation_error_deg']:.3f} deg")
print("board removed afterwards:",gr.p.removed==[50], "| yielded sim steps:",steps)
print("events:",[(k,{kk:vv for kk,vv in d.items() if kk in ('state','calibrated','mode','baseline_m')}) for k,d in events])
labels=[t for t,_ in gr.p.texts]
print("GUI texts: progress updates",sum('calibrating' in t for t in labels),"| labels",[t for t in labels if t in ('A','B')],"| banner",[t for t in labels if t.startswith('stereo pair')])
print("baseline line drawn:",len(gr.p.lines)==1)
assert rig["frames_used"]==gr.RIG_CALIBRATION_POSES and t["translation_error_mm"]<2.0 and t["rotation_error_deg"]<0.3
assert gr.p.removed==[50] and any(k=="stereo_rig" for k,_ in events)
# the other drawings must not raise
robot._draw_view_rays([0,0.6,0.5],[0.04,0.6,0.5],[0,0.16,0.3])
robot._draw_pose_overlay({"pod_id":"X","center_world":[0.0,0.16,0.28]},[0.0,0.161,0.28])

# --- the camera agent's FIRST STEP must run (a regression test: `import pybullet as p` further down inside
# _g_camera makes `p` a local name there, so using p at the top of that generator crashes every agent at once) ---
from pybullet_sim.pod_registry import PODS
def first_step(gui):
    r = gr.GantryRobot.__new__(gr.GantryRobot)
    r._gui_enabled = gui; r.client = 0; r.robot_id = 1; r.live_view = None; r.rigs = {}
    r._emit = lambda *a, **k: None
    r.agent = lambda side, role: types.SimpleNamespace(side=side, toward=-1, stage_joints=(3, 4), joint=5, key=f"{side}_{role}")
    def settle(targets): yield
    r._g_settle = settle
    g = r._g_camera("front", [PODS[0]], lambda pod: {}, {"assessed": {}})
    next(g)                                   # raises UnboundLocalError if p is used before the local import
    return r
r0 = first_step(False); r1 = first_step(True)
assert r0.live_view is not None and r0.live_view.mode == "off"
assert r1.live_view is not None and r1.live_view.mode == "pybullet" and len(r1.live_view.bodies) == 2
print("camera agent first step: OK (live view off / in-GUI screens)")
print("OK")
