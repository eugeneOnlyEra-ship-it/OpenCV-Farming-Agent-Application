"""check_pod_markers.py -- test the ArUco pod localization WITHOUT PyBullet.

PyBullet is not needed: this rasterises the marker geometry (the same shapes pod_markers.add_pod_marker
builds) through the same view/projection matrices the camera agents use, then runs the real
detection + solvePnP code on those images for all 72 pods x 3 views.

  python dev/check_pod_markers.py

It checks the geometry/conventions (marker orientation, wall side, camera conventions, pose maths). It does
NOT check PyBullet's own renderer -- the first real PyBullet run still has to confirm detection there.
"""
import sys, math, numpy as np, cv2
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pybullet_sim import pod_markers as pm
from pybullet_sim.pod_registry import PODS, TROUGH_HALF_EXTENTS_M, ROW_SPACING_M
from pybullet_sim.multiview_localization import IMAGE_WIDTH as W, IMAGE_HEIGHT as H, CAMERA_FOV_DEG as FOV, CAMERA_NEAR as NEAR, CAMERA_FAR as FAR

def look_at(eye, target, up=(0,0,1)):
    eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
    f = target-eye; f/=np.linalg.norm(f); s=np.cross(f,up); s/=np.linalg.norm(s); u=np.cross(s,f)
    V = np.eye(4); V[0,:3]=s; V[1,:3]=u; V[2,:3]=-f; V[:3,3]=-V[:3,:3]@eye
    return V
def projection():
    f=1/math.tan(math.radians(FOV)/2); a=W/H
    return np.array([[f/a,0,0,0],[0,f,0,0],[0,0,(FAR+NEAR)/(NEAR-FAR),2*FAR*NEAR/(NEAR-FAR)],[0,0,-1,0]])
def quat_to_matrix(q):
    x,y,z,w=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def project(P, V, pts):
    h=np.c_[pts,np.ones(len(pts))]@(P@V).T; n=h[:,:3]/h[:,3:4]
    return np.c_[(n[:,0]+1)/2*W,(1-n[:,1])/2*H], h[:,3]

def marker_shapes(pod, marker_id, half_y):
    """The same shape list add_pod_marker feeds PyBullet, as (centre_local, half_extents, shade)."""
    bits=(np.asarray(cv2.aruco.generateImageMarker(pm.aruco_dictionary(),marker_id,pm.MARKER_CELLS,borderBits=1))>127).astype(np.uint8)
    cell=pm.MARKER_SIDE_M/pm.MARKER_CELLS; x0=y0=-pm.MARKER_SIDE_M/2
    shapes=[((0,0,-pm.PLATE_THICKNESS_M/2),(pm.PLATE_SIDE_M/2,pm.PLATE_SIDE_M/2,pm.PLATE_THICKNESS_M/2),215)]
    for r,c0,c1 in pm._black_runs(bits):
        w=(c1-c0+1)*cell
        shapes.append(((x0+c0*cell+w/2,y0+r*cell+cell/2,-pm.PLATE_THICKNESS_M-pm.CELL_RAISE_M/2),(w/2,cell/2,pm.CELL_RAISE_M/2),25))
    return shapes

def render(pod, marker_id, eye, V, P, half_y):
    q=pm._matrix_to_quaternion(pm.marker_rotation_world(pod)); R=quat_to_matrix(q)
    assert np.allclose(R, pm.marker_rotation_world(pod), atol=1e-9), "quaternion round-trip"
    s=pm._aisle_sign(pod)
    base=pm.marker_origin_world(pod,half_y)
    SS=int(__import__("os").environ.get("SS","1"))
    img=np.full((H*SS,W*SS),120,np.uint8)
    for c,h,shade in marker_shapes(pod,marker_id,half_y):
        zf=c[2]-h[2]                       # outward (viewer-side) face of the box
        corners=np.array([[c[0]+sx*h[0],c[1]+sy*h[1],zf] for sx,sy in ((-1,-1),(1,-1),(1,1),(-1,1))])
        world=base+corners@R.T
        # backface: the visible face normal is -z_marker
        if np.dot(-R[:,2], eye-world.mean(0))<=0: continue
        uv,wcl=project(P,V,world)
        if (wcl<=0).any(): continue
        cv2.fillConvexPoly(img,np.round(uv*SS-0.5).astype(np.int32),int(shade),lineType=cv2.LINE_8)
    if SS>1: img=cv2.resize(img,(W,H),interpolation=cv2.INTER_AREA)
    return np.dstack([img,img,img,np.full_like(img,255)])


def run(ss):
    os.environ["SS"] = str(ss)
    P = projection(); half_y = TROUGH_HALF_EXTENTS_M[1]
    valid_pods = seen = total = 0; errs = []
    for idx, pod in enumerate(PODS):
        sign = 1 if pod["side"] == "front" else -1
        row_y = pod["row"] * ROW_SPACING_M
        views = []
        for vi, plant in enumerate(pod["plants"]):
            eye = np.array([plant["x"], row_y + sign * 0.505, pod["position_m"]["z"] + TROUGH_HALF_EXTENTS_M[2] + 0.22])
            target = [pod["position_m"]["x"], pod["position_m"]["y"], pod["position_m"]["z"] + TROUGH_HALF_EXTENTS_M[2]]
            V = look_at(eye, target)
            views.append({"view_index": vi, "camera_position_world": eye.tolist(),
                          "view_matrix": V.flatten(order="F").tolist(),
                          "rgba": render(pod, pm.marker_id_for_pod(idx), eye, V, P, half_y)})
        pose = pm.estimate_pod_pose_from_markers(views, pod, pm.marker_id_for_pod(idx), TROUGH_HALF_EXTENTS_M)
        seen += pose["contributing_view_count"]; total += len(views)
        if pose["valid"]:
            valid_pods += 1
            nominal = np.array([pod["position_m"][k] for k in "xyz"])
            errs.append(np.linalg.norm(np.array(pose["center_world"]) - nominal) * 1000)
    return valid_pods, seen, total, np.array(errs)

if __name__ == "__main__":
    for ss, label in ((8, "accurate rendering (8x supersampled)"), (1, "harsh rendering (aliased, 1x)")):
        v, seen, total, e = run(ss)
        print(f"{label}: pods localized {v}/{len(PODS)}, views with marker read {seen}/{total}, "
              f"position error mm mean {e.mean():.1f} / max {e.max():.1f}")
        if ss == 8:
            assert v == len(PODS) and seen == total and e.max() < 5.0, "geometry/convention check failed"
        else:
            assert v >= 0.9 * len(PODS) and e.max() < 30.0, "marker too hard to read under aliased rendering"
    print("OK")
