#!/usr/bin/env python3
"""
tools/generate_gantry_urdf.py -- writes urdf/gantry.urdf

WHY THIS EXISTS
    The machine's URDF is ~400 lines of near-identical XML (four agents, two
    drive boxes, each with several links and joints). Editing it by hand is slow
    and error-prone, and its numbers must agree with the constants in
    gantry_robot.py. So the URDF is GENERATED from this script, and the
    dimensions that both files share are IMPORTED from gantry_robot.py (single
    source of truth). Change a shared dimension in gantry_robot.py and re-run
    this script; change something only the URDF cares about (a colour, a
    decorative vent) here.

USAGE (from the project root)
    python3 tools/generate_gantry_urdf.py            # (re)write urdf/gantry.urdf
    python3 tools/generate_gantry_urdf.py --check    # exit 1 if urdf/gantry.urdf is stale

    After regenerating, run  python3 verify_motion.py  -- it checks that the
    URDF and the constants agree and that nothing collides.

HOW THE URDF IS LAID OUT (the kinematic tree; every joint is prismatic)
    base_link                                   fixed at (0, 0, BASE_Z), no visual
    +- carriage_joint  (X)  carriage_link       machine position along a horizontal rail
       +- row_joint    (Y)  row_link            machine position along a vertical rail
       |                                        (chassis, portal posts, outrigger rails)
       +- box_front / box_back (Y, slides)      merged drive box, one per side of the beam
          +- hoist_<side>_camera  (-Z)          the rod's top end; positive value = down
          |  +- stage1_ (X) +- stage2_ (X)      two-stage telescoping traverse
          |                                     (carriage + camera head at the tip)
          +- hoist_<side>_service (-Z)
             +- stage1_ (X) +- stage2_ (X)      (carriage + tank/nozzle at the tip)
                +- applicator_<side>_service    contact rod + gold pad, slides toward the pod

    Joint names are looked up BY NAME in gantry_robot.py (never by index), so
    adding joints does not break the Python side. PyBullet's rule "link index ==
    the index of the joint that leads to it" is used by
    GantryRobot.agent_head_position_from_link().

CONVENTIONS
    * Everything is in metres, radians. Links are never rotated, so a link's
      local x / y / z are world x / y / z (front side = +Y, back side = -Y).
    * z values in each link are relative to that link's origin. The chassis
      link's origin is the CENTRE of the chassis plate; the vertical reference
      table in the file header lists the important heights.
    * No collision shapes anywhere: the scene is visual-only; clearance is
      checked numerically by verify_motion.py instead.
    * XML comments must not contain a double dash, so the header avoids them
      (a final regex pass also sanitises every comment).
"""

import os
import re
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))   # the project root, so the `pybullet_sim` package is importable
from pybullet_sim import gantry_robot as gr  # noqa: E402  (shared dimensions live there; it uses
                                             #   package-relative imports, so it can't be imported bare)

OUT_PATH = os.path.join(ROOT, "urdf", "gantry.urdf")

# --------------------------------------------------------------------------
# Dimensions shared with gantry_robot.py (imported: change them THERE)
# --------------------------------------------------------------------------
AGENT_X = gr.AGENT_X_OFFSET_M                 # rods hang at x = -AGENT_X (camera) / +AGENT_X (service)
BAY = gr.BAY_OFFSET_M                         # camera rod this far OUT of the box centre, service rod this far IN
BOX_REACH = gr.BOX_REACH_M                    # box centre's distance from the beam, engaged
BOX_STROKE = gr.BOX_STROKE_M                  # how far a box slides in
BOX_HALF_X = gr.BOX_HALF_X_M                  # lid half-width (the whole box is 2x this wide)
LID_HALF_Y = gr.BOX_HALF_Y_M                  # lid half-depth
BODY_HALF_Y = LID_HALF_Y - 0.005              # the body is 5 mm shallower than its lid
HOIST_ORIGIN_Z = gr.HOIST_ORIGIN_ABOVE_CHASSIS_M   # bottom face of the boxes, above the chassis-plate centre
HEAD_DROP = gr.HEAD_DROP_M                    # tool centre below the hoist origin
RAIL_DROP = gr.RAIL_DROP_M                    # traverse / carriage centre below the hoist origin
YOKE_Z = -gr.ROD_YOKE_INSET_M                 # the yoke block's z inside the hoist link
TOOL_Z = -(HEAD_DROP - RAIL_DROP)             # tool centre, relative to the carriage link
STAGE_TRAVEL = gr.STAGE_LIMIT_PER_STAGE_M     # each telescoping stage slides this far
BAR_LEN = 2 * gr.BAR_HALF_LENGTH_M            # sleeve and both bars are this long
BAR_BEHIND = gr.BAR_CENTRE_BEHIND_M           # stack centre sits this far BEHIND the rod
CARRIAGE = tuple(2 * h for h in gr.CARRIAGE_HALF_M)
CAMERA_RADIUS = gr.CAMERA_HALF_X_M
TWO_LENSES = gr.STEREO_RIG_ENABLED            # the head carries a stereo pair (stereo_rig.py) -> draw both lenses
LENS_BASELINE = gr.stereo_rig.BASELINE_M      # lens-to-lens distance
LENS_RADIUS = 0.010                           # lens glass on each 30 mm module
TANK_RADIUS = gr.SERVICE_TANK_RADIUS_M
TUBE_END = gr.SERVICE_NOZZLE_TUBE_END_M       # tank centre -> end of the nozzle tube, toward the pod
PAD_FRONT = gr.APPLICATOR_PAD_FRONT_M
ROD_BACK = gr.APPLICATOR_ROD_BACK_M
PAD_RADIUS = gr.APPLICATOR_PAD_RADIUS_M

# --------------------------------------------------------------------------
# URDF-only dimensions (nothing in Python depends on these)
# --------------------------------------------------------------------------
BOX_UPPER_LIMIT = 0.30                        # box joint limit; only needs to exceed BOX_STROKE
APPLICATOR_UPPER_LIMIT = 0.10                 # applicator joint limit; only needs to exceed APPLICATOR_STROKE_M
HOIST_UPPER_LIMIT = 2.6                       # deepest hoist extension (pod z = 0.28 needs ~2.2)

# Colours (r g b a). Nothing depends on them; they are how you tell the parts apart on screen.
MATERIALS = {
    "chassis_orange": (0.90, 0.55, 0.10, 1), "wheel_dark": (0.10, 0.10, 0.12, 1),
    "portal_grey": (0.60, 0.60, 0.64, 1), "crossrail_dark": (0.25, 0.25, 0.28, 1),
    "box_teal": (0.10, 0.45, 0.55, 1), "box_purple": (0.38, 0.17, 0.52, 1),
    "box_seam": (0.10, 0.10, 0.12, 1), "lid_grey": (0.72, 0.73, 0.76, 1),
    "panel_grey": (0.55, 0.57, 0.60, 1), "motor_dark": (0.20, 0.21, 0.24, 1),
    "cap_light": (0.85, 0.86, 0.88, 1), "vent_dark": (0.05, 0.05, 0.06, 1),
    "led_blue": (0.25, 0.65, 1.00, 1), "led_amber": (1.00, 0.70, 0.10, 1),
    "led_green": (0.20, 0.95, 0.35, 1), "hoist_blue": (0.15, 0.45, 0.75, 1),
    "guide_steel": (0.62, 0.64, 0.68, 1), "sleeve_dark": (0.30, 0.32, 0.36, 1),
    "bar_light": (0.80, 0.82, 0.86, 1), "carriage_orange": (0.92, 0.45, 0.10, 1),
    "camera_dark": (0.06, 0.06, 0.06, 1), "lens_glass": (0.20, 0.55, 0.95, 1),
    "lens_glass_b": (0.20, 0.85, 0.70, 1),     # (single-lens head only)
    "module_a": (0.15, 0.55, 1.00, 1),         # stereo pair: camera A's module (left): bright blue
    "module_b": (0.10, 0.95, 0.65, 1),         # camera B's module (right): bright green-teal
    "lens_dark": (0.02, 0.03, 0.08, 1),        # the lens glass on each module
    "tank_purple": (0.55, 0.20, 0.75, 1), "tank_band": (0.95, 0.95, 0.97, 1),
    "nozzle_grey": (0.35, 0.35, 0.38, 1), "pad_gold": (0.95, 0.75, 0.15, 1),
    "rod_steel": (0.78, 0.79, 0.82, 1),
}

HALF_PI = 1.5708      # rotates a cylinder's axis: rpy=(HALF_PI,0,0) turns Z into Y; (0,HALF_PI,0) turns Z into X


# --------------------------------------------------------------------------
# Tiny XML helpers: every <visual>/<link>/<joint> below is built from these so
# the numbers stay readable. Numbers are printed with %g (0.20 -> 0.2).
# --------------------------------------------------------------------------
def g(v):
    return f"{v:g}"


def vec(*v):
    return " ".join(g(x) for x in v)


def box(x, y, z):
    return f'<box size="{vec(x, y, z)}"/>'


def cylinder(radius, length):
    return f'<cylinder radius="{g(radius)}" length="{g(length)}"/>'


def sphere(radius):
    return f'<sphere radius="{g(radius)}"/>'


def visual(geometry, material, xyz=(0, 0, 0), rpy=(0, 0, 0)):
    """One <visual> line. `geometry` comes from box()/cylinder()/sphere()."""
    return (f'    <visual><origin xyz="{vec(*xyz)}" rpy="{vec(*rpy)}"/>'
            f'<geometry>{geometry}</geometry><material name="{material}"/></visual>\n')


def link(name, visuals, mass, inertia, comment=None):
    """A <link>. Inertia is one number used for the diagonal; nothing here is
    dynamically meaningful (all motion is position-controlled joints), it just
    has to be positive so PyBullet is happy."""
    out = f"\n  <!-- {comment} -->\n" if comment else "\n"
    out += f'  <link name="{name}">\n{visuals}'
    out += (f'    <inertial><mass value="{g(mass)}"/><inertia ixx="{g(inertia)}" ixy="0" ixz="0" '
            f'iyy="{g(inertia)}" iyz="0" izz="{g(inertia)}"/></inertial>\n  </link>\n')
    return out


def prismatic(name, parent, child, xyz, axis, lower, upper, effort, velocity, damping, friction):
    """A sliding <joint>. `axis` is the direction of POSITIVE joint values."""
    return (f'  <joint name="{name}" type="prismatic">\n'
            f'    <parent link="{parent}"/><child link="{child}"/>\n'
            f'    <origin xyz="{vec(*xyz)}" rpy="0 0 0"/><axis xyz="{vec(*axis)}"/>\n'
            f'    <limit lower="{g(lower)}" upper="{g(upper)}" effort="{g(effort)}" velocity="{g(velocity)}"/>\n'
            f'    <dynamics damping="{g(damping)}" friction="{g(friction)}"/>\n'
            f'  </joint>\n')


# --------------------------------------------------------------------------
# The file header: a long comment that documents the machine for whoever opens
# the URDF. Numbers in it are formatted from the constants above so it cannot
# drift out of date.
# --------------------------------------------------------------------------
def header():
    return f'''<?xml version="1.0"?>
<!--
  gantry.urdf  -  GENERATED by tools/generate_gantry_urdf.py (edit the generator, not this file)

  Rail-riding machine: two sliding drive boxes, four agents, each with a traverse head.

  One chassis rides on top of the rail network. Two outrigger rails stand
  above it and reach out over the aisle on both sides of the beam. A single
  DRIVE BOX rides on those rails on each side of the beam and houses the winch
  drives + electronics for BOTH agents on that side (one shared enclosure is
  cheaper than two): a two-tone body (teal camera bay, purple service bay), a
  common lid carrying one gear-motor + encoder per bay, vents and a status LED
  per bay on the outward face, and a green power LED on a shared panel.

       FRONT side (+Y)   camera agent   (x - {g(AGENT_X)}, OUTER bay)   service agent (x + {g(AGENT_X)}, INNER bay)
       BACK  side (-Y)   the same, mirrored

  The two bays sit at different depths: the camera's rod hangs {g(BAY)}m further
  out than the box centre, the service agent's {g(BAY)}m further in. That
  staggering is what lets the two heads sweep along the pod past each other's
  rods without touching.

  The drive boxes SLIDE along the outriggers (prismatic joints box_front /
  box_back, value = inward travel from the operating position):
    engaged     value 0      box centre {g(BOX_REACH)}m either side of the beam - operating
    disengaged  value {g(BOX_STROKE)}  boxes slid in toward each other, {g(BOX_REACH - BOX_STROKE)}m either side -
                             the compact configuration used while the machine
                             changes horizontal pole and while parked on a dock

  Every agent is a hoist (prismatic joint, positive = down) mounted on its
  box's underside. The long rod is drawn at runtime (gantry_robot.py); its top
  end comes out of the box's bottom face, its bottom end into a YOKE.

  TRAVERSE HEAD: each rod ends in a yoke carrying a TELESCOPING traverse: a short
  sleeve sitting BEHIND the rod (on the side away from the pod's middle) and two
  nested bars (stage1_*, stage2_*, {g(STAGE_TRAVEL)}m each) that slide out of it along X, with
  a carriage at the tip. The tool hangs from the carriage, so the head can move
  ALONG the pod, from plant to plant, without moving the rod. Retracted, the whole
  thing is compact and stays on the far side of its own rod, so it never crosses
  the outer vertical rail's line while the hoist descends past it; it only
  extends once the head is down at the pod:
    camera agent   dark cylinder with a blue lens, facing the trough
    service agent  purple tank + nozzle tube + a slim CONTACT APPLICATOR that
                   extends (its own prismatic joint, applicator_*) until the
                   gold pad touches a plant, holds, then retracts

  Kinematic chain:
    base_link -> carriage_joint (X) -> carriage_link          (no visual)
              -> row_joint      (Y) -> row_link               (chassis, portal, outriggers)
                 -> box_front (Y, slides) -> front_box_link   (merged drive box)
                       -> hoist_front_camera  (-Z) -> front_camera_link  (yoke + sleeve)
                             -> stage1_front_camera (X) -> stage2_front_camera (X) -> front_camera_stage_link (carriage + camera)
                       -> hoist_front_service (-Z) -> front_service_link (yoke + sleeve)
                             -> stage1_front_service (X) -> stage2_front_service (X) -> front_service_stage_link (carriage + tank)
                                   -> applicator_front_service -> front_service_applicator_link
                 -> box_back ... (mirror)

  Drive rules (enforced in gantry_robot.py, not here): the chassis only moves
  along ONE axis at a time and only along a rail; it changes horizontal pole
  (moves in Y) only with the boxes slid in; a box only slides with every hoist
  stowed, every head centred and the machine standing still; hoists only
  deploy with the boxes out.

  Vertical reference, relative to the chassis-plate centre (which sits 0.09m
  above the rail's centreline; beam half-thickness is 0.03):
       +0.289  top of the drive-box motors / encoder caps
       +0.247  top of the drive-box lid
       +0.239  top of the outrigger rails = top of the drive-box body
       +0.215  underside of the outrigger rails
       +{g(HOIST_ORIGIN_Z)}  hoist joint origin = bottom of each drive box (rods come out here)
       +{g(HOIST_ORIGIN_Z - RAIL_DROP)}  telescoping traverse centre (hoist origin - {g(RAIL_DROP)})
       +{g(HOIST_ORIGIN_Z - HEAD_DROP)}  tool-head centre (hoist origin - {g(HEAD_DROP)}) when the hoist is stowed
       +0.015  top of the chassis plate
        0.000  chassis plate centre  (base_link is placed here)
       -0.060  wheel bottoms = the rail's top face
  Change any of these together with the matching constants in
  gantry_robot.py - verify_motion.py will tell you if something now collides.
-->
<robot name="pod_farm_gantry">

'''


def materials():
    out = ""
    for name, rgba in MATERIALS.items():
        out += f'  <material name="{name}"><color rgba="{vec(*rgba)}"/></material>\n'
    return out


# --------------------------------------------------------------------------
# The fixed base and the two sliding axes that move the whole machine
# --------------------------------------------------------------------------
def base_and_axes():
    out = '\n  <!-- fixed anchor: deliberately NO visual, the machine travels the whole rail network -->\n'
    out += ('  <link name="base_link">\n'
            '    <inertial><mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>\n'
            '  </link>\n')

    # X axis: driving along a horizontal rail. An empty link -- it only carries the next joint.
    out += link("carriage_link", "", 1.0, 0.01,
                comment='X axis: travel along a horizontal rail ("horizontal pole")')
    out += prismatic("carriage_joint", "base_link", "carriage_link", (0, 0, 0), (1, 0, 0),
                     -0.5, 2.5, 250, 0.6, 5.0, 1.0)

    # Y axis: driving along an outer vertical rail. The visible chassis hangs on this link.
    v = "    <!-- chassis plate -->\n"
    v += visual(box(0.26, 0.20, 0.03), "chassis_orange")
    v += "    <!-- wheels for travel along X (axle along Y) -->\n"
    v += visual(cylinder(0.022, 0.024), "wheel_dark", (0.05, 0, -0.038), (HALF_PI, 0, 0))
    v += visual(cylinder(0.022, 0.024), "wheel_dark", (-0.05, 0, -0.038), (HALF_PI, 0, 0))
    v += "    <!-- wheels for travel along Y (axle along X), used on the outer rails -->\n"
    v += visual(cylinder(0.022, 0.024), "wheel_dark", (0, 0.05, -0.038), (0, HALF_PI, 0))
    v += visual(cylinder(0.022, 0.024), "wheel_dark", (0, -0.05, -0.038), (0, HALF_PI, 0))
    v += "    <!-- portal posts -->\n"
    v += visual(box(0.024, 0.06, 0.20), "portal_grey", (AGENT_X, 0, 0.115))
    v += visual(box(0.024, 0.06, 0.20), "portal_grey", (-AGENT_X, 0, 0.115))
    v += "    <!-- outrigger rails: run along Y; the drive boxes slide on them -->\n"
    v += visual(box(0.04, 1.10, 0.024), "crossrail_dark", (AGENT_X, 0, 0.227))
    v += visual(box(0.04, 1.10, 0.024), "crossrail_dark", (-AGENT_X, 0, 0.227))
    out += link("row_link", v, 2.0, 0.02,
                comment="Y axis: travel along an outer vertical rail (row to row, and out to the dock bars).\n"
                        "       The visible machine body is attached to this link.")
    out += prismatic("row_joint", "carriage_link", "row_link", (0, 0, 0), (0, 1, 0),
                     -1.5, 4.0, 250, 0.6, 5.0, 1.0)
    return out


# --------------------------------------------------------------------------
# One merged drive box (front or back). Its link frame sits at the box centre
# in x/y (z as the chassis frame), so joint value 0 = the engaged position.
# --------------------------------------------------------------------------
def drive_box(side):
    sy = 1 if side == "front" else -1          # +1 front (+Y), -1 back (-Y): "outward" is +sy
    cam_y, svc_y = sy * BAY, -sy * BAY         # camera bay OUTER, service bay INNER (box-local y)
    out_face = sy * BODY_HALF_Y                # y of the outward face of the body

    v = "    <!-- two-tone body: teal camera bay (-x half), purple service bay (+x half), dark seam between -->\n"
    half_width = (2 * BOX_HALF_X - 0.01) / 2                # each colour half is (0.315 - 0.01) / 2 = 0.1525 wide
    v += visual(box(half_width, 2 * BODY_HALF_Y, 0.084), "box_teal", (-half_width / 2, 0, 0.197))
    v += visual(box(half_width, 2 * BODY_HALF_Y, 0.084), "box_purple", (half_width / 2, 0, 0.197))
    v += visual(box(0.004, 0.0022, 0.084), "box_seam", (0, out_face * 1.0005, 0.197))
    v += "    <!-- common lid -->\n"
    v += visual(box(2 * BOX_HALF_X, 2 * LID_HALF_Y, 0.008), "lid_grey", (0, 0, 0.243))
    v += "    <!-- one gear-motor + encoder cap per bay, over each rod -->\n"
    for x, y in ((-AGENT_X, cam_y), (AGENT_X, svc_y)):
        v += visual(cylinder(0.02, 0.03), "motor_dark", (x, y, 0.262))
        v += visual(cylinder(0.012, 0.012), "cap_light", (x, y, 0.283))
    for x in (-AGENT_X, AGENT_X):              # cooling vents, one group per bay, outward face
        for z in (0.170, 0.181, 0.192):
            v += visual(box(0.05, 0.003, 0.006), "vent_dark", (x, out_face + sy * 0.001, z))
    v += visual(sphere(0.007), "led_blue", (-AGENT_X, out_face + sy * 0.002, 0.217))     # camera bay LED
    v += visual(sphere(0.007), "led_amber", (AGENT_X, out_face + sy * 0.002, 0.217))     # service bay LED
    v += visual(box(0.06, 0.0024, 0.05), "panel_grey", (0, out_face + sy * 0.0012, 0.19))  # shared panel
    v += visual(sphere(0.0055), "led_green", (0, out_face + sy * 0.0034, 0.205))           # power LED

    out = link(f"{side}_box_link", v, 1.0, 0.005,
               comment=f"MERGED DRIVE BOX, {side.upper()} side: winch drives + electronics for the camera agent "
                       f"(x-{g(AGENT_X)}, outer bay)\n       and the service agent (x+{g(AGENT_X)}, inner bay) in one "
                       "enclosure. Slides along the outriggers.")
    out += prismatic(f"box_{side}", "row_link", f"{side}_box_link", (0, sy * BOX_REACH, 0), (0, -sy, 0),
                     0.0, BOX_UPPER_LIMIT, 200, 0.35, 5.0, 1.0)
    return out


# --------------------------------------------------------------------------
# The two tool heads (visuals only; they hang from the carriage link)
# --------------------------------------------------------------------------
def camera_head(toward):
    """Horizontal dark cylinder with a blue lens on the end that faces the pod.
    `toward` = -1 for the front side (pod is at -Y from the agent), +1 for back."""
    body = visual(cylinder(CAMERA_RADIUS, 0.07), "camera_dark", (0, 0, TOOL_Z), (HALF_PI, 0, 0))
    if not TWO_LENSES:
        return ("    <!-- CAMERA head: horizontal dark cylinder, blue lens facing the trough -->\n"
                + body + visual(cylinder(0.02, 0.012), "lens_glass", (0, toward * 0.041, TOOL_Z), (HALF_PI, 0, 0)))
    # Stereo pair. stereo_rig.rig_cameras puts camera A half a baseline to the LEFT of the head centre and B to
    # the RIGHT, as seen looking at the pod; "right" is (toward, 0, 0) in world x, so A is at -toward*b/2.
    # (A = blue, B = teal.)
    xa, xb = -toward * LENS_BASELINE / 2, toward * LENS_BASELINE / 2
    # Each camera is a bright 30 mm module on a dark bracket bar at the head's end, with a dark lens on its face:
    # big enough to see in the GUI. Purely decorative (visual only): the collision volumes still come from
    # CAMERA_HALF_X_M etc.; the modules stick out ~5 mm past the 60 mm head, which nothing else is close to.
    bar_y = toward * 0.0405
    mod_y = toward * (0.0465 + 0.009)
    lens_y = toward * (0.0645 + 0.002)
    return ("    <!-- CAMERA head: dark cylinder + bracket bar carrying a stereo pair of camera modules "
            "(A blue = left, B green-teal = right) -->\n"
            + body
            + visual(box(0.084, 0.012, 0.016), "camera_dark", (0, bar_y, TOOL_Z))
            + visual(box(0.030, 0.018, 0.030), "module_a", (xa, mod_y, TOOL_Z))
            + visual(box(0.030, 0.018, 0.030), "module_b", (xb, mod_y, TOOL_Z))
            + visual(cylinder(LENS_RADIUS, 0.004), "lens_dark", (xa, lens_y, TOOL_Z), (HALF_PI, 0, 0))
            + visual(cylinder(LENS_RADIUS, 0.004), "lens_dark", (xb, lens_y, TOOL_Z), (HALF_PI, 0, 0)))


def service_head(toward):
    """Upright purple tank with a white band and a nozzle tube facing the pod.
    The contact applicator (its own link) slides out of the tube's end."""
    return ("    <!-- SERVICE head: upright purple tank with a white band, and a nozzle tube facing the trough.\n"
            "         The contact applicator (next link) slides out of the tube's end. -->\n"
            + visual(cylinder(TANK_RADIUS, 0.06), "tank_purple", (0, 0, TOOL_Z))
            + visual(cylinder(TANK_RADIUS + 0.0015, 0.008), "tank_band", (0, 0, TOOL_Z))
            + visual(cylinder(0.008, 0.035), "nozzle_grey", (0, toward * 0.0495, TOOL_Z), (HALF_PI, 0, 0)))


# --------------------------------------------------------------------------
# One agent: hoist link (yoke + sleeve) -> stage1 bar -> stage2 bar + carriage
# + tool -> (service only) applicator
# --------------------------------------------------------------------------
def agent(side, role):
    sy = 1 if side == "front" else -1
    toward = -sy                                # direction from this agent toward the pod it serves
    name = f"{side}_{role}"
    is_camera = role == "camera"
    x = -AGENT_X if is_camera else AGENT_X      # rod x, relative to the machine centre
    local_y = sy * BAY if is_camera else -sy * BAY    # rod y, relative to the box centre
    ext_dir = 1 if is_camera else -1            # the traverse extends toward +x (camera) / -x (service)
    lo, hi = (0.0, STAGE_TRAVEL) if ext_dir > 0 else (-STAGE_TRAVEL, 0.0)
    behind = -ext_dir * BAR_BEHIND              # stack centre: BEHIND the rod, away from the outer rail's line

    # hoist link: the stub the rod plugs into, the yoke block, and the fixed sleeve of the telescope
    v = visual(cylinder(0.02, 0.03), "hoist_blue", (0, 0, -0.015))
    v += visual(box(0.05, 0.04, 0.012), "hoist_blue", (0, 0, YOKE_Z))
    v += visual(box(BAR_LEN, 0.03, 0.026), "sleeve_dark", (behind, 0, -RAIL_DROP))
    out = link(f"{name}_link", v, 0.4, 0.002,
               comment=f"{side.upper()} {role.upper()} agent: hoist (hangs from the {side} drive box) -> "
                       "yoke + sleeve of the telescoping traverse")
    out += prismatic(f"hoist_{name}", f"{side}_box_link", f"{name}_link", (x, local_y, HOIST_ORIGIN_Z),
                     (0, 0, -1), 0.0, HOIST_UPPER_LIMIT, 200, 0.6, 5.0, 1.0)

    # first telescoping stage: a bar sliding out of the sleeve
    out += link(f"{name}_bar1_link", visual(box(BAR_LEN, 0.022, 0.02), "guide_steel", (behind, 0, 0)),
                0.15, 0.0005, comment="telescoping traverse, first stage: a bar sliding out of the sleeve")
    out += prismatic(f"stage1_{name}", f"{name}_link", f"{name}_bar1_link", (0, 0, -RAIL_DROP), (1, 0, 0),
                     lo, hi, 60, 0.25, 3.0, 0.5)

    # second stage: a thinner bar sliding out of the first, with the carriage and the TOOL at its tip
    v = visual(box(BAR_LEN, 0.016, 0.014), "bar_light", (behind, 0, 0))
    v += visual(box(*CARRIAGE), "carriage_orange")
    v += camera_head(toward) if is_camera else service_head(toward)
    out += link(f"{name}_stage_link", v, 0.3, 0.001,
                comment="second stage: a thinner bar sliding out of the first, with the carriage and the tool at its tip")
    out += prismatic(f"stage2_{name}", f"{name}_bar1_link", f"{name}_stage_link", (0, 0, 0), (1, 0, 0),
                     lo, hi, 60, 0.25, 3.0, 0.5)

    if not is_camera:
        # the contact applicator: a slim rod sliding out of the nozzle tube; its origin is the tube's end
        rod_len = ROD_BACK + 0.005                     # from ROD_BACK behind the tube end to 5 mm beyond it
        v = visual(cylinder(0.006, rod_len), "rod_steel", (0, toward * -(ROD_BACK - rod_len / 2), 0), (HALF_PI, 0, 0))
        v += visual(cylinder(PAD_RADIUS, 0.008), "pad_gold", (0, toward * (PAD_FRONT - 0.004), 0), (HALF_PI, 0, 0))
        out += link(f"{name}_applicator_link", v, 0.1, 0.0005,
                    comment="contact applicator: a slim rod that slides out of the nozzle tube until its gold pad touches\n"
                            f"       a plant. Link origin = the tube's end ({g(TUBE_END)} from the tank centre, toward the pod).")
        out += prismatic(f"applicator_{name}", f"{name}_stage_link", f"{name}_applicator_link",
                         (0, toward * TUBE_END, TOOL_Z), (0, toward, 0),
                         0.0, APPLICATOR_UPPER_LIMIT, 40, 0.25, 2.0, 0.2)
    return out


# --------------------------------------------------------------------------
# Assemble the file
# --------------------------------------------------------------------------
def build():
    body = header() + materials() + base_and_axes()
    for side in ("front", "back"):
        body += drive_box(side)
        body += agent(side, "camera")
        body += agent(side, "service")
    body += "\n</robot>\n"
    # XML forbids a double dash inside a comment: collapse any that slipped in
    return re.sub(r"<!--(.*?)-->", lambda m: "<!--" + m.group(1).replace("--", "-") + "-->", body, flags=re.S)


def canonical(xml_text):
    """A comparable form of a URDF: element tags and attributes with numbers
    rounded, comments dropped -- so two files that describe the same robot
    compare equal even if they were formatted differently."""
    def norm(value):
        try:
            return f"{float(value):.6g}"
        except ValueError:
            return value

    def walk(e):
        return (e.tag, tuple(sorted((k, " ".join(norm(t) for t in v.split())) for k, v in e.attrib.items())),
                tuple(walk(c) for c in e))
    return walk(ET.fromstring(xml_text))


def main():
    text = build()
    if "--check" in sys.argv:
        if not os.path.exists(OUT_PATH):
            print("urdf/gantry.urdf is missing"); sys.exit(1)
        if canonical(open(OUT_PATH).read()) != canonical(text):
            print("urdf/gantry.urdf is STALE: it differs from what tools/generate_gantry_urdf.py produces.")
            print("Run  python3 tools/generate_gantry_urdf.py  to regenerate it.")
            sys.exit(1)
        print("urdf/gantry.urdf matches the generator.")
        return
    with open(OUT_PATH, "w") as f:
        f.write(text)
    print(f"wrote {OUT_PATH} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
