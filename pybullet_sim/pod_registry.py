"""
pod_registry.py

Defines the pod farm layout as a ROWS x COLUMNS grid of *stations* --
each station is a single vertical metal pole with elongated
rectangular trough pods stacked at several heights, mirrored on the
pole's front (+ into its aisle) and back (- into its aisle) face. This
matches the layout sketched in Opencv_agent_pods.svg (one rectangular
trough shape, repeated at N layers, on 2 sides of a station), scaled
up to a full grid of stations.

Grid layout:
  - COLUMNS poles per row (X axis, meters) -- the direction the
    gantry's carriage travels along within a row.
  - ROWS of stations (Y axis, meters) -- each row sits in its own
    aisle; the gap between rows is sized to be an actual human-
    walkable aisle, not just clearance between pole centers.
  - LAYERS_PER_STATION trough heights per pole (Z axis, meters).
  - SIDES = ("front", "back") -> the two faces of each pole, offset a
    little either side of the row's centerline.

Spacing rationale (this is the part worth tuning for a real farm):
  - COLUMN_SPACING_M is pole-to-pole spacing *within* a row -- the
    gantry rides a rail along this direction, so it mainly needs
    clearance for the trough width (TROUGH_HALF_EXTENTS_M) plus a
    little margin for manual access to a single pole, not a full
    walking aisle.
  - AISLE_CLEAR_WIDTH_M is the actual open, walkable gap between one
    row's front-facing troughs and the next row's back-facing troughs
    -- i.e. the space a person (or the robot) would walk down between
    two rows. 1.0m sits in the standard range for a single-person
    working aisle (roughly 0.9-1.2m is typical guidance for
    warehouse/greenhouse pedestrian aisles); ROW_SPACING_M is then
    *derived* from that clear width plus the trough overhang on both
    sides, so the aisle stays exactly 1.0m clear regardless of how
    SIDE_OFFSET_M is tuned.

Design note (documented substitution, see handoff doc Section 6):
Physical pods/plants don't exist, so at each stop the simulated robot
loads a real dataset image pre-matched to a known ground-truth
condition instead of rendering organic matter. In this standalone
build that image pool is a small set of placeholder images generated
by generate_placeholder_images.py; swap PODS[i]["image_path"] to point
at real Roboflow/Mendeley crops once they're wired in -- no other code
needs to change, since the rest of the pipeline only cares about the
path.

WHERE THIS FILE FITS
    This is the DATA LAYER. It knows nothing about the robot or PyBullet: it only
    describes the farm (where every pole, trough and plant is, and what is "really"
    growing there). Everything else imports from it:
        gantry_robot.py   the scene + machine (positions, spacings, plants, docks, rails)
        main.py           the visiting order (stations_in_route_order / service_visit_order)
        verify_motion.py  the same, to audit the machine against the real layout
        agent_*_robot.py  the older iterations (legacy; see docs/DESIGN_LOG.md)
    Change the farm here (rows/columns, spacings, plants) and the rest follows -- but read
    docs/EXTENDING.md first: several machine dimensions are sized against these numbers.

COORDINATES  (metres, world frame)
    x  along a row, left -> right; column c's pole is at x = c * COLUMN_SPACING_M
    y  across rows; row r's beam is at y = r * ROW_SPACING_M. "front" faces +Y, "back" faces -Y
    z  up; layer l's trough centre is at z = BASE_HEIGHT_M + l * LAYER_SPACING_M
    A pole is at (col * 0.75, row * 1.32). Its 8 pods are 4 layers x {front, back}.

THE POD DICT  (one per trough; PODS is the list of all 72)
    pod_id        "R0C1-L2-B"  = row 0, column 1, layer 2, back side (unique, stable)
    index         0..71 build order
    row, col, layer, side       location; side is "front" or "back"
    crop_type     "cabbage" | "lettuce" | "mushroom"   (cycles through pods)
    position_m    {"x","y","z"} the trough CENTRE  (y = row_y +/- SIDE_OFFSET_M)
    plants        PLANTS_PER_POD dicts {"index","x","z","y_wall","y_front"}: each plant's
                  x/z, the y of the trough wall it grows from, and the y of its front tip
    ground_truth  {"growth_stage": str, "diseased": bool}  what is REALLY in the pod (only the
                  stub classifier and the log's agreement column use it)
    image_path    the stand-in "camera frame" for this pod (see generate_placeholder_images.py)

THE FIVE ORDERINGS  (do not mix them up)
    stations_in_route_order()  what the machine actually does: stations serpentine by row,
                               each with its front and back pods, TOP layer first   <- USED
    service_visit_order()      that, flattened: front then back per station         <- USED
    group_by_station(pods)     regroup a truncated flat list (main.py --pods N)     <- USED
    pod_visit_order()          older per-layer order for the single-agent iterations  (legacy)
    station_visit_order()      older rail-only stop list incl. docks                  (legacy)
"""

import itertools
import os

# --- what grows here -----------------------------------------------------------------------
CROPS = ["cabbage", "lettuce", "mushroom"]

# Growth stages per crop (must match the fixed Roboflow class lists). The classifier returns one
# of these strings and decision_agent.NEAR_HARVEST_STAGES refers to them BY NAME, so if you
# rename a stage, update both (and the placeholder images: python3 generate_placeholder_images.py).
GROWTH_STAGES = {
    "cabbage": ["S2_3", "S4_5", "S6_7", "S8", "S9"],
    "lettuce": ["Early Growth", "Leafy growth", "Head formation", "Harvest stage"],
    "mushroom": ["Juvenile", "Intermediate", "Harvest"],
}

# --- the grid. Changing ROWS/COLUMNS/LAYERS resizes the farm; rails, docks and the route follow.
#     (The machine geometry is NOT derived from these: see docs/EXTENDING.md "resize the farm".)
ROWS = 3                     # rows of stations, each in its own walkable aisle
COLUMNS = 3                  # poles per row
LAYERS_PER_STATION = 4       # stacked trough heights per pole (matches the SVG: 4 boxes)
SIDES = ("front", "back")    # both faces of each pole

COLUMN_SPACING_M = 0.75    # pole-to-pole along a row -- trough width (0.44m) + service clearance
LAYER_SPACING_M = 0.42     # vertical gap between trough heights
BASE_HEIGHT_M = 0.28       # height of the lowest trough off the ground
SIDE_OFFSET_M = 0.16       # how far a trough sticks out from the pole centerline (+/-)

# AISLE_CLEAR_WIDTH_M is measured between the trough CENTRE-line offsets (+/- SIDE_OFFSET_M). The
# trough walls stick out a further 0.08 m each, so the OPEN gap between facing trough WALLS is
# ROW_SPACING_M - 2*0.24 = 0.84 m (0.77 m between facing plant tips).
AISLE_CLEAR_WIDTH_M = 1.0  # walking gap between facing trough centre-lines (see note above)
ROW_SPACING_M = AISLE_CLEAR_WIDTH_M + 2 * SIDE_OFFSET_M  # pole-centerline-to-pole-centerline

TROUGH_HALF_EXTENTS_M = (0.22, 0.08, 0.045)  # (x, y, z) -- long + shallow + thin, per the SVG shape

# --- several plants across each pod ---
# Each elongated trough carries a row of plants along its length, set in planting
# ports in its aisle-facing wall (the wall the agents work from), so the camera
# and service agents have to address them one after another. A plant sticks out
# PLANT_PROTRUSION_M beyond the wall; its bounding box is a cube of side
# 2 * PLANT_HALF_SIZE_M. Offsets are along X from the trough's centre.
PLANT_X_OFFSETS_M = (-0.11, 0.0, 0.11)
PLANTS_PER_POD = len(PLANT_X_OFFSETS_M)
PLANT_PROTRUSION_M = 0.035
PLANT_HALF_SIZE_M = 0.035

# --- dock rails (added per the station/pole sketch) ---
# Two rail-only extensions, one beyond row 0 and one beyond the last row:
# where the agent is parked before work starts and where it parks again
# once finished. Pure rail -- no poles of their own, no stations mounted
# on them, cantilevered ("C" shaped) off the nearest row's end poles
# rather than needing new dedicated vertical supports.
#
# 0.75m (was 0.5m): the agents hang up to 0.505m to either side of a
# row's beam (see gantry_robot.CAMERA_REACH_M), so the dock bar has to
# sit far enough outside row 0 / the last row that no agent ever brushes
# it when servicing the outermost troughs. It
# also matches the sketch better -- bar "1" sits roughly 0.6x a full
# row-to-row gap above row "2", not half of it.
DOCK_SPACING_M = 0.75
TOP_DOCK_Y = -DOCK_SPACING_M
BOTTOM_DOCK_Y = (ROWS - 1) * ROW_SPACING_M + DOCK_SPACING_M
DOCK_PARK_X = ((COLUMNS - 1) * COLUMN_SPACING_M) / 2  # centered on the rail

# The horizontal rails ("horizontal poles") exactly as numbered in the
# sketch: 1 = top dock bar, 2/3/4 = the three rows of poles, and the
# unlabeled bottom bar (the end dock). The agent visits them in this
# order. y is the rail's world-Y coordinate.
HORIZONTAL_RAILS = (
    [("1", TOP_DOCK_Y)]
    + [(str(r + 2), r * ROW_SPACING_M) for r in range(ROWS)]
    + [("end", BOTTOM_DOCK_Y)]
)

# Where the per-pod stand-in camera images live (created by generate_placeholder_images.py).
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "sample_images")


def station_visit_order():
    """
    Station-level (not trough-level) visit order for the rail-only
    agent, per the confirmed sketch: park at the top dock, sweep each
    row's 3 poles left-to-right (alternating direction row to row so
    the agent doesn't waste travel snapping back to column 0 each
    time), then park at the bottom dock. No layers/sides/crops here --
    this is purely about which horizontal pole and which pole position
    on it, for reintroducing the arm/perception logic later without
    having to redo the rail-travel logic.

    Each stop is a dict: {label, x, y, is_dock, row, col}. row/col are
    None for dock stops.

    LEGACY: only agent_rail_demo.py (the rail-only iteration) uses this. The current
    machine uses stations_in_route_order() below.
    """
    stops = [{
        "label": "dock-top", "x": DOCK_PARK_X, "y": TOP_DOCK_Y,
        "is_dock": True, "row": None, "col": None,
    }]

    for row in range(ROWS):
        col_range = range(COLUMNS) if row % 2 == 0 else range(COLUMNS - 1, -1, -1)
        row_y = row * ROW_SPACING_M
        for col in col_range:
            stops.append({
                "label": f"R{row}C{col}", "x": col * COLUMN_SPACING_M, "y": row_y,
                "is_dock": False, "row": row, "col": col,
            })

    stops.append({
        "label": "dock-bottom", "x": DOCK_PARK_X, "y": BOTTOM_DOCK_Y,
        "is_dock": True, "row": None, "col": None,
    })
    return stops


def _ground_truth_cycle():
    """
    Deterministic, repeatable assignment of (growth_stage, disease) ground
    truth to pods, so a demo run is reproducible and includes a mix of
    healthy/diseased and early/late growth stages -- useful for showing
    the decision agent actually branching, not just logging one outcome
    over and over.
    """
    pattern = [
        (0, False),   # earliest stage, healthy
        (1, False),
        (2, True),    # mid stage, diseased -> should trigger "flag for treatment"
        (-1, False),  # last (harvest) stage, healthy -> should trigger tighter monitoring
    ]
    return itertools.cycle(pattern)


def build_pod_registry():
    """Returns a list of pod dicts describing the full farm layout:
    ROWS x COLUMNS poles x LAYERS_PER_STATION heights x 2 sides (see the POD DICT
    section in the module docstring for every key).

    Deterministic: crops cycle cabbage -> lettuce -> mushroom over pods, and each
    pole's ground truth follows the repeating pattern of _ground_truth_cycle(), so
    every run sees the same farm."""
    pods = []
    crop_cycle = itertools.cycle(CROPS)
    pod_index = 0

    for row in range(ROWS):
        row_y = row * ROW_SPACING_M

        for col in range(COLUMNS):
            station_x = col * COLUMN_SPACING_M
            gt_cycle = _ground_truth_cycle()

            for layer in range(LAYERS_PER_STATION):
                layer_z = BASE_HEIGHT_M + layer * LAYER_SPACING_M

                for side in SIDES:
                    crop = next(crop_cycle)
                    stages = GROWTH_STAGES[crop]
                    stage_offset, diseased = next(gt_cycle)
                    stage = stages[stage_offset if stage_offset >= 0 else len(stages) - 1]

                    side_y = row_y + (SIDE_OFFSET_M if side == "front" else -SIDE_OFFSET_M)
                    pod_id = f"R{row}C{col}-L{layer}-{side[0].upper()}"
                    image_name = (
                        f"{crop}_{stage}_{'diseased' if diseased else 'healthy'}.png"
                    ).replace(" ", "_")

                    out = 1 if side == "front" else -1        # +Y for the front face, -Y for the back
                    wall_y = side_y + out * TROUGH_HALF_EXTENTS_M[1]
                    plants = [
                        {"index": k, "x": station_x + dx, "z": layer_z,
                         "y_wall": wall_y, "y_front": wall_y + out * PLANT_PROTRUSION_M}
                        for k, dx in enumerate(PLANT_X_OFFSETS_M)
                    ]

                    pods.append({
                        "pod_id": pod_id,
                        "index": pod_index,
                        "row": row,
                        "col": col,
                        "layer": layer,
                        "side": side,
                        "crop_type": crop,
                        "position_m": {"x": station_x, "y": side_y, "z": layer_z},
                        "plants": plants,
                        "ground_truth": {"growth_stage": stage, "diseased": diseased},
                        "image_path": os.path.join(IMAGE_DIR, image_name),
                    })
                    pod_index += 1

    return pods


PODS = build_pod_registry()


def get_pod(pod_id):
    """The pod dict with this id, e.g. get_pod("R0C1-L2-B"); KeyError if unknown."""
    for p in PODS:
        if p["pod_id"] == pod_id:
            return p
    raise KeyError(f"Unknown pod_id: {pod_id}")


def station_key(pod):
    """(row, col) identifying the pole a pod belongs to."""
    return (pod["row"], pod["col"])


def pod_visit_order():
    """
    Boustrophedon (serpentine) traversal across the whole grid:
      - within a row, sweep columns left-to-right, then right-to-left
        on the next row (minimizes X travel between consecutive poles)
      - within each pole, alternate bottom-to-top / top-to-bottom on
        consecutive poles (minimizes Z travel, same trick as before)
      - front-then-back at each layer (a Y-axis move rather than a
        full carriage/lift re-position) before climbing to the next layer

    LEGACY: the single-agent iterations (agent_arm_demo.py etc.) use this. The current
    machine works both faces at once: see stations_in_route_order().
    """
    order = []
    station_counter = 0

    for row in range(ROWS):
        col_range = range(COLUMNS) if row % 2 == 0 else range(COLUMNS - 1, -1, -1)

        for col in col_range:
            station_pods = [p for p in PODS if p["row"] == row and p["col"] == col]
            station_pods.sort(key=lambda p: (p["layer"], 0 if p["side"] == "front" else 1))

            if station_counter % 2 == 1:
                layers_desc = sorted(set(p["layer"] for p in station_pods), reverse=True)
                station_pods.sort(key=lambda p: (
                    layers_desc.index(p["layer"]), 0 if p["side"] == "front" else 1
                ))

            order.extend(station_pods)
            station_counter += 1

    return order


def stations_in_route_order():
    """
    The machine's stops, in the order it makes them: row by row (rail 2,
    then 3, then 4), serpentine across the columns so it never doubles back.
    One entry per station (pole):

        {"row": r, "col": c,
         "front": [pods on the +Y face, TOP layer first ... layer 0],
         "back":  [pods on the -Y face, TOP layer first ... layer 0]}

    The machine carries a camera agent and a service agent on EACH side of
    the beam (see gantry_robot.py), so at a station the front and back faces
    are worked at the same time, each side top layer -> bottom layer.
    """
    stations = []
    for row in range(ROWS):
        col_range = range(COLUMNS) if row % 2 == 0 else range(COLUMNS - 1, -1, -1)
        for col in col_range:
            here = [p for p in PODS if p["row"] == row and p["col"] == col]
            stations.append({
                "row": row,
                "col": col,
                "front": sorted((p for p in here if p["side"] == "front"),
                                key=lambda p: p["layer"], reverse=True),
                "back": sorted((p for p in here if p["side"] == "back"),
                               key=lambda p: p["layer"], reverse=True),
            })
    return stations


def service_visit_order():
    """Flat list of every pod in the order stations are visited: each
    station's front face (top layer first) then its back face. Used for the
    `--pods N` smoke-test option and for building the scene; the agents
    themselves work a station's two faces in parallel (stations_in_route_order)."""
    return [p for st in stations_in_route_order() for p in st["front"] + st["back"]]


def group_by_station(pods):
    """Regroup a (possibly truncated) flat pod list into station dicts of the
    same shape as stations_in_route_order(). Consecutive pods on the same
    station are grouped; a station cut short by truncation simply has fewer pods."""
    stations = []
    for pod in pods:
        key = (pod["row"], pod["col"])
        if not stations or (stations[-1]["row"], stations[-1]["col"]) != key:
            stations.append({"row": pod["row"], "col": pod["col"], "front": [], "back": []})
        stations[-1][pod["side"]].append(pod)
    return stations


if __name__ == "__main__":
    print(f"{ROWS} rows x {COLUMNS} columns x {LAYERS_PER_STATION} layers x 2 sides "
          f"= {len(PODS)} pods")
    print(f"column spacing: {COLUMN_SPACING_M}m | row spacing: {ROW_SPACING_M}m "
          f"(aisle clear width: {AISLE_CLEAR_WIDTH_M}m)")
    for p in pod_visit_order()[:16]:
        print(p["pod_id"], p["crop_type"], p["ground_truth"], p["position_m"])
