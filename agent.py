"""
Decision layer for OpenCV-Farming-Agent.

Takes the dict classify_pod() returns and decides what the robot does next —
this is the piece that satisfies the Agentic Vision Award requirement that
visual evidence changes the next action, not just describes a fixed result.

Four possible actions:
    log_healthy               - nothing wrong, keep moving
    flag_for_harvest          - crop is at its harvest-ready growth stage
    flag_for_treatment        - disease detected with enough confidence to act on
    schedule_frequent_monitoring - reading was too uncertain to act on either way;
                                    re-check this pod sooner than its normal interval

Only the first three ("log and move on", "flag for treatment", "schedule
more frequent monitoring") were in the original proposal wording.
flag_for_harvest was added here because the growth-stage dataset writeup
specifically called out the harvest-stage class as the one that should
actually trigger a harvest decision — worth a distinct action rather than
folding it into log_healthy. Easy to remove if you'd rather keep it to
three.
"""

from perception import MODEL_REGISTRY

# Confidence floors — deliberately separate from the model's own detection
# threshold (perception.CONF_THRESHOLD). A detection can clear that floor
# and still be too shaky to act on here; these are the "is this reading
# trustworthy enough to change a real decision" thresholds.
DISEASE_ACTION_THRESHOLD = 0.50
GROWTH_ACTION_THRESHOLD = 0.40

# Per-action robot behavior, consumed by main.py so the *physical*
# response is visibly different depending on the decision -- this is
# what turns "we printed a label" into "the robot did something
# different", which is the actual Agentic Vision Award requirement.
# Values carried over from this project's original decision_agent.py
# (now superseded by this module), plus a value for flag_for_harvest,
# which decision_agent.py didn't have as a separate action.
ACTION_DWELL_SECONDS = {
    "log_healthy": 0.4,
    "schedule_frequent_monitoring": 0.9,
    "flag_for_harvest": 1.2,
    "flag_for_treatment": 1.6,
}


def decide_action(classification, disease_threshold=DISEASE_ACTION_THRESHOLD, growth_threshold=GROWTH_ACTION_THRESHOLD):
    """
    classification: the dict returned by perception.classify_pod().
    Returns {"action": str, "reason": str} plus the original classification
    under "based_on", so the caller has full traceability from action back
    to the visual evidence that produced it.
    """
    crop_type = classification["crop_type"]
    harvest_stage = MODEL_REGISTRY[crop_type]["growth"]["harvest_stage"]

    growth_stage = classification["growth_stage"]
    growth_conf = classification["growth_confidence"]
    disease_flag = classification["disease_flag"]
    disease_conf = classification["disease_confidence"]

    # 1. A confident disease read takes priority over everything else —
    #    treatment/human-review flags shouldn't wait behind a growth-stage check.
    if disease_flag and disease_conf >= disease_threshold:
        return {
            "action": "flag_for_treatment",
            "reason": f"{classification['disease_name']} detected at {disease_conf:.2f} confidence",
            "based_on": classification,
        }

    # 2. Any reading too shaky to act on — either model came back empty or
    #    under its own action threshold — gets re-checked sooner rather than
    #    having a low-confidence guess drive a real decision.
    growth_uncertain = growth_stage is None or growth_conf < growth_threshold
    disease_uncertain = disease_flag and disease_conf < disease_threshold
    if growth_uncertain or disease_uncertain:
        return {
            "action": "schedule_frequent_monitoring",
            "reason": (
                f"low-confidence read this visit (growth={growth_conf:.2f}, disease={disease_conf:.2f}) "
                f"— re-checking sooner rather than acting on it"
            ),
            "based_on": classification,
        }

    # 3. Confidently at harvest stage, no disease.
    if growth_stage == harvest_stage:
        return {
            "action": "flag_for_harvest",
            "reason": f"{crop_type} at {growth_stage} ({growth_conf:.2f} confidence)",
            "based_on": classification,
        }

    # 4. Confidently healthy and still growing — nothing to do but log it.
    return {
        "action": "log_healthy",
        "reason": f"{crop_type} at {growth_stage}, no disease ({growth_conf:.2f} / {disease_conf:.2f} confidence)",
        "based_on": classification,
    }
