"""
llm_planner.py  --  an LLM-in-the-loop planner with hard guardrails, runnable OFFLINE.

THE IDEA
    perception (OpenCV 5 + ONNX)  ->  evidence (numbers)  ->  LLM proposes a task
         ->  GUARDRAILS (plain code) accept / reject  ->  final task  ->  robot order

    The LLM never touches the robots and never writes robot steps. It only
    answers one question -- "given this evidence and this pod's history, which
    task should happen next?" -- as strict JSON. Ordinary code then decides
    whether that answer is allowed. If the LLM is slow, offline, confused, or
    tricked, the system simply keeps the rule-based decision.

THE BACKENDS (a backend is just: a function (system_prompt, evidence_dict, image_path) -> text)
    ScriptedBackend    NOT an LLM. A tiny hand-written stand-in so you can run and
                       test the whole pipeline with nothing installed.
    OllamaBackend      A REAL LLM running on your own machine through Ollama
                       (http://localhost:11434). Works with no internet and no AWS.
    (later) Bedrock    Same shape: call the Converse API, return the text. Nothing
                       else in this file changes.

THE GUARDRAILS (resolve(); the LLM cannot change any of these)
    1. Output must be strict JSON in the schema, else it is discarded.
    2. The LLM may only AGREE with the rules or ask to RECAPTURE (take a better look).
       It can never create a TREAT or HARVEST the rules did not find, and can never
       cancel one (no "this is fine, ignore it").
    3. It may not soften a case with strong evidence (>=3 boxes at >=0.85 confidence).
    4. It may not stall a pod: after one LLM-requested recapture, the next visit follows the rules.
    5. Approval, dispatch and steps stay with worker_planner.py (humans approve treatment).
"""

import base64
import json
import re
import urllib.request
from dataclasses import dataclass, field

TASKS = ("TREAT", "HARVEST", "RECAPTURE", "NONE")
URGENCY = ("low", "normal", "high")
RULE_TASK = {"log_healthy": "NONE", "schedule_frequent_monitoring": "RECAPTURE",
             "flag_for_harvest": "HARVEST", "flag_for_treatment": "TREAT"}
STRONG_CONF, STRONG_BOXES = 0.85, 3

SYSTEM_PROMPT = """You are the planning brain of a vertical-farm robot system. A computer-vision model has \
already analysed a photo of one growing pod and a simple rule table has proposed a task. You see the numbers it \
produced and this pod's recent history. Decide which task should happen next.

Tasks you may choose: TREAT (a disease needs treatment), HARVEST (crop is ready), RECAPTURE (the reading is \
uncertain - take a better photo before acting), NONE (nothing to do).

Be conservative. Prefer RECAPTURE when evidence is weak or contradictory (low confidence, very few boxes, a \
reading that flips between visits). Choose the rule table's task when the evidence is clear.

Reply with ONLY one JSON object, no other text:
{"task": "TREAT|HARVEST|RECAPTURE|NONE", "reason": "<= 200 chars, cite the numbers", \
"urgency": "low|normal|high", "notes_for_human": ["<= 3 short strings"]}"""


# ---------------------------------------------------------------------------
# Evidence: what the LLM is allowed to see
# ---------------------------------------------------------------------------
def build_evidence(pod, classification, decision, history, step, n_steps, harvest_stage):
    c = classification
    disease_boxes = [d for d in c["disease_detections"] if c["disease_flag"] and d["class_name"] == c["disease_name"]]
    return {
        "pod": {"id": pod["pod_id"], "crop": pod["crop_type"], "side": pod["side"], "visit": step + 1, "of": n_steps},
        "growth": {"stage": c["growth_stage"], "confidence": round(float(c["growth_confidence"]), 2),
                   "harvest_stage": harvest_stage, "n_boxes": len(c["growth_detections"])},
        "disease": {"flagged": bool(c["disease_flag"]), "name": c["disease_name"],
                    "confidence": round(float(c["disease_confidence"]), 2), "n_boxes": len(disease_boxes),
                    "n_all_boxes": len(c["disease_detections"])},
        "rule_table": {"action": decision["action"], "task": RULE_TASK[decision["action"]],
                       "reason": decision["reason"]},
        "previous_visits": history[-3:],
        "allowed_tasks": list(TASKS),
    }


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class ScriptedBackend:
    """NOT AN LLM. Hand-written stand-in with the same interface, so the pipeline and the guardrails can be
    exercised with nothing installed. It mimics one plausible LLM habit: 'weak evidence -> take a better look'."""
    name = "scripted stand-in (not an LLM)"

    def __call__(self, system, ev, image_path=None):
        rule, d, g = ev["rule_table"]["task"], ev["disease"], ev["growth"]
        task, why, urg, notes = rule, "Evidence is clear; rule table's choice stands.", "normal", []
        if rule == "TREAT" and d["confidence"] < 0.60 and d["n_boxes"] <= 2:
            task, urg = "RECAPTURE", "low"
            why = f"Only {d['n_boxes']} box(es) at {d['confidence']}: weak for treatment, get a closer look first."
        elif rule == "HARVEST" and g["confidence"] < 0.60:
            task = "RECAPTURE"
            why = f"Harvest stage read at only {g['confidence']}: confirm before cutting."
        elif rule == "TREAT" and any(p.get("disease") == d["name"] for p in ev["previous_visits"]):
            urg, notes = "high", [f"{d['name']} seen on a previous visit too - spreading risk."]
            why = "Same disease on consecutive visits: treat soon."
        return json.dumps({"task": task, "reason": why, "urgency": urg, "notes_for_human": notes})


class OllamaBackend:
    """A REAL local LLM via Ollama. Offline, free, private. Install Ollama, `ollama pull <model>`, keep it running.
    Text-only models work (the evidence is numbers). Pass use_image=True with a vision model to also send the photo."""

    def __init__(self, model="llama3.2:3b", url="http://localhost:11434", use_image=False, timeout=180):
        self.model, self.url, self.use_image, self.timeout = model, url.rstrip("/"), use_image, timeout
        self.name = f"ollama:{model}" + (" +image" if use_image else "")

    def __call__(self, system, ev, image_path=None):
        msg = {"role": "user", "content": json.dumps(ev)}
        if self.use_image and image_path:
            with open(image_path, "rb") as f:
                msg["images"] = [base64.b64encode(f.read()).decode()]
        body = {"model": self.model, "stream": False, "format": "json", "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, msg]}
        req = urllib.request.Request(self.url + "/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)["message"]["content"]


# ---------------------------------------------------------------------------
# Strict parsing + guardrails
# ---------------------------------------------------------------------------
@dataclass
class Proposal:
    task: str
    reason: str
    urgency: str
    notes: list = field(default_factory=list)
    ignored_keys: list = field(default_factory=list)


def _clean(s, n):
    return re.sub(r"[\x00-\x1f]+", " ", str(s)).strip()[:n]


def parse_proposal(text):
    """Strict. Returns Proposal or None. Unknown keys (e.g. an LLM trying to add "approve": true) are ignored
    and recorded, never obeyed."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(d, dict) or d.get("task") not in TASKS or d.get("urgency", "normal") not in URGENCY:
        return None
    notes = d.get("notes_for_human", [])
    notes = [_clean(n, 160) for n in notes[:3]] if isinstance(notes, list) else []
    known = {"task", "reason", "urgency", "notes_for_human"}
    return Proposal(d["task"], _clean(d.get("reason", ""), 200), d.get("urgency", "normal"), notes,
                    sorted(set(d) - known))


@dataclass
class Resolution:
    final_task: str
    rule_task: str
    verdict: str            # agreed | downgraded | rejected | fallback
    why: str
    proposal: object = None
    source: str = ""


def resolve(rule_task, evidence, proposal, prior_downgrade=False):
    """The guardrails. Pure function: easy to read, easy to test, impossible for the LLM to talk its way around."""
    if proposal is None:
        return Resolution(rule_task, rule_task, "fallback", "LLM unavailable or its reply was invalid: rules kept")
    t, d = proposal.task, evidence["disease"]
    if t == rule_task:
        return Resolution(rule_task, rule_task, "agreed", "LLM agrees with the rule table", proposal)
    if t != "RECAPTURE":
        return Resolution(rule_task, rule_task, "rejected",
                          f"LLM wanted {t} but may only agree or ask for a RECAPTURE: rules kept", proposal)
    # t == RECAPTURE and rules said something else
    if rule_task in ("TREAT", "HARVEST") and prior_downgrade:
        return Resolution(rule_task, rule_task, "rejected",
                          "already requested one recapture for this pod: rules decide now", proposal)
    if rule_task == "TREAT" and d["confidence"] >= STRONG_CONF and d["n_boxes"] >= STRONG_BOXES:
        return Resolution(rule_task, rule_task, "rejected",
                          f"evidence too strong to soften ({d['n_boxes']} boxes at {d['confidence']}): rules kept",
                          proposal)
    if rule_task == "RECAPTURE":
        return Resolution(rule_task, rule_task, "agreed", "both want a recapture", proposal)
    return Resolution("RECAPTURE", rule_task, "downgraded",
                      f"accepted: LLM asks for a better look instead of {rule_task}", proposal)


class LlmPlanner:
    def __init__(self, backend):
        self.backend = backend

    def plan(self, pod, classification, decision, history, step, n_steps, harvest_stage, image_path=None,
             prior_downgrade=False):
        ev = build_evidence(pod, classification, decision, history, step, n_steps, harvest_stage)
        rule_task = ev["rule_table"]["task"]
        try:
            text = self.backend(SYSTEM_PROMPT, ev, image_path)
            proposal = parse_proposal(text)
        except Exception as exc:                       # offline, timeout, bad model name...
            res = Resolution(rule_task, rule_task, "fallback", f"backend error ({exc.__class__.__name__}): rules kept")
            res.source = getattr(self.backend, "name", "?")
            return ev, res
        res = resolve(rule_task, ev, proposal, prior_downgrade)
        res.source = getattr(self.backend, "name", "?")
        return ev, res
