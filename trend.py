"""
trend.py

Turns a pod's history (from dynamo_client.LocalDynamoTable) plus its
current visit's reading into a short human-readable trend note --
"stage advanced", "disease flagged 3rd visit in a row", etc. This is
what makes the persisted history actually useful for something beyond
storage: it's the signal a real system would use to decide whether
"still diseased after 3 checks" should escalate differently than "just
flagged for the first time", even though decide_action() itself only
ever sees one frame at a time.

Deliberately NOT wired into decide_action() itself -- that function's
four actions are a per-frame reading, tested and working, and folding
multi-visit trend logic into it would change what a single classify_pod()
call means. This stays a separate, additive layer: main.py logs the
trend note alongside the per-visit decision rather than letting it
override it.
"""


def summarize_trend(prior_rows, current):
    """
    prior_rows: this pod's history from BEFORE this visit (oldest first),
                as returned by LocalDynamoTable.query() before the
                current visit's put_item() call.
    current: this visit's classification dict (growth_stage, disease_flag, ...).
    Returns a short string, or "" if there's no prior visit to compare against.
    """
    if not prior_rows:
        return "first visit"

    notes = []
    last = prior_rows[-1]

    if current["growth_stage"] != last.get("growth_stage"):
        notes.append(f"stage advanced: {last.get('growth_stage')} -> {current['growth_stage']}")
    else:
        notes.append(f"stage unchanged ({current['growth_stage']})")

    if current["disease_flag"]:
        streak = 1
        for row in reversed(prior_rows):
            if row.get("disease_flag"):
                streak += 1
            else:
                break
        notes.append(f"disease flagged {streak}x in a row" if streak > 1 else "disease newly flagged")
    elif last.get("disease_flag"):
        notes.append("disease cleared since last visit")

    return "; ".join(notes)
