"""Evaluate the violation detector against hand-labelled ground truth (all clips).

Run after detect_violations.py has produced outputs/<clip>_events.csv for every clip:
    python3 backend/evaluate.py

Ground truth = how many cars really run the red light in each clip, counted by watching the video.
A confirmed VIOLATION is a true positive if it matches a real red-light runner. LATE_ENTRY and
NEEDS_REVIEW are reported separately: they go to a human and are not counted as confirmed violations.
"""
import csv
from pathlib import Path

# clip -> number of cars that really run the red light (from watching the videos)
GROUND_TRUTH = {
    "clip1": 1, "clip2": 1, "clip3": 1, "clip4": 1, "clip5": 1,
    "clip6": 0, "clip7": 1, "clip8": 0, "clip9": 1,
}
NOTES = {
    "clip6": "cross-traffic car flagged for review (acceptable)",
    "clip8": "green light, all legal",
    "clip9": "one low-confidence car sent to review",
}

rows, tp_all, fp_all, fn_all = [], 0, 0, 0
for clip, truth in GROUND_TRUTH.items():
    path = Path(f"outputs/{clip}_events.csv")
    if not path.exists():
        print(f"missing {path} - run detect_violations.py {clip} first")
        continue
    with open(path) as f:
        events = list(csv.DictReader(f))
    violations = sum(e["event"] == "VIOLATION" for e in events)
    review = sum(e["event"] == "NEEDS_REVIEW" for e in events)
    late = sum(e["event"] == "LATE_ENTRY" for e in events)
    legal = sum(e["event"] == "LEGAL" for e in events)
    tp = min(violations, truth)
    fp = max(violations - truth, 0)
    fn = max(truth - violations, 0)
    tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
    rows.append((clip, truth, violations, tp, fp, fn, review, late, legal, NOTES.get(clip, "")))

if rows:
    print("| Clip | Real runners | Confirmed violations | TP | FP | FN | Needs review | Late entry | Legal | Note |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    precision = tp_all / (tp_all + fp_all) if tp_all + fp_all else float("nan")
    recall = tp_all / (tp_all + fn_all) if tp_all + fn_all else float("nan")
    print()
    print(f"Clips evaluated: {len(rows)}   TP={tp_all}  FP={fp_all}  FN={fn_all}")
    print(f"Precision: {precision:.0%}   Recall: {recall:.0%}")
    with open("outputs/evaluation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "real_runners", "confirmed_violations", "TP", "FP", "FN", "needs_review", "late_entry", "legal", "note"])
        w.writerows(rows)
    print("Saved: outputs/evaluation.csv")