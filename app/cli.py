"""
CLI for Grand Mobile Support Copilot:
- Batch prediction: `python -m app.cli predict --in tickets.jsonl --out pred.jsonl`
- Classifier training: `python -m app.cli train`
- Evaluation (Accuracy for categories, Macro-F1 for priorities, latency, cost): `python -m app.cli evaluate`
"""
import argparse
import gzip
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.schemas import TriageRequest
from app.triage import TriageService


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    open_fn = (
        (lambda p: gzip.open(p, "rt", encoding="utf-8"))
        if path.suffix == ".gz" or path.name.endswith(".jsonl.gz")
        else (lambda p: open(p, "r", encoding="utf-8"))
    )
    records: List[Dict[str, Any]] = []
    with open_fn(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_macro_f1(y_true: List[str], y_pred: List[str]) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Compute exact Macro-F1 and per-class precision/recall/F1 without external dependencies."""
    if not y_true:
        return 0.0, {}
    classes = sorted(set(y_true) | set(y_pred))
    per_class: Dict[str, Dict[str, float]] = {}
    f1_scores: List[float] = []

    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == cls),
        }
        # Include in macro average if class is present in ground truth
        if per_class[cls]["support"] > 0:
            f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return round(macro_f1, 4), per_class


def run_predict(in_path: Path, out_path: Path) -> Dict[str, Any]:
    service = TriageService()
    records = read_jsonl(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_cat: List[str] = []
    y_pred_cat: List[str] = []
    y_true_prio: List[str] = []
    y_pred_prio: List[str] = []
    latencies_ms: List[float] = []
    total_cost = 0.0
    escalated = 0
    is_synthetic = False

    outputs: List[Dict[str, Any]] = []
    for idx, raw in enumerate(records):
        if raw.get("is_synthetic_demo"):
            is_synthetic = True
        # Support top-level `text` fallback if `messages` not present
        # Never pass ground-truth labels into the triage input (they would leak into facts.extra_fields)
        payload = {k: v for k, v in raw.items() if k not in ("category", "priority", "label", "is_synthetic_demo")}
        if "messages" not in payload and "text" in payload:
            payload["messages"] = [{"role": "player", "text": str(payload["text"]), "attachments": int(payload.get("attachments", 0))}]

        req = TriageRequest(**payload)
        res = service.analyze(req, record_cache=True)
        out_dict = res.dict()
        outputs.append(out_dict)

        latencies_ms.append(res.processing_time_ms)
        total_cost += res.cost_usd
        if res.needs_human:
            escalated += 1

        gt_cat = raw.get("category") or raw.get("label")
        if isinstance(gt_cat, str) and gt_cat in service.taxonomy.valid_categories:
            y_true_cat.append(gt_cat)
            y_pred_cat.append(res.category)

        gt_prio = raw.get("priority")
        if isinstance(gt_prio, str) and gt_prio in service.taxonomy.valid_priorities:
            y_true_prio.append(gt_prio)
            y_pred_prio.append(res.priority)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in outputs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    lat_sorted = sorted(latencies_ms)
    n = len(lat_sorted)
    avg_lat = round(sum(lat_sorted) / n, 2) if n else 0.0
    p50_lat = round(lat_sorted[n // 2], 2) if n else 0.0
    p95_idx = min(n - 1, max(0, int(math.ceil(0.95 * n) - 1))) if n else 0
    p95_lat = round(lat_sorted[p95_idx], 2) if n else 0.0

    cat_acc: Optional[float] = None
    if y_true_cat:
        correct = sum(1 for t, p in zip(y_true_cat, y_pred_cat) if t == p)
        cat_acc = round(correct / len(y_true_cat), 4)

    prio_macro_f1: Optional[float] = None
    prio_per_class: Dict[str, Any] = {}
    if y_true_prio:
        prio_macro_f1, prio_per_class = compute_macro_f1(y_true_prio, y_pred_prio)

    summary = {
        "input_file": str(in_path),
        "output_file": str(out_path),
        "is_synthetic_demo_data": is_synthetic or "synthetic" in in_path.name.lower(),
        "tickets_processed": n,
        "escalation_rate_pct": round((escalated / n) * 100.0, 2) if n else 0.0,
        "category_accuracy": cat_acc,
        "category_labeled_count": len(y_true_cat),
        "priority_macro_f1": prio_macro_f1,
        "priority_labeled_count": len(y_true_prio),
        "priority_per_class": prio_per_class,
        "latency_mean_ms": avg_lat,
        "latency_p50_ms": p50_lat,
        "latency_p95_ms": p95_lat,
        "avg_cost_usd_per_ticket": round(total_cost / n, 6) if n else 0.0,
    }
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Grand Mobile Support Copilot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # predict
    p_pred = subparsers.add_parser("predict", help="Batch triage JSONL tickets")
    p_pred.add_argument("--in", dest="in_file", required=True, help="Input JSONL file path")
    p_pred.add_argument("--out", dest="out_file", required=True, help="Output JSONL file path")

    # train
    p_train = subparsers.add_parser("train", help="Train classifier on tickets_train.jsonl.gz")
    p_train.add_argument("--train-file", dest="train_file", default=None, help="Training JSONL/JSONL.GZ path")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate on validation dataset")
    p_eval.add_argument("--valid-file", dest="valid_file", default=None, help="Validation JSONL path")
    p_eval.add_argument("--out", dest="out_file", default="pred_valid.jsonl", help="Output predictions path")

    args = parser.parse_args(argv)

    if args.command == "train":
        train_path = Path(args.train_file) if args.train_file else settings.train_data_path
        service = TriageService(train_path=train_path)
        cnt = service.classifier.train_from_jsonl(train_path, save_to=settings.model_weights_path)
        mode_str = "SYNTHETIC DEMO" if "synthetic" in train_path.name.lower() else "REAL"
        print(f"[{mode_str}] Trained classifier on {cnt} samples -> {settings.model_weights_path}")
        return 0

    if args.command == "predict":
        in_path = Path(args.in_file)
        if not in_path.exists():
            # Fallback check in data/ or examples/ if default tickets.jsonl is requested
            for alt in [
                settings.valid_data_path,
                settings.root_dir / "examples" / "sample_tickets.jsonl",
            ]:
                if alt.exists() and in_path.name in ("tickets.jsonl", "tickets_valid.jsonl"):
                    print(f"[NOTICE] '{in_path}' not found; using demonstration file '{alt}'")
                    in_path = alt
                    break
        out_path = Path(args.out_file)
        summary = run_predict(in_path, out_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.command == "evaluate":
        val_path = Path(args.valid_file) if args.valid_file else settings.valid_data_path
        out_path = Path(args.out_file)
        summary = run_predict(val_path, out_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

