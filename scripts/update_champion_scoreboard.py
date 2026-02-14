#!/usr/bin/env python3
import argparse
import glob
import json
import os
from datetime import datetime, timezone


def load_summaries(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8-sig") as f:
            obj = json.load(f)
        rows.append(
            {
                "path": path,
                "round": int(obj.get("round", 0)),
                "tag": obj.get("tag"),
                "champion_before": obj.get("champion_before"),
                "challenger_before": obj.get("challenger_before"),
                "champion_after": obj.get("champion_after"),
                "challenger_after": obj.get("challenger_after"),
                "challenger_win_rate_decisive": float(obj.get("challenger_win_rate_decisive", 0.0)),
                "draws": int(obj.get("draws", 0)),
                "total_games": int(obj.get("total_games", 0)),
                "promoted": bool(obj.get("promoted", False)),
            }
        )
    return rows


def make_markdown(rows, limit):
    rows = sorted(rows, key=lambda x: x["tag"] or "")[-limit:]
    lines = []
    lines.append("## 챔피언전 누적 점수표 (자동 생성)")
    lines.append(f"- updated_at: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- source: logs/champ-cycle-*-summary.json (report 기반 산출)")
    lines.append("")
    lines.append("| tag | challenger_before | champion_before | challenger_dec_win_rate | promoted | champion_after |")
    lines.append("|---|---|---|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['challenger_before']} | {r['champion_before']} | "
            f"{r['challenger_win_rate_decisive']*100:.4f}% | {str(r['promoted']).lower()} | {r['champion_after']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def replace_section(path, header, body):
    start_marker = f"{header}\n"
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        return
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    idx = text.find(start_marker)
    if idx < 0:
        new_text = text.rstrip() + "\n\n" + body
    else:
        next_idx = text.find("\n## ", idx + len(start_marker))
        if next_idx < 0:
            new_text = text[:idx] + body
        else:
            new_text = text[:idx] + body + text[next_idx + 1 :]
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_text)


def main():
    parser = argparse.ArgumentParser(description="Build champion-cycle scoreboard from summary reports.")
    parser.add_argument("--pattern", default="logs/champ-cycle-*-summary.json")
    parser.add_argument("--out-json", default="logs/champion-scoreboard.json")
    parser.add_argument("--out-md", default="logs/champion-scoreboard.md")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--update-log", default="TRAINING_RUN_LOG.md")
    args = parser.parse_args()

    rows = load_summaries(args.pattern)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_pattern": args.pattern,
        "count": len(rows),
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md = make_markdown(rows, max(1, args.limit))
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)

    replace_section(args.update_log, "## 챔피언전 누적 점수표 (자동 생성)", md)
    print(f"updated: {args.out_json}")
    print(f"updated: {args.out_md}")
    print(f"updated: {args.update_log}")


if __name__ == "__main__":
    main()
