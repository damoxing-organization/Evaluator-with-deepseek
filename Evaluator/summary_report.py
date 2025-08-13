# -*- coding: utf-8 -*-
"""
独立汇总脚本：
- 手动运行: python summary_report.py [--path /path/to/results.jsonl]
- 默认查找当前目录的 results/results.jsonl
- 计算题作为客观题中的特别类型
- “整体”仅计入 客观题（选择/判断/填空） + 计算题，不计入主观题
"""
from pathlib import Path
import json
import re
import argparse
from collections import Counter

DEFAULT_RESULTS = Path(__file__).resolve().parent / "results" / "results.jsonl"

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

def _detect_type(question: str) -> str:
    q = (question or "").replace("\n", " ").replace("\r", " ")
    if "判断" in q or "对或错" in q or "对/错" in q or "是否正确" in q:
        return "判断"
    if "填写" in q or "填空" in q or "补全" in q or "空缺" in q:
        return "填空"
    if "选择" in q or "单选" in q or "多选" in q or "从以下" in q:
        return "选择"
    if "计算" in q or re.search(r"(^|[，。；,.\s])求(值|解|出)?", q):
        return "计算题"
    if re.search(r"[（(]\s*[A-D][、\.)]", q):
        return "选择"
    return "其他"

def _pct(num: int, den: int) -> float:
    return 0.0 if den == 0 else (num / den * 100.0)

def summarize(results_path: Path) -> str:
    per_type_tot = Counter()
    per_type_ok  = Counter()
    obj_pool_tot = 0   # 选择/判断/填空总数
    obj_pool_ok  = 0
    calc_tot = 0
    calc_ok  = 0
    all_tot  = 0       # 所有题目数
    all_ok   = 0       # 整体正确：客观题 + 计算题

    subj_scores = []
    subj_missing = 0

    for row in _iter_jsonl(results_path):
        all_tot += 1
        q = row.get("question", "")
        t = _detect_type(q)

        has_is_correct = ("is_correct" in row)
        has_subj = ("score_value" in row or "score_label" in row)

        if t == "计算题":
            calc_tot += 1
            per_type_tot[t] += 1
            if has_is_correct:
                if row.get("is_correct", False):
                    calc_ok += 1
                    per_type_ok[t] += 1
            elif "score_value" in row:
                try:
                    sv = float(row["score_value"])
                    if abs(sv - 1.0) < 1e-9:
                        calc_ok += 1
                        per_type_ok[t] += 1
                except:
                    pass
            continue

        if has_is_correct:
            per_type_tot[t] += 1
            if row.get("is_correct", False):
                per_type_ok[t] += 1
                if t in ("选择", "判断", "填空"):
                    obj_pool_ok += 1
            if t in ("选择", "判断", "填空"):
                obj_pool_tot += 1

        if has_subj:
            sv = row.get("score_value", None)
            if sv is None:
                subj_missing += 1
            else:
                try:
                    subj_scores.append(float(sv))
                except:
                    subj_missing += 1

    # 整体：客观题（选择/判断/填空）+ 计算题
    all_ok = obj_pool_ok + calc_ok
    lines = []
    lines.append(f"客观题（选择/判断/填空）正确率：{_pct(obj_pool_ok, obj_pool_tot):.2f}%（{obj_pool_ok}/{obj_pool_tot}）")
    lines.append("细分 & 补充")
    lines.append("各题型正确率：")
    for t in ["判断", "填空", "选择", "计算题"]:
        tot = per_type_tot[t] if t != "计算题" else calc_tot
        ok  = per_type_ok[t] if t != "计算题" else calc_ok
        lines.append(f"{t}：{_pct(ok, tot):.2f}%（{ok}/{tot}）")
    lines.append(f"整体：{_pct(all_ok, obj_pool_tot + calc_tot):.2f}%（{all_ok}/{obj_pool_tot + calc_tot}）。")

    if subj_scores or subj_missing:
        avg = (sum(subj_scores) / len(subj_scores)) if subj_scores else 0.0
        denom = len(subj_scores) + subj_missing
        buckets = [0.0, 0.25, 0.5, 0.75, 1.0]
        dist = Counter()
        for s in subj_scores:
            nearest = min(buckets, key=lambda b: abs(b - float(s)))
            dist[nearest] += 1
        lines.append("主观题")
        lines.append(f"平均得分：{avg:.3f}（按实际有分值的题计算均值；缺失不计入）")
        parts = [f"得{b:g}分: {dist[b]}/{denom}" for b in buckets]
        if subj_missing:
            parts.append(f"未评分/缺失:{subj_missing}/{denom}")
        lines.append(f"得分分布（分母 {denom}）：")
        lines.append("  " + "  ".join(parts))

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="汇总 results.jsonl 评测结果")
    parser.add_argument("--path", type=str, default=None,
                        help="results.jsonl 路径，缺省则为项目根目录 results/results.jsonl")
    args = parser.parse_args()
    path = Path(args.path) if args.path else DEFAULT_RESULTS
    if not path.exists():
        raise FileNotFoundError(f"未找到结果文件：{path}")
    print(summarize(path))

if __name__ == "__main__":
    main()
