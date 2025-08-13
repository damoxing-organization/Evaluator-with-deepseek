import json
import os
from typing import List, Dict, Tuple, Optional
from config import Config
from deepseek_evaluator import DeepSeekEvaluator
from tqdm import tqdm


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().replace("\u3000", " ")
    s = s.replace("\r\n", "\n")
    return s


class Evaluator:
    def __init__(self):
        self.deepseek = DeepSeekEvaluator()
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    def _judge_exact(self, gold: str, pred: str) -> bool:
        g = _normalize_text(gold)
        p = _normalize_text(pred)
        return g != "" and p != "" and g == p

    def _is_subjective(self, qtype: str, question: str, gold_answer: str) -> bool:
        """在缺少 question_type 时，用题干关键词/答案长度做启发式判定主观题"""
        qs = str(qtype or "").lower()
        if ("主观" in qs) or (qs in ("subjective",)):
            return True

        q = str(question or "")
        # 常见主观题关键词（可按需增减）
        subjective_keywords = [
            "请简要回答", "请证明", "试证明", "证明", "试述", "简述",
            "说明", "解释", "比较", "为何", "为什么", "试问", "试分析", "推导"
        ]
        if any(kw in q for kw in subjective_keywords):
            return True

        # 标准答案很长通常也是主观题（阈值可调）
        if len(_normalize_text(gold_answer)) >= 80:
            return True

        return False

    def _score_value_from_label(self, label: Optional[str]) -> Optional[float]:
        if label is None:
            return None
        mapping = {
            "完全符合": 1.0,
            "基本符合": 0.75,
            "部分符合": 0.5,
            "不太符合": 0.25,
            "完全不符": 0.0,
        }
        return mapping.get(label, None)

    def _strip_newlines(self, text: str) -> str:
        """去掉字符串里的换行，防止jsonl多行"""
        if text is None:
            return ""
        return str(text).replace("\r\n", " ").replace("\n", " ")

    def evaluate(self, ground_truth: Dict[str, Dict], model_outputs: List[Dict]) -> Tuple[
        float, List[Dict], Optional[float]]:
        results = []
        errors = []
        correct_count = 0
        total_count = 0

        subj_scores = []

        for item in tqdm(model_outputs, total=len(model_outputs), desc="评估中", unit="题"):
            _id = item.get("ID")
            gt = ground_truth.get(_id)
            pred_answer = item.get("pred_answer", "")
            gold_answer = (gt or {}).get("answer") or item.get("gold_answer", "")
            qtype = (gt or {}).get("question_type") or item.get("question_type", "") or "未知"
            question = item.get("question", "")

            if not gold_answer:
                results.append({
                    "ID": _id,
                    "question": self._strip_newlines(question),
                    "gold_answer": self._strip_newlines(gold_answer),
                    "pred_answer": self._strip_newlines(pred_answer),
                    "is_correct": None,
                    "subjective_score": None,
                    "subjective_label": None
                })
                continue

            total_count += 1

            if self._judge_exact(gold_answer, pred_answer):
                correct = True
                subj_score = 1.0
                subj_label = "完全符合"
            else:
                correct, subj_score, subj_label = self.deepseek.evaluate_answer(qtype, question, gold_answer, pred_answer)

            if subj_score is not None:
                subj_scores.append(subj_score)

            if correct:
                correct_count += 1
            else:
                errors.append({
                    "ID": _id,
                    "question": self._strip_newlines(question),
                    "gold_answer": self._strip_newlines(gold_answer),
                    "pred_answer": self._strip_newlines(pred_answer),
                    "subjective_label": subj_label
                })

            if self._is_subjective(qtype, question, gold_answer):
                score_value = subj_score if subj_score is not None else self._score_value_from_label(subj_label)
                results.append({
                    "ID": _id,
                    "score_label": subj_label,
                    "score_value": score_value,
                    "question": self._strip_newlines(question),
                    "gold_answer": self._strip_newlines(gold_answer),
                    "pred_answer": self._strip_newlines(pred_answer)
                })
            else:
                results.append({
                    "ID": _id,
                    "question": self._strip_newlines(question),
                    "gold_answer": self._strip_newlines(gold_answer),
                    "pred_answer": self._strip_newlines(pred_answer),
                    "is_correct": bool(correct),
                    "subjective_score": subj_score,
                    "subjective_label": subj_label
                })

        acc = (correct_count / total_count) if total_count else 0.0
        subjective_avg = (sum(subj_scores) / len(subj_scores)) if subj_scores else None

        with open(Config.RESULTS_PATH, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(Config.ERRORS_PATH, "w", encoding="utf-8") as f:
            for item in errors:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return acc, errors, subjective_avg
