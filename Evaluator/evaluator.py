import json
import os
from typing import List, Dict, Tuple, Optional
from config import Config
from deepseek_evaluator import DeepSeekEvaluator
from tqdm import tqdm


def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    # 统一空白和大小写，去除包裹符号
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

    def evaluate(self, ground_truth: Dict[str, Dict], model_outputs: List[Dict]) -> Tuple[
        float, List[Dict], Optional[float]]:
        """
        Args:
            ground_truth: {ID: {question, answer, question_type}}
            model_outputs: list of {ID, question, gold_answer, pred_answer, question_type}
        Returns:
            acc: 客观题（或被判定为完全正确的主观题）准确率
            errors: 错误样本（含基本字段）
            subjective_avg: 主观题平均分（0~1），若没有主观题则为 None
        """
        results = []
        errors = []
        correct_count = 0
        total_count = 0

        subj_scores = []  # 用于主观题平均分（0~1）

        # 使用 tqdm 展示评估进度
        for item in tqdm(model_outputs, total=len(model_outputs), desc="评估中", unit="题"):
            _id = item.get("ID")
            gt = ground_truth.get(_id)
            pred_answer = item.get("pred_answer", "")
            gold_answer = (gt or {}).get("answer") or item.get("gold_answer", "")
            qtype = (gt or {}).get("question_type") or item.get("question_type", "") or "未知"
            question = item.get("question", "")

            if not gold_answer:
                # 没有标准答案，跳过 ACC 统计，但写入结果文件以便排查
                results.append({
                    "ID": _id, "question_type": qtype, "question": question,
                    "gold_answer": gold_answer, "pred_answer": pred_answer,
                    "is_correct": None, "subjective_score": None, "subjective_label": None
                })
                continue

            total_count += 1

            # 1) 先尝试客观精确匹配
            if self._judge_exact(gold_answer, pred_answer):
                correct = True
                subj_score = 1.0
                subj_label = "完全符合"
            else:
                # 2) 调用主观打分（DeepSeek），得到 [0,1] 之间的分数
                #    若请求失败，score 可能为 None
                correct, subj_score, subj_label = self.deepseek.evaluate_answer(qtype, question, gold_answer, pred_answer)


            if subj_score is not None:
                subj_scores.append(subj_score)

            # 认为 score==1.0 为正确；否则为错误（包括 None）
            if correct:
                correct_count += 1
            else:
                errors.append({
                    "ID": _id,
                    "question": question,
                    "gold_answer": gold_answer,
                    "pred_answer": pred_answer,
                    "question_type": qtype,
                    "subjective_label": subj_label
                })

            results.append({
                "ID": _id,
                "question_type": qtype,
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "is_correct": bool(correct),
                "subjective_score": subj_score,
                "subjective_label": subj_label
            })

        acc = (correct_count / total_count) if total_count else 0.0
        subjective_avg = (sum(subj_scores) / len(subj_scores)) if subj_scores else None

        # 写出结果文件
        with open(Config.RESULTS_PATH, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        with open(Config.ERRORS_PATH, "w", encoding="utf-8") as f:
            for item in errors:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return acc, errors, subjective_avg
