
import re
import uuid

def extract_id(entry: dict, used_ids: set) -> str:
    id_candidate = entry.get("ID") or entry.get("id") or entry.get("metadata", {}).get("ID")
    if id_candidate:
        return str(id_candidate)
    new_id = str(uuid.uuid4())
    while new_id in used_ids:
        new_id = str(uuid.uuid4())
    return new_id

def guess_question_type(text: str) -> str:
    if any(k in text for k in ["填空", "空缺", "填写"]):
        return "填空"
    elif any(k in text for k in ["选择", "选出", "单选", "多选"]):
        return "选择"
    elif any(k in text for k in ["判断", "对错"]):
        return "判断"
    elif any(k in text for k in ["简答", "简述", "说明", "解释"]):
        return "简答"
    else:
        return "未知"

def parse_entry(entry: dict, used_ids: set) -> dict:
    parsed = {}
    parsed["ID"] = extract_id(entry, used_ids)
    used_ids.add(parsed["ID"])

    question = entry.get("input") or entry.get("prompt") or entry.get("question") or ""
    question = question.strip().replace("<|endoftext|>", "")
    parsed["question"] = question

    pred = entry.get("predict") or entry.get("output") or entry.get("answer") or ""
    parsed["pred_answer"] = re.sub(r"^解答[：:：\s]*", "", pred.strip())

    gold = entry.get("label") or entry.get("gold_answer") or entry.get("reference") or ""
    parsed["gold_answer"] = re.sub(r"^(输出|答案)[：:：\s]*", "", gold.strip())

    parsed["question_type"] = entry.get("question_type") or guess_question_type(question)
    return parsed
