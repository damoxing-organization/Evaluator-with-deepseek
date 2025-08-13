
import json
from config import Config
from ai_parser import parse_entry

def load_data():
    model_outputs = []
    ground_truth = {}
    used_ids = set()

    with open(Config.INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            parsed = parse_entry(raw, used_ids)
            model_outputs.append(parsed)
            if parsed["gold_answer"]:
                ground_truth[parsed["ID"]] = {
                    "question": parsed["question"],
                    "answer": parsed["gold_answer"],
                    "question_type": parsed["question_type"]
                }

    return ground_truth, model_outputs
