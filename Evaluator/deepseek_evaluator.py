import requests
import os
class DeepSeekEvaluator:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")  # ← 替换成你的真实 API key

        self.api_url = "https://api.deepseek.com/v1/chat/completions"

    def build_prompt(self, question_type: str, question: str, gold_answer: str, pred_answer: str) -> str:
        return f"""你是一位严格的阅卷教师，请根据题型判断考生答案与标准答案的符合程度，并仅回复以下五种标签之一：
“完全符合”“基本符合”“部分符合”“不太符合”“完全不符”。

题型：{question_type}
题干：{question.strip()}
标准答案：{gold_answer.strip()}
考生答案：{pred_answer.strip()}


对于客观题如单选、多选、判断题，请判断考生答案是否正确，并严格依据标准答案给出判断。
对于填空题，请比对考生答案和标准答案大体含义是否一致。
对于计算题，请根据最终答案和解答过程综合评判。
对于主观题，请根据常识和答案综合判断，评分可以不完全依据上述标准，根据实际情况注意客观性即可。
请根据考生回答的准确性、覆盖点、逻辑表达等进行综合判断，只回复一个标签，不要附加解释。"""

    def evaluate_answer(self, question_type: str, question: str, gold_answer: str, pred_answer: str):
        prompt = self.build_prompt(question_type, question, gold_answer, pred_answer)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个严格的判题老师，只能返回：完全符合、基本符合、部分符合、不太符合、完全不符。"},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"].strip()

            score_map = {
                "完全符合": 1.0,
                "基本符合": 0.75,
                "部分符合": 0.5,
                "不太符合": 0.25,
                "完全不符": 0.0
            }

            for label, score in score_map.items():
                if label in result:
                    return (score == 1.0), score, label

            return False, None, "无效评分"
        except Exception as e:
            print(f"❗ DeepSeek API 请求失败: {e}")
            return False, None, "请求失败"
