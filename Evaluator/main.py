from data_loader import load_data
from evaluator import Evaluator
from config import Config

def main():
    print("开始评估答题大模型性能...")

    ground_truth, model_outputs = load_data()
    print(f"加载完成: {len(model_outputs)} 个模型回答，{len(ground_truth)} 个标准答案")

    evaluator = Evaluator()
    acc, errors, subj_avg = evaluator.evaluate(ground_truth, model_outputs)

    print("\n评估完成!")
    print("\n详细结果已保存至:")
    print(f"- 评估结果: {Config.RESULTS_PATH}")
    print(f"- 错误记录: {Config.ERRORS_PATH}")

if __name__ == "__main__":
    main()
