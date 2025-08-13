
class Config:
    INPUT_PATH = "./generated_predictions.jsonl"  # 输入预测结果文件
    OUTPUT_DIR = "./results"                     # 输出目录
    RESULTS_PATH = f"{OUTPUT_DIR}/results.jsonl" # 所有评估结果
    ERRORS_PATH = f"{OUTPUT_DIR}/errors.jsonl"   # 错误样本列表
