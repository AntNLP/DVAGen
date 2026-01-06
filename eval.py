from src.configs import get_eval_args
from src.infer.eval import evaluate


if __name__ == "__main__":
    eval_args = get_eval_args()
    results = evaluate(eval_args)
    print(f"Evaluation Results of {eval_args.eval.test_data_file} ({eval_args.eval.task_type.value} Task):")
    print(results)
