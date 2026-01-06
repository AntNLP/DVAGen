from src.configs import get_infer_args
from src.infer.chat import chat


if __name__ == "__main__":
    infer_args = get_infer_args()
    chat(infer_args)
