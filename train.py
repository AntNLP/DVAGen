from src.configs import get_train_args
from src.train.train import train


if __name__ == "__main__":
    train_args = get_train_args()
    # print(train_args)
    train(train_args)
