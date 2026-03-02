import numpy as np
from sklearn.model_selection import train_test_split
import pickle

def generate_train_test_data(
    x, y, feature_names, target_names,
    filename="train_test_data.pkl",
    test_size=0.2,
    random_state=42,
):
    """Generate and save train-test split data"""
    x, y = np.array(x), np.array(y)

    # Use 80% for training, 20% for testing
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    with open(filename, "wb") as f:
        pickle.dump({
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "feature_names": feature_names,
            "target_names": target_names
        }, f)
    return x_train, y_train, x_test, y_test, feature_names, target_names

def load_train_test_data(filename="train_test_data.pkl"):
    """Load previously saved train-test split data"""
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["x_train"], data["y_train"], data["x_test"], data["y_test"], data["feature_names"], data["target_names"]

