import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils.extract_top_features import extract_and_save_top_features

if __name__ == "__main__":
    top_features, feature_means = extract_and_save_top_features()
    print("="*100)
    print(f"The top features are:\n\n{top_features}\n")
    print("="*100)