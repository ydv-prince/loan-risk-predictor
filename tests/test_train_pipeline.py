import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.train_pipeline import TrainingPipeline

if __name__ == "__main__":
    print("Testing Training Pipeline...")
    TrainingPipeline().run_pipeline()
    print("Training pipeline test completed successfully.")
