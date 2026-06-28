
import sys
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

sys.path.append("..")
ROOT_DIR = Path(__file__).resolve().parent.parent

from src.data_loader import DataLoader
from src.logger import Logger
from src.pipeline import HotelPipeline

logger = Logger("baseline_modeling")
loader=DataLoader("../data/merged/hotels.csv", "baseline_modeling")
df=loader.load()

output_dir_path = ROOT_DIR / "models" / "baseline"
output_dir_path.mkdir(parents=True, exist_ok=True)



models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

best_score = -1
best_model = None
best_name = None

logger.info("=" * 50)
logger.info("Starting baseline model comparison")

for name, model in models.items():
    logger.info(f"Training {name}")
    pipeline = HotelPipeline(
        df=df,
        target="price_category",
        log_file="baseline_modeling",
        model=model
    )

    pipeline.fit()

    score = pipeline.score()
    logger.info(f"{name} score: {score:.4f}")
    if score > best_score:
        best_score = score
        best_model = pipeline.model
        best_name = name
        logger.info(f"New best model: {best_name} ({best_score:.4f})")

output_file = output_dir_path / f"{best_name}.joblib"
output_file.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, output_file)

logger.info("=" * 50)
logger.info(f"Best model: {best_name}")
logger.info(f"Best score: {best_score:.4f}")
logger.info("Best model saved to ../models/baseline/best_model.joblib")
logger.info("=" * 50)