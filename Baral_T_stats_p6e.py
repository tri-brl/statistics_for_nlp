import pandas as pd
from pathlib import Path
RESULTS_DIR = Path(r"C:\Users\aviba\results")

rope = pd.read_csv(RESULTS_DIR / "6d_ROPE_classification.csv")
print(rope.groupby(["representation", "classification"])
      .size().unstack(fill_value=0).to_string())