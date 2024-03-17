import json
from pathlib import Path
from datetime import datetime


def write_evaluations(evaluations, model_name, dataset_name, split):
    current_date = datetime.now().strftime("%m_%d_%Y")
    current_time = datetime.now().strftime("%H_%M_%S")
    results_dir = Path(
        f"../results/{current_date}/{model_name}/{dataset_name}_{split}_{current_time}"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "results.json", "w") as f:
        f.write(json.dumps(evaluations))

    print(f"Results written to {results_dir}")
