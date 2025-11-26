import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from prueba_emociones.cli import main


def run_cli(args: list[str], working_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{working_dir / 'src'}" + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "prueba_emociones.cli.main", *args],
        cwd=working_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def write_training_csv(path: Path) -> None:
    rows = [
        {"text": "Me siento muy feliz hoy", "label": "positivo"},
        {"text": "Estoy triste y desanimado", "label": "negativo"},
        {"text": "El día es maravilloso y alegre", "label": "positivo"},
        {"text": "Odio cuando llueve sin parar", "label": "negativo"},
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def trained_model(tmp_path: Path) -> tuple[Path, Path]:
    data_path = tmp_path / "train.csv"
    model_path = tmp_path / "modelo.json"
    write_training_csv(data_path)
    result = run_cli(["train", str(data_path), "--model-path", str(model_path)], working_dir=Path.cwd())
    assert result.returncode == 0, result.stderr
    return data_path, model_path


def test_train_creates_model_file(trained_model: tuple[Path, Path]) -> None:
    _, model_path = trained_model
    assert model_path.exists()
    data = json.loads(model_path.read_text(encoding="utf-8"))
    assert "label_counts" in data
    assert data["label_counts"].get("positivo", 0) > 0


def test_predict_single_text(trained_model: tuple[Path, Path]) -> None:
    _, model_path = trained_model
    result = run_cli(
        ["predict", "--model-path", str(model_path), "--text", "Estoy muy feliz y contento"],
        working_dir=Path.cwd(),
    )
    assert result.returncode == 0, result.stderr
    assert "positivo" in result.stdout


def test_batch_prediction_writes_output(trained_model: tuple[Path, Path], tmp_path: Path) -> None:
    data_path, model_path = trained_model
    predict_input = tmp_path / "input.csv"
    with predict_input.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text"])
        writer.writeheader()
        writer.writerow({"text": "El clima es horrible y estoy molesto"})
        writer.writerow({"text": "Hoy celebro mi cumpleaños feliz"})

    output_path = tmp_path / "predicciones.csv"
    result = run_cli(
        [
            "predict",
            "--model-path",
            str(model_path),
            "--input",
            str(predict_input),
            "--output",
            str(output_path),
        ],
        working_dir=Path.cwd(),
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
    with output_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    assert len(rows) == 2
    assert {row["prediction"] for row in rows} <= {"positivo", "negativo"}
