from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def tokenize(text: str) -> List[str]:
    """Tokeniza un texto en palabras en minúscula."""

    return [token.strip().lower() for token in text.split() if token.strip()]


@dataclass
class SimpleEmotionModel:
    label_counts: Dict[str, int] = field(default_factory=dict)
    word_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    total_documents: int = 0
    vocabulary: List[str] = field(default_factory=list)

    @classmethod
    def train(cls, samples: Iterable[tuple[str, str]]) -> "SimpleEmotionModel":
        label_counts: Dict[str, int] = {}
        word_counts: Dict[str, Dict[str, int]] = {}
        vocabulary: set[str] = set()
        total_documents = 0

        for text, label in samples:
            total_documents += 1
            label_counts[label] = label_counts.get(label, 0) + 1
            if label not in word_counts:
                word_counts[label] = {}

            for token in tokenize(text):
                vocabulary.add(token)
                word_counts[label][token] = word_counts[label].get(token, 0) + 1

        return cls(
            label_counts=label_counts,
            word_counts=word_counts,
            total_documents=total_documents,
            vocabulary=sorted(vocabulary),
        )

    def _label_log_probability(self, label: str, tokens: List[str]) -> float:
        label_total_words = sum(self.word_counts.get(label, {}).values())
        vocab_size = max(len(self.vocabulary), 1)
        log_prob = math.log(self.label_counts[label] / self.total_documents)

        for token in tokens:
            count = self.word_counts.get(label, {}).get(token, 0)
            log_prob += math.log((count + 1) / (label_total_words + vocab_size))
        return log_prob

    def predict(self, text: str) -> str:
        tokens = tokenize(text)
        best_label: Optional[str] = None
        best_score = float("-inf")

        for label in self.label_counts:
            score = self._label_log_probability(label, tokens)
            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None:
            raise ValueError("El modelo no tiene etiquetas entrenadas.")

        return best_label

    def predict_many(self, texts: Iterable[str]) -> List[str]:
        return [self.predict(text) for text in texts]

    def to_dict(self) -> Dict[str, object]:
        return {
            "label_counts": self.label_counts,
            "word_counts": self.word_counts,
            "total_documents": self.total_documents,
            "vocabulary": self.vocabulary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SimpleEmotionModel":
        return cls(
            label_counts=dict(data.get("label_counts", {})),
            word_counts={k: dict(v) for k, v in dict(data.get("word_counts", {})).items()},
            total_documents=int(data.get("total_documents", 0)),
            vocabulary=list(data.get("vocabulary", [])),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SimpleEmotionModel":
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return cls.from_dict(data)


def _read_csv_samples(
    data_path: Path, text_column: str = "text", label_column: str = "label"
) -> List[tuple[str, str]]:
    samples: List[tuple[str, str]] = []
    with data_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError("El CSV debe contener encabezados.")
        missing_columns = {
            column for column in (text_column, label_column) if column not in reader.fieldnames
        }
        if missing_columns:
            raise ValueError(f"Columnas faltantes en CSV: {', '.join(sorted(missing_columns))}")

        for row in reader:
            text = row[text_column]
            label = row[label_column]
            samples.append((text, label))
    return samples


def _read_texts(data_path: Path, text_column: str = "text") -> List[str]:
    texts: List[str] = []
    with data_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or text_column not in reader.fieldnames:
            raise ValueError(f"La columna {text_column} es obligatoria en el CSV de entrada.")
        for row in reader:
            texts.append(row[text_column])
    return texts


def _write_predictions(output_path: Path, texts: List[str], predictions: List[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "prediction"])
        writer.writerows(zip(texts, predictions))


def _handle_train(args: argparse.Namespace) -> None:
    samples = _read_csv_samples(
        Path(args.data_path), text_column=args.text_column, label_column=args.label_column
    )
    if not samples:
        raise SystemExit("El archivo de entrenamiento está vacío.")

    model = SimpleEmotionModel.train(samples)
    model.save(Path(args.model_path))
    print(f"Modelo entrenado y guardado en {args.model_path}")


def _handle_predict(args: argparse.Namespace) -> None:
    if (args.text is None and args.input is None) or (args.text is not None and args.input is not None):
        raise SystemExit("Proporcione solo --text o --input, pero no ambos.")

    model = SimpleEmotionModel.load(Path(args.model_path))

    if args.text is not None:
        prediction = model.predict(args.text)
        print(prediction)
        return

    texts = _read_texts(Path(args.input), text_column=args.text_column)
    predictions = model.predict_many(texts)

    if args.output:
        _write_predictions(Path(args.output), texts, predictions)
        print(f"Predicciones guardadas en {args.output}")
    else:
        for item_text, pred in zip(texts, predictions):
            print(f"{item_text}\t{pred}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Herramienta CLI para entrenar y predecir emociones simples."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Entrena un modelo simple de emociones")
    train_parser.add_argument(
        "data_path",
        help="Ruta al CSV de entrenamiento con columnas text y label.",
    )
    train_parser.add_argument(
        "--model-path",
        default="model.json",
        help="Ruta donde se guardará el modelo entrenado.",
    )
    train_parser.add_argument(
        "--text-column",
        default="text",
        help="Columna del CSV que contiene el texto.",
    )
    train_parser.add_argument(
        "--label-column",
        default="label",
        help="Columna del CSV que contiene la etiqueta.",
    )
    train_parser.set_defaults(func=_handle_train)

    predict_parser = subparsers.add_parser(
        "predict", help="Realiza predicciones a partir de un modelo entrenado"
    )
    predict_parser.add_argument(
        "--model-path",
        default="model.json",
        help="Ruta al modelo entrenado.",
    )
    predict_parser.add_argument(
        "--text",
        help="Texto individual para predecir su emoción.",
    )
    predict_parser.add_argument(
        "--input",
        help="CSV con columna de texto para predecir en lote.",
    )
    predict_parser.add_argument(
        "--output",
        help="Ruta para guardar predicciones en CSV.",
    )
    predict_parser.add_argument(
        "--text-column",
        default="text",
        help="Columna del CSV de entrada con el texto.",
    )
    predict_parser.set_defaults(func=_handle_predict)

    return parser


def app(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    app()
