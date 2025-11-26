"""CLI simple para entrenar y predecir intensidad emocional."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from .data import load_hf_dataset
from .model import EmotionModel, train_model


def _load_texts_from_file(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _read_dataset_from_args(args: argparse.Namespace) -> tuple[pd.DataFrame, str, str]:
    if args.dataset:
        return load_hf_dataset(
            name=args.dataset,
            split=args.split,
            text_column=args.text_column,
            label_column=args.label_column,
            limit=args.limit,
        )

    if not args.data_file:
        raise SystemExit("Debes indicar --dataset o --data-file para entrenar.")

    df = pd.read_csv(args.data_file)
    text_col = args.text_column or "text"
    label_col = args.label_column or "label"
    missing = [col for col in (text_col, label_col) if col not in df.columns]
    if missing:
        raise SystemExit(f"Faltan columnas en el CSV: {missing}")
    if args.limit:
        df = df.iloc[: args.limit]
    return df[[text_col, label_col]], text_col, label_col


def cmd_train(args: argparse.Namespace) -> None:
    df, text_col, label_col = _read_dataset_from_args(args)
    model, result = train_model(
        df,
        text_column=text_col,
        label_column=label_col,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    output = Path(args.model_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)

    print("Modelo guardado en", output)
    print("Accuracy:", f"{result.accuracy:.3f}")
    print("\nReporte de clasificación:\n", result.report)


def cmd_predict(args: argparse.Namespace) -> None:
    model = EmotionModel.load(args.model)
    texts: List[str] = []

    if args.text:
        texts.append(args.text)
    if args.file:
        texts.extend(_load_texts_from_file(Path(args.file)))
    if not texts:
        raise SystemExit("Debes indicar --text o --file con ejemplos a predecir.")

    predictions = model.predict(texts)
    for text, pred in zip(texts, predictions):
        print(f"[{pred}] {text}")

    if args.probabilities:
        probas = model.predict_proba(texts)
        classes = model.pipeline.named_steps["clf"].classes_
        print("\nProbabilidades:")
        for text, row in zip(texts, probas):
            formatted = ", ".join(f"{cls}={prob:.2f}" for cls, prob in zip(classes, row))
            print(f"- {text[:50]}... -> {formatted}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Herramientas CLI para prueba_emociones")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Entrenar un modelo")
    train_parser.add_argument("--dataset", help="Nombre del dataset de Hugging Face a descargar")
    train_parser.add_argument("--split", default="train", help="Split a usar del dataset")
    train_parser.add_argument("--data-file", type=Path, help="CSV local con columnas de texto y etiqueta")
    train_parser.add_argument("--text-column", help="Nombre de la columna de texto")
    train_parser.add_argument("--label-column", help="Nombre de la columna de etiqueta")
    train_parser.add_argument("--limit", type=int, help="Limitar filas para iteraciones rápidas")
    train_parser.add_argument("--model-out", default="models/prueba_emociones.joblib", help="Ruta de salida del modelo")
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Proporción de validación")
    train_parser.add_argument("--random-state", type=int, default=42, help="Semilla de aleatoriedad")
    train_parser.set_defaults(func=cmd_train)

    predict_parser = sub.add_parser("predict", help="Predecir con un modelo entrenado")
    predict_parser.add_argument("--model", required=True, help="Ruta al modelo .joblib")
    predict_parser.add_argument("--text", help="Texto individual a predecir")
    predict_parser.add_argument("--file", help="Archivo de texto con un ejemplo por línea")
    predict_parser.add_argument("--probabilities", action="store_true", help="Mostrar probabilidades")
    predict_parser.set_defaults(func=cmd_predict)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":
    main()
