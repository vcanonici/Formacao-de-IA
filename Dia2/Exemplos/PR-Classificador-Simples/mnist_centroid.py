#!/usr/bin/env python3
"""Classificador MNIST basico por centroide de classe, apenas stdlib.

Este exemplo NAO e uma rede neural. Mostra um PR pequeno e testavel
sem resolver o treinamento e a aplicacao web dos exercicios.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import struct
from typing import Iterable, Iterator

IMAGE_SIZE = 28 * 28
FILE_NAMES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def examples_from_idx(
    image_file: Path, label_file: Path, limit: int = 0
) -> Iterator[tuple[bytes, int]]:
    """Le pares (pixels, rotulo) em streaming; limit=0 usa o arquivo inteiro."""
    if limit < 0:
        raise ValueError("O limite deve ser zero ou positivo.")

    with gzip.open(image_file, "rb") as images, gzip.open(label_file, "rb") as labels:
        image_header = images.read(16)
        label_header = labels.read(8)
        if len(image_header) != 16 or len(label_header) != 8:
            raise ValueError("Cabecalho IDX incompleto.")
        image_magic, image_count, rows, cols = struct.unpack(">IIII", image_header)
        label_magic, label_count = struct.unpack(">II", label_header)
        if image_magic != 2051 or label_magic != 2049:
            raise ValueError("Arquivo IDX com identificador invalido.")
        if rows * cols != IMAGE_SIZE or (rows, cols) != (28, 28):
            raise ValueError("Amostras MNIST devem ter 28 x 28 pixels.")
        if image_count != label_count:
            raise ValueError("Quantidade de imagens e rotulos nao coincide.")

        count = min(image_count, limit) if limit else image_count
        for _ in range(count):
            pixels, label = images.read(IMAGE_SIZE), labels.read(1)
            if len(pixels) != IMAGE_SIZE or len(label) != 1:
                raise ValueError("Arquivo IDX truncado.")
            if label[0] > 9:
                raise ValueError("Rotulo MNIST fora do intervalo 0-9.")
            yield pixels, label[0]


def train_centroids(examples: Iterable[tuple[bytes, int]]) -> tuple[dict[int, list[float]], int]:
    """Calcula a imagem media de cada classe vista no conjunto de treino."""
    totals = [[0] * IMAGE_SIZE for _ in range(10)]
    counts = [0] * 10
    for pixels, label in examples:
        if len(pixels) != IMAGE_SIZE or not 0 <= label <= 9:
            raise ValueError("Exemplo de treinamento invalido.")
        counts[label] += 1
        for index, pixel in enumerate(pixels):
            totals[label][index] += pixel

    num_examples = sum(counts)
    if num_examples == 0:
        raise ValueError("Nao ha exemplos de treinamento.")
    centroids = {
        label: [value / counts[label] for value in totals[label]]
        for label in range(10)
        if counts[label]
    }
    return centroids, num_examples


def predict(pixels: bytes, centroids: dict[int, list[float]]) -> int:
    """Retorna a classe com menor distancia quadratica ao centroide."""
    if len(pixels) != IMAGE_SIZE:
        raise ValueError("Imagem invalida: esperado 28 x 28 pixels.")
    if not centroids:
        raise ValueError("O modelo nao contem classes treinadas.")
    return min(
        centroids,
        key=lambda label: sum(
            (pixel - center) ** 2 for pixel, center in zip(pixels, centroids[label])
        ),
    )


def evaluate(
    examples: Iterable[tuple[bytes, int]], centroids: dict[int, list[float]]
) -> dict[str, int | float]:
    """Avalia os exemplos recebidos; nao usa rotulos durante a previsao."""
    correct = 0
    total = 0
    for pixels, expected in examples:
        total += 1
        correct += int(predict(pixels, centroids) == expected)
    if not total:
        raise ValueError("Nao ha exemplos para avaliacao.")
    return {"tested": total, "correct": correct, "accuracy": correct / total}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Pasta contendo os quatro arquivos originais MNIST IDX+gzip.",
    )
    parser.add_argument(
        "--train-limit", type=int, default=2000,
        help="Quantidade de exemplos de treino (0 = todos; padrao = 2000).",
    )
    parser.add_argument(
        "--test-limit", type=int, default=200,
        help="Quantidade de exemplos de teste (0 = todos; padrao = 200).",
    )
    args = parser.parse_args(argv)
    for value in (args.train_limit, args.test_limit):
        if value < 0:
            parser.error("Os limites devem ser zero ou positivos.")

    train_images, train_labels = FILE_NAMES["train"]
    test_images, test_labels = FILE_NAMES["test"]
    try:
        centroids, n_train = train_centroids(
            examples_from_idx(args.data_dir / train_images, args.data_dir / train_labels,
                              limit=args.train_limit)
        )
        metrics = evaluate(
            examples_from_idx(args.data_dir / test_images, args.data_dir / test_labels,
                              limit=args.test_limit),
            centroids,
        )
    except (OSError, EOFError, ValueError) as error:
        parser.exit(1, f"Erro ao processar MNIST: {error}\n")
    print(json.dumps({"method": "nearest-centroid", "train_samples": n_train, **metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
