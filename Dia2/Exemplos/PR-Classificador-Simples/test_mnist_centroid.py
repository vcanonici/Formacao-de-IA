"""Testes locais: criam pequenos arquivos IDX validos sem baixar o MNIST."""

from __future__ import annotations

from contextlib import redirect_stdout
import gzip
import io
from pathlib import Path
import struct
import tempfile
import unittest

from mnist_centroid import (
    IMAGE_SIZE, evaluate, examples_from_idx, main, predict, train_centroids,
)


def save_mnist_pair(
    folder: Path, prefix: str, samples: list[tuple[bytes, int]]
) -> tuple[Path, Path]:
    names = {
        "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
        "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
    }
    image_path, label_path = (folder / filename for filename in names[prefix])
    with gzip.open(image_path, "wb") as images:
        images.write(struct.pack(">IIII", 2051, len(samples), 28, 28))
        for pixels, _ in samples:
            images.write(pixels)
    with gzip.open(label_path, "wb") as labels:
        labels.write(struct.pack(">II", 2049, len(samples)))
        for _, label in samples:
            labels.write(bytes([label]))
    return image_path, label_path


class NearestCentroidTests(unittest.TestCase):
    def setUp(self) -> None:
        self.black = bytes(IMAGE_SIZE)
        self.white = bytes([255]) * IMAGE_SIZE
        self.samples = [(self.black, 0), (self.white, 1), (self.black, 0)]

    def test_idx_reader_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images, labels = save_mnist_pair(Path(tmp), "train", self.samples)
            self.assertEqual(list(examples_from_idx(images, labels)), self.samples)
            self.assertEqual(list(examples_from_idx(images, labels, limit=2)), self.samples[:2])

    def test_predict_uses_pixels_and_not_label(self) -> None:
        centroids, count = train_centroids(self.samples)
        self.assertEqual(count, 3)
        self.assertEqual(predict(self.black, centroids), 0)
        self.assertEqual(predict(self.white, centroids), 1)
        self.assertEqual(evaluate([(self.white, 0)], centroids),
                         {"tested": 1, "correct": 0, "accuracy": 0.0})

    def test_rejects_empty_training_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "Nao ha exemplos"):
            train_centroids([])

    def test_rejects_invalid_idx_magic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images, labels = save_mnist_pair(Path(tmp), "train", self.samples)
            with gzip.open(labels, "wb") as destination:
                destination.write(struct.pack(">II", 999, len(self.samples)))
                destination.write(bytes([label for _, label in self.samples]))
            with self.assertRaisesRegex(ValueError, "identificador invalido"):
                list(examples_from_idx(images, labels))

    def test_rejects_mismatched_sample_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images, labels = save_mnist_pair(Path(tmp), "train", self.samples)
            with gzip.open(labels, "wb") as destination:
                destination.write(struct.pack(">II", 2049, len(self.samples) + 1))
            with self.assertRaisesRegex(ValueError, "nao coincide"):
                list(examples_from_idx(images, labels))

    def test_cli_end_to_end_on_synthetic_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            save_mnist_pair(folder, "train", self.samples)
            save_mnist_pair(folder, "test", [(self.black, 0), (self.white, 1)])
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(["--data-dir", str(folder), "--train-limit", "0", "--test-limit", "0"])
            self.assertEqual(result, 0)
            self.assertIn('"accuracy": 1.0', output.getvalue())
            self.assertIn('"train_samples": 3', output.getvalue())


if __name__ == "__main__":
    unittest.main()
