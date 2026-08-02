from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TFAutoencoderDataBundle:
    x_train_clean: np.ndarray
    x_test_clean: np.ndarray
    x_train_noisy: np.ndarray
    x_test_noisy: np.ndarray
    y_train: np.ndarray | None = None
    y_test: np.ndarray | None = None


class TFAutoencoderDataLoader:
    """
    Loader utility for NPZ-based denoising autoencoder datasets.
    """

    @staticmethod
    def _ensure_channel_dim(images: np.ndarray) -> np.ndarray:
        if images.ndim == 3:
            return np.expand_dims(images, axis=-1)
        return images

    @staticmethod
    def _to_grayscale(images: np.ndarray) -> np.ndarray:
        if images.shape[-1] == 1:
            return images

        gray = np.mean(images, axis=-1, keepdims=True)
        return gray.astype(np.float32)

    @staticmethod
    def load_npz(
        npz_path: str,
        to_grayscale: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        data = np.load(npz_path)

        x_train = data["x_train"].astype(np.float32)
        x_test = data["x_test"].astype(np.float32)

        y_train = data["y_train"] if "y_train" in data.files else None
        y_test = data["y_test"] if "y_test" in data.files else None

        x_train = TFAutoencoderDataLoader._ensure_channel_dim(x_train)
        x_test = TFAutoencoderDataLoader._ensure_channel_dim(x_test)

        if to_grayscale:
            x_train = TFAutoencoderDataLoader._to_grayscale(x_train)
            x_test = TFAutoencoderDataLoader._to_grayscale(x_test)

        return x_train, x_test, y_train, y_test

    @staticmethod
    def add_noise(
        images: np.ndarray,
        noise_factor: float = 0.2,
        seed: int = 42,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        noisy = images + noise_factor * rng.normal(
            loc=0.0,
            scale=1.0,
            size=images.shape,
        )

        return np.clip(noisy, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def build_noisy_bundle(
        npz_path: str,
        noise_factor: float = 0.2,
        seed: int = 42,
        to_grayscale: bool = True,
    ) -> TFAutoencoderDataBundle:
        x_train_clean, x_test_clean, y_train, y_test = TFAutoencoderDataLoader.load_npz(
            npz_path=npz_path,
            to_grayscale=to_grayscale,
        )

        x_train_noisy = TFAutoencoderDataLoader.add_noise(
            x_train_clean,
            noise_factor=noise_factor,
            seed=seed,
        )

        x_test_noisy = TFAutoencoderDataLoader.add_noise(
            x_test_clean,
            noise_factor=noise_factor,
            seed=seed + 1,
        )

        return TFAutoencoderDataBundle(
            x_train_clean=x_train_clean,
            x_test_clean=x_test_clean,
            x_train_noisy=x_train_noisy,
            x_test_noisy=x_test_noisy,
            y_train=y_train,
            y_test=y_test,
        )
