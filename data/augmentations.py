"""Underwater-specific image augmentations for marine species recognition.

Includes both standard augmentations and domain-specific transforms
that simulate real underwater imaging conditions: turbidity, marine snow,
color attenuation, caustic lighting, and backscatter.
"""

import logging
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from PIL import Image

logger = logging.getLogger(__name__)


# --- Custom Underwater Albumentations Transforms ---


class SyntheticTurbidity(A.ImageOnlyTransform):
    """Simulate underwater turbidity with blur and green-blue color shift.

    Args:
        blur_limit: Maximum Gaussian blur kernel size.
        green_shift: Range of green channel intensity increase.
        blue_shift: Range of blue channel intensity increase.
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        blur_limit: Tuple[int, int] = (3, 7),
        green_shift: Tuple[int, int] = (5, 30),
        blue_shift: Tuple[int, int] = (5, 25),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply, p)
        self.blur_limit = blur_limit
        self.green_shift = green_shift
        self.blue_shift = blue_shift

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply turbidity simulation."""
        result = img.copy().astype(np.float32)

        # Apply Gaussian blur
        ksize = np.random.randint(self.blur_limit[0], self.blur_limit[1] + 1)
        if ksize % 2 == 0:
            ksize += 1
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)

        # Shift green and blue channels
        g_shift = np.random.randint(self.green_shift[0], self.green_shift[1] + 1)
        b_shift = np.random.randint(self.blue_shift[0], self.blue_shift[1] + 1)

        result[:, :, 1] = np.clip(result[:, :, 1] + g_shift, 0, 255)  # Green
        result[:, :, 2] = np.clip(result[:, :, 2] + b_shift, 0, 255)  # Blue (BGR)

        return result.astype(np.uint8)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("blur_limit", "green_shift", "blue_shift")


class MarineSnow(A.ImageOnlyTransform):
    """Simulate marine snow particles (bright speckles) in underwater images.

    Args:
        num_particles: Range of particle count (min, max).
        particle_size: Range of particle radius in pixels.
        brightness: Range of particle brightness (0-255).
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        num_particles: Tuple[int, int] = (50, 300),
        particle_size: Tuple[int, int] = (1, 3),
        brightness: Tuple[int, int] = (180, 255),
        always_apply: bool = False,
        p: float = 0.3,
    ):
        super().__init__(always_apply, p)
        self.num_particles = num_particles
        self.particle_size = particle_size
        self.brightness = brightness

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply marine snow overlay."""
        result = img.copy()
        h, w = result.shape[:2]

        n_particles = np.random.randint(
            self.num_particles[0], self.num_particles[1] + 1
        )

        for _ in range(n_particles):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            radius = np.random.randint(
                self.particle_size[0], self.particle_size[1] + 1
            )
            bright = np.random.randint(
                self.brightness[0], self.brightness[1] + 1
            )
            color = (bright, bright, bright)
            cv2.circle(result, (x, y), radius, color, -1)

        # Slight blur to make particles look natural
        result = cv2.GaussianBlur(result, (3, 3), 0.5)
        return result

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("num_particles", "particle_size", "brightness")


class ColorAttenuation(A.ImageOnlyTransform):
    """Simulate depth-dependent color attenuation (red channel absorption).

    Underwater, red light is absorbed first, followed by orange and yellow.
    This creates the characteristic blue-green tint of deep water images.

    Args:
        red_reduction: Range of red channel reduction factor (0-1).
        green_reduction: Range of green channel reduction factor (0-1).
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        red_reduction: Tuple[float, float] = (0.2, 0.7),
        green_reduction: Tuple[float, float] = (0.0, 0.2),
        always_apply: bool = False,
        p: float = 0.4,
    ):
        super().__init__(always_apply, p)
        self.red_reduction = red_reduction
        self.green_reduction = green_reduction

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply color attenuation simulating depth."""
        result = img.copy().astype(np.float32)

        red_factor = 1.0 - np.random.uniform(
            self.red_reduction[0], self.red_reduction[1]
        )
        green_factor = 1.0 - np.random.uniform(
            self.green_reduction[0], self.green_reduction[1]
        )

        # Assuming RGB format from albumentations
        result[:, :, 0] *= red_factor    # Red
        result[:, :, 1] *= green_factor  # Green (slight reduction)
        # Blue stays mostly unchanged

        return np.clip(result, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("red_reduction", "green_reduction")


class CausticLighting(A.ImageOnlyTransform):
    """Simulate caustic lighting patterns (rippling light from surface).

    Creates bright, irregular ripple patterns typical of shallow water
    with sunlight filtering through the surface.

    Args:
        intensity: Range of caustic intensity (0-1).
        scale: Range of pattern scale factor.
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        intensity: Tuple[float, float] = (0.1, 0.4),
        scale: Tuple[float, float] = (0.02, 0.08),
        always_apply: bool = False,
        p: float = 0.3,
    ):
        super().__init__(always_apply, p)
        self.intensity = intensity
        self.scale = scale

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply caustic lighting pattern."""
        result = img.copy().astype(np.float32)
        h, w = result.shape[:2]

        intensity = np.random.uniform(self.intensity[0], self.intensity[1])
        freq = np.random.uniform(self.scale[0], self.scale[1])

        # Generate caustic pattern using overlapping sine waves
        y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Multiple overlapping waves at different angles
        phase1 = np.random.uniform(0, 2 * np.pi)
        phase2 = np.random.uniform(0, 2 * np.pi)
        angle1 = np.random.uniform(0, np.pi)
        angle2 = np.random.uniform(0, np.pi)

        wave1 = np.sin(
            freq * (x_coords * np.cos(angle1) + y_coords * np.sin(angle1)) + phase1
        )
        wave2 = np.sin(
            freq * 1.5 * (x_coords * np.cos(angle2) + y_coords * np.sin(angle2))
            + phase2
        )

        # Combine waves and create caustic pattern
        caustic = np.clip((wave1 + wave2) / 2.0, 0, 1)
        caustic = caustic ** 2  # Sharpen the bright spots

        # Apply as additive lighting
        for c in range(3):
            result[:, :, c] += caustic * 255 * intensity

        return np.clip(result, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("intensity", "scale")


class BackscatterReduction(A.ImageOnlyTransform):
    """Simulate backscatter (contrast reduction from suspended particles).

    Args:
        fog_intensity: Range of fog/haze overlay intensity (0-1).
        fog_color: Base color of the backscatter (BGR).
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        fog_intensity: Tuple[float, float] = (0.1, 0.4),
        fog_color: Tuple[int, int, int] = (140, 170, 130),  # Blue-green tint
        always_apply: bool = False,
        p: float = 0.3,
    ):
        super().__init__(always_apply, p)
        self.fog_intensity = fog_intensity
        self.fog_color = fog_color

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply backscatter simulation."""
        result = img.copy().astype(np.float32)
        alpha = np.random.uniform(self.fog_intensity[0], self.fog_intensity[1])

        fog = np.full_like(result, self.fog_color, dtype=np.float32)
        result = result * (1 - alpha) + fog * alpha

        return np.clip(result, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("fog_intensity", "fog_color")


# --- Transform Pipeline Factories ---


def get_albumentations_train_transform(
    image_size: int = 224,
    underwater_augment: bool = True,
) -> A.Compose:
    """Get training augmentation pipeline using albumentations.

    Args:
        image_size: Target image size.
        underwater_augment: Whether to include underwater-specific augmentations.

    Returns:
        Albumentations Compose pipeline.
    """
    transforms_list: List[A.BasicTransform] = [
        # Standard augmentations
        A.RandomResizedCrop(
            height=image_size,
            width=image_size,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.33),
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.3,
        ),
    ]

    if underwater_augment:
        transforms_list.extend(
            [
                SyntheticTurbidity(p=0.3),
                MarineSnow(p=0.2),
                ColorAttenuation(p=0.3),
                CausticLighting(p=0.2),
                BackscatterReduction(p=0.2),
            ]
        )

    transforms_list.extend(
        [
            A.GaussNoise(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms_list)


def get_albumentations_test_transform(
    image_size: int = 224,
) -> A.Compose:
    """Get test/validation transform pipeline using albumentations.

    Args:
        image_size: Target image size.

    Returns:
        Albumentations Compose pipeline (resize + center crop + normalize).
    """
    return A.Compose(
        [
            A.Resize(height=int(image_size * 1.143), width=int(image_size * 1.143)),
            A.CenterCrop(height=image_size, width=image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


class AlbumentationsWrapper:
    """Wrapper to make albumentations transforms compatible with PIL images.

    Args:
        transform: Albumentations Compose pipeline.
    """

    def __init__(self, transform: A.Compose) -> None:
        self.transform = transform

    def __call__(self, image: Image.Image) -> "torch.Tensor":
        """Apply transform to a PIL image.

        Args:
            image: PIL Image.

        Returns:
            Transformed image tensor.
        """
        import torch

        img_array = np.array(image)
        augmented = self.transform(image=img_array)
        return augmented["image"]


def get_train_transforms(
    image_size: int = 224,
    underwater_augment: bool = True,
    use_albumentations: bool = True,
) -> "callable":
    """Get training transform pipeline.

    Args:
        image_size: Target image size.
        underwater_augment: Whether to use underwater-specific augmentations.
        use_albumentations: Use albumentations (True) or torchvision (False).

    Returns:
        Transform callable compatible with PIL images.
    """
    if use_albumentations and underwater_augment:
        albu_transform = get_albumentations_train_transform(
            image_size, underwater_augment
        )
        return AlbumentationsWrapper(albu_transform)

    # Fallback torchvision transforms (no underwater-specific augmentations)
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_test_transforms(
    image_size: int = 224,
    use_albumentations: bool = True,
) -> "callable":
    """Get test/validation transform pipeline.

    Args:
        image_size: Target image size.
        use_albumentations: Use albumentations (True) or torchvision (False).

    Returns:
        Transform callable compatible with PIL images.
    """
    if use_albumentations:
        albu_transform = get_albumentations_test_transform(image_size)
        return AlbumentationsWrapper(albu_transform)

    return T.Compose(
        [
            T.Resize(int(image_size * 1.143)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
