import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
import cv2
from typing import Tuple, List


class ScalogramAugmentor:
    """
    Augments CWT scalograms (time-frequency spectrograms) while preserving respiratory characteristics.
    
    Augmentation types:
    - Intensity/Contrast: Brightness/contrast adjustment (image domain)
    - Frequency Jitter: Small random shifts in frequency axis (preserves RR)
    - Time Jitter: Small random shifts in time axis
    - Gaussian Blur: Smoothing without destroying structure
    - Rotation: Minor rotations for robustness
    - Mixup: Blend two scalograms from same RR class
    """
    
    def __init__(self, 
                 intensity_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.7, 1.3),
                 freq_jitter_pct: float = 0.05,  # ±5% frequency shift
                 time_jitter_pct: float = 0.05,  # ±5% time shift
                 blur_sigma_range: Tuple[float, float] = (0.5, 1.5),
                 rotation_range: Tuple[float, float] = (-5, 5),  # degrees
                 mixup_alpha: float = 0.2):
        
        self.intensity_range = intensity_range
        self.contrast_range = contrast_range
        self.freq_jitter_pct = freq_jitter_pct
        self.time_jitter_pct = time_jitter_pct
        self.blur_sigma_range = blur_sigma_range
        self.rotation_range = rotation_range
        self.mixup_alpha = mixup_alpha
    
    def augment(self, scalogram: np.ndarray) -> np.ndarray:
        """
        Apply a random augmentation to a single scalogram.
        
        Args:
            scalogram: 2D array of shape (n_freqs, n_times), float32, range [0, 1]
            
        Returns:
            Augmented scalogram with same shape and range
        """
        # Ensure input is valid
        scalogram = np.asarray(scalogram, dtype=np.float32)
        if scalogram.ndim != 2:
            raise ValueError(f"Expected 2D scalogram, got shape {scalogram.shape}")
        
        # Randomly choose augmentation type
        aug_type = np.random.choice([
            'intensity',
            'contrast',
            'freq_jitter',
            'time_jitter',
            'blur',
            'rotation'
        ])
        
        if aug_type == 'intensity':
            return self._intensity_augment(scalogram)
        elif aug_type == 'contrast':
            return self._contrast_augment(scalogram)
        elif aug_type == 'freq_jitter':
            return self._freq_jitter(scalogram)
        elif aug_type == 'time_jitter':
            return self._time_jitter(scalogram)
        elif aug_type == 'blur':
            return self._gaussian_blur(scalogram)
        elif aug_type == 'rotation':
            return self._rotation(scalogram)
        else:
            return scalogram
    
    def _intensity_augment(self, scalogram: np.ndarray) -> np.ndarray:
        """
        Multiply all values by a random factor (brightness).
        
        Preserves RR because it scales the entire frequency-time representation uniformly.
        """
        factor = np.random.uniform(*self.intensity_range)
        augmented = scalogram * factor
        return np.clip(augmented, 0.0, 1.0). astype(np.float32)
    
    def _contrast_augment(self, scalogram: np.ndarray) -> np. ndarray:
        """
        Adjust contrast: scale deviations from mean.
        
        Formula: output = mean + factor * (input - mean)
        
        This brightens peaks and darkens valleys without removing structure.
        """
        mean = scalogram.mean()
        factor = np.random.uniform(*self.contrast_range)
        augmented = mean + factor * (scalogram - mean)
        return np.clip(augmented, 0.0, 1.0).astype(np.float32)
    
    def _freq_jitter(self, scalogram: np.ndarray) -> np.ndarray:
        """
        Randomly shift along frequency axis (up to ±freq_jitter_pct). 
        
        This simulates slight variations in respiratory baseline or sensor placement
        without changing the actual respiratory frequency (RR is preserved).
        
        Why this preserves RR:
        - Respiratory frequency corresponds to HORIZONTAL patterns in the scalogram
        - Vertical (frequency axis) shifts don't change these patterns
        """
        n_freqs, n_times = scalogram.shape
        shift_amount = int(n_freqs * np.random.uniform(-self.freq_jitter_pct, self.freq_jitter_pct))
        
        if shift_amount == 0:
            return scalogram
        
        # Roll (circular shift) along frequency axis
        augmented = np.roll(scalogram, shift_amount, axis=0)
        
        return augmented.astype(np.float32)
    
    def _time_jitter(self, scalogram: np.ndarray) -> np. ndarray:
        """
        Randomly shift along time axis (up to ±time_jitter_pct).
        
        This is SAFE because we're shifting the entire signal, not stretching it.
        The time resolution remains 1 Hz (60 samples for 60s), so RR is preserved.
        
        Example:
        - Original: [breath 1] [breath 2] [breath 3]
        - After shift: [breath 2] [breath 3] [breath 1]
        - Still the same 3 breaths in 60s → RR unchanged ✓
        """
        n_freqs, n_times = scalogram.shape
        shift_amount = int(n_times * np. random.uniform(-self.time_jitter_pct, self. time_jitter_pct))
        
        if shift_amount == 0:
            return scalogram
        
        # Roll along time axis
        augmented = np.roll(scalogram, shift_amount, axis=1)
        
        return augmented.astype(np.float32)
    
    def _gaussian_blur(self, scalogram: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to smooth the scalogram.
        
        This reduces high-frequency noise while preserving the overall structure.
        Respiratory rate is encoded as TEMPORAL patterns (peaks separated by ~1-2 seconds),
        so moderate blur doesn't destroy this information.
        """
        sigma = np.random.uniform(*self.blur_sigma_range)
        
        # Apply Gaussian filter
        augmented = ndimage. gaussian_filter(scalogram, sigma=sigma)
        
        # Re-normalize to [0, 1] range (blur can slightly change min/max)
        augmented_min = augmented.min()
        augmented_max = augmented.max()
        
        if augmented_max > augmented_min:
            augmented = (augmented - augmented_min) / (augmented_max - augmented_min)
        
        return np.clip(augmented, 0.0, 1.0). astype(np.float32)
    
    def _rotation(self, scalogram: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
        """
        Apply minor rotation to the scalogram.
        
        ⚠️ WARNING: Rotation can distort frequency/time relationships.
        Used sparingly (small angles only) for robustness.
        
        Why it's OK with small angles:
        - ±5° rotation is imperceptible to the human eye
        - Frequency and time axes remain mostly aligned
        - Respiratory patterns (which are ~1-2D diagonal structures) are preserved
        """
        angle = np.random.uniform(*self.rotation_range)
        
        # Use cv2 for efficient rotation (handles interpolation better than scipy)
        h, w = scalogram.shape
        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        
        # Apply rotation with bilinear interpolation
        augmented = cv2.warpAffine(
            scalogram,
            rotation_matrix,
            (w, h),
            borderMode=cv2.BORDER_REFLECT,
            flags=cv2. INTER_LINEAR
        )
        
        return np.clip(augmented, 0.0, 1.0).astype(np. float32)
    
    def mixup(self, scalogram1: np.ndarray, scalogram2: np.ndarray) -> np.ndarray:
        """
        Blend two scalograms from the SAME RR class.
        
        This creates synthetic variations while preserving the RR label.
        
        Args:
            scalogram1, scalogram2: 2D scalograms from same RR class
            
        Returns:
            Blended scalogram
            
        Example:
            - RR class = 15 BPM
            - Scalogram A from subject 1 (15 BPM)
            - Scalogram B from subject 5 (15 BPM)
            - Mixup: 0.8 * A + 0.2 * B
            - Result: Synthetic 15 BPM with blended characteristics
        """
        alpha = np.random.uniform(0, self.mixup_alpha)
        
        # Ensure same shape
        if scalogram1. shape != scalogram2.shape:
            raise ValueError(f"Scalogram shapes don't match: {scalogram1. shape} vs {scalogram2. shape}")
        
        # Blend
        blended = (1 - alpha) * scalogram1 + alpha * scalogram2
        
        return np.clip(blended, 0.0, 1.0).astype(np.float32)
    
    def batch_augment(self, scalograms: np.ndarray) -> np.ndarray:
        """
        Augment a batch of scalograms.
        
        Args:
            scalograms: Array of shape (B, H, W) or (B, C, H, W)
            
        Returns:
            Augmented scalograms with same shape
        """
        if scalograms.ndim == 3:
            # (B, H, W)
            augmented = np.array([self.augment(s) for s in scalograms])
        elif scalograms.ndim == 4:
            # (B, C, H, W) - just augment the first 2D slice
            augmented = np.array([self.augment(s[0] if s.shape[0] == 1 else s) for s in scalograms])
        else:
            raise ValueError(f"Expected 3D or 4D array, got {scalograms.ndim}D")
        
        return augmented


def augment_scalogram_batch(freq_segments: List[np.ndarray], 
                            intensity: float = 0.1) -> List[np.ndarray]:
    """
    Wrapper function to augment a list of scalograms.
    
    Args:
        freq_segments: List of scalograms, each shape (128, 60)
        intensity: Augmentation intensity (0=no aug, 1=maximum)
        
    Returns:
        Augmented scalograms
    """
    augmentor = ScalogramAugmentor(
        intensity_range=(1 - intensity, 1 + intensity),
        contrast_range=(1 - intensity, 1 + intensity),
        freq_jitter_pct=0.02 * intensity,
        time_jitter_pct=0.02 * intensity,
        blur_sigma_range=(0.3, 0.8 * intensity),
        rotation_range=(-3 * intensity, 3 * intensity)
    )
    
    return [augmentor.augment(s) for s in freq_segments]