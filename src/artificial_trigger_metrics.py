"""Image-only SSIM reporting for the existing artificial-trigger test cohort.

Compare full images after reversing Download_data's normalization. Use the
standard Gaussian-window SSIM (sigma=1.5, 11x11 window, population covariance)
with data_range=1.0. Triggered values are deliberately not clipped: this
measures the actual injected inputs, including values outside the image range.
"""

import numpy as np


class ArtificialTriggerSSIM:
    _NORMALIZATION = {
        'MNIST': ((0.1307,), (0.3081,)),
        'FashionMNIST': ((0.5,), (0.5,)),
        'CIFAR10': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        'CIFAR100': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        'ImageNet': ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        'GTSRB': ((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
    }

    def __init__(self, dataset):
        self.dataset = dataset
        self.count = 0
        self.total = 0.0
        self.supported = dataset in self._NORMALIZATION
        if self.supported:
            # Import only when the pre-training image report is requested.
            from skimage.metrics import structural_similarity
            self._ssim = structural_similarity
            mean, std = self._NORMALIZATION[dataset]
            self._mean = np.asarray(mean, dtype=np.float64)[:, None, None]
            self._std = np.asarray(std, dtype=np.float64)[:, None, None]

    def update(self, original, triggered):
        """Add one already-selected sample; never change tensors or RNG state."""
        if not self.supported:
            self.count += 1
            return
        before = original.detach().cpu().numpy().astype(np.float64)
        after = triggered.detach().cpu().numpy().astype(np.float64)
        if (before.shape != after.shape or before.ndim != 3
                or before.shape[0] != len(self._mean) or min(before.shape[-2:]) < 11):
            raise ValueError('Image SSIM requires matching CHW images with height/width >= 11')
        before = before * self._std + self._mean
        after = after * self._std + self._mean
        value = self._ssim(
            before, after, data_range=1.0, channel_axis=0,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        self.total += float(value)
        self.count += 1

    @property
    def mean(self):
        return self.total / self.count if self.supported and self.count else None

    def summary(self, strength):
        value = 'N/A' if self.mean is None else f'{self.mean:.6f}'
        text = (f'Artificial Trigger SSIM | stage: before_training | dataset: {self.dataset} | '
                f'strength: {strength:g} | mean_ssim: {value} | samples: {self.count}')
        if not self.supported:
            text += ' | reason: SSIM applies to image datasets only'
        elif not self.count:
            text += ' | reason: no eligible artificial-trigger test samples'
        return text
