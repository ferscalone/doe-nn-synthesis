"""Физическая модель: дифракция Френеля «с одним преобразованием Фурье».

K(x,y) = exp(jkz)/(j*lambda*z) * exp(j*pi/(lambda*z)*(x^2+y^2))
         * FFT{ H(u,v) * exp(j*pi/(lambda*z)*(u^2+v^2)) }

H(u,v) = exp(j*phi(u,v)) - поле сразу за фазовым ДОЭ (освещение плоской волной).
Множитель перед FFT и выходная квадратичная фаза не меняют модуль |K|,
поэтому при обучении считаем только |FFT{H * Q_in}|; в numpy-версии
(fresnel_numpy) формула реализована полностью, как на слайде.
"""
import math

import numpy as np
import torch

# ---- Параметры задачи -------------------------------------------------------
N = 128                      # размер ДОЭ и изображений, пикселов
DX = 10e-6                   # шаг пиксела ПВМС, м
WAVELENGTH = 532e-9          # длина волны, м
DISTANCES = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)   # плоскости восстановления, м

# Информативные области (строка, столбец). В задании написано [0:61], [66:127];
# в выданных картинках ненулевые пиксели занимают индексы 0..61 и 66..127
# включительно (62 px), поэтому по умолчанию берём включительные границы.
ROI_INCLUSIVE = [
    (slice(0, 62), slice(0, 62)),
    (slice(0, 62), slice(66, 128)),
    (slice(66, 128), slice(0, 62)),
    (slice(66, 128), slice(66, 128)),
]
# Буквальное прочтение как Python-срезы (61 px). Считаем метрики и так, для сверки.
ROI_LITERAL = [
    (slice(0, 61), slice(0, 61)),
    (slice(0, 61), slice(66, 127)),
    (slice(66, 127), slice(0, 61)),
    (slice(66, 127), slice(66, 127)),
]
ROI = ROI_INCLUSIVE
ROI_SIZE = 62


def output_pixel_pitch(z, n=N, dx=DX, wl=WAVELENGTH):
    """Шаг пиксела в плоскости восстановления для метода с одним БПФ."""
    return wl * z / (n * dx)


def chirp_nyquist_ratio(z, n=N, dx=DX, wl=WAVELENGTH):
    """>= 1 означает, что квадратичная фаза exp(j*pi*r^2/(lambda*z)) дискретизована
    без наложения спектров (макс. набег фазы между соседними пикселами <= pi)."""
    return wl * z / (n * dx * dx)


# ---- Кодирование фазы в 8-битное изображение ---------------------------------
def wrap_phase(phi):
    """Приводит фазу к диапазону [-pi, pi)."""
    if isinstance(phi, torch.Tensor):
        return torch.remainder(phi + math.pi, 2 * math.pi) - math.pi
    return np.mod(phi + np.pi, 2 * np.pi) - np.pi


def phase_to_uint8(phi):
    """-pi -> 0, +pi -> 255 (как требует задание)."""
    phi = wrap_phase(np.asarray(phi, dtype=np.float64))
    return np.clip(np.round((phi + np.pi) / (2 * np.pi) * 255.0), 0, 255).astype(np.uint8)


def uint8_to_phase(img):
    return np.asarray(img, dtype=np.float64) / 255.0 * 2 * np.pi - np.pi


# ---- Эталонная numpy-реализация (полная формула со слайда) --------------------
def fresnel_numpy(H, z, dx=DX, wl=WAVELENGTH, fftshift=True):
    """Полная формула Френеля с одним БПФ. H - комплексное поле (n, n)."""
    n = H.shape[-1]
    k = 2 * np.pi / wl
    c = (np.arange(n) - n / 2) * dx
    u, v = np.meshgrid(c, c, indexing="ij")
    dxo = wl * z / (n * dx)                       # шаг в плоскости восстановления
    co = (np.arange(n) - n / 2) * dxo
    x, y = np.meshgrid(co, co, indexing="ij")
    q_in = np.exp(1j * np.pi / (wl * z) * (u ** 2 + v ** 2))
    spec = np.fft.fft2(np.fft.ifftshift(H * q_in))
    if fftshift:
        spec = np.fft.fftshift(spec)
    pref = np.exp(1j * k * z) / (1j * wl * z)
    q_out = np.exp(1j * np.pi / (wl * z) * (x ** 2 + y ** 2))
    return pref * q_out * spec * dx * dx          # dx^2 - мера интегрирования


def reconstruct_numpy(phi, distances=DISTANCES, fftshift=True):
    """Модули восстановленных полей для всех плоскостей: (P, n, n)."""
    H = np.exp(1j * np.asarray(phi, dtype=np.float64))
    return np.stack([np.abs(fresnel_numpy(H, z, fftshift=fftshift)) for z in distances])


# ---- Дифференцируемая torch-реализация (для обучения) ------------------------
class Fresnel(torch.nn.Module):
    """Прямое и обратное распространение ДОЭ -> P плоскостей (батчево, на GPU)."""

    def __init__(self, n=N, dx=DX, wl=WAVELENGTH, distances=DISTANCES, fftshift=True):
        super().__init__()
        self.fftshift = fftshift
        self.n_planes = len(distances)
        c = (torch.arange(n, dtype=torch.float64) - n / 2) * dx
        u, v = torch.meshgrid(c, c, indexing="ij")
        r2 = u ** 2 + v ** 2
        q = torch.stack([torch.exp(1j * math.pi / (wl * z) * r2) for z in distances]).to(torch.complex64)
        # Храним как вещественный (P, n, n, 2): так буфер надёжно копируется между
        # несколькими GPU; в чекпойнт не пишется (это константа, считается заново).
        self.register_buffer("q_ri", torch.view_as_real(q).contiguous(), persistent=False)

    @property
    def q(self):
        return torch.view_as_complex(self.q_ri)                   # (P, n, n) complex64

    def _fft(self, x):
        x = torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho")
        return torch.fft.fftshift(x, dim=(-2, -1)) if self.fftshift else x

    def _ifft(self, x):
        if self.fftshift:
            x = torch.fft.ifftshift(x, dim=(-2, -1))
        return torch.fft.fftshift(torch.fft.ifft2(x, norm="ortho"), dim=(-2, -1))

    def forward_field(self, h):
        """h: комплексное поле ДОЭ (B, n, n) -> поля в плоскостях (B, P, n, n)."""
        return self._fft(h.unsqueeze(1) * self.q)

    def forward_phase(self, phi):
        """phi: (B, n, n) -> модули полей |K| (B, P, n, n)."""
        return self.forward_field(torch.polar(torch.ones_like(phi), phi)).abs()

    def backward_field(self, k):
        """Обратное распространение: поля в плоскостях (B, P, n, n) -> поля в
        плоскости ДОЭ, по отдельности для каждой плоскости (B, P, n, n)."""
        return self._ifft(k) * self.q.conj()


def roi_crop(x, rois=ROI):
    """x: (..., P, n, n) -> (..., P, h, w): из плоскости k берём область k."""
    return torch.stack([x[..., k, r, c] for k, (r, c) in enumerate(rois)], dim=-3)


def roi_crop_np(x, rois=ROI):
    return np.stack([x[k][r, c] for k, (r, c) in enumerate(rois)])
