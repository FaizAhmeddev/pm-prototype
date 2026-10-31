"""Signal simulation, feature extraction and the three-model prediction + health fusion."""
import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis, skew

FS, N, HORIZON = 20480, 2048, 60.0            # sample rate (Hz), window length, RUL cap (days)
FAULTS = ["normal", "inner", "outer", "ball", "imbalance"]
NAMES = ["rms", "peak", "kurtosis", "skew", "crest", "band_0_200", "band_200_2k",
         "band_2k_8k", "env_peak_hz", "bearing_temp", "oil_pressure"]
_RATE = {"inner": 5.4, "outer": 3.6, "ball": 2.4}   # fault frequency / shaft frequency


def simulate_window(fault="normal", sev=0.0, rpm=1800, rng=None):
    rng = rng or np.random.default_rng()
    t = np.arange(N) / FS
    f = rpm / 60
    x = 0.5 * np.sin(2 * np.pi * f * t) + 0.1 * rng.standard_normal(N)
    if fault == "imbalance":
        x += (0.3 + 2.5 * sev) * np.sin(2 * np.pi * f * t + 0.3)
    elif fault in _RATE:
        h = np.zeros(N)
        h[:: int(FS / (_RATE[fault] * f))] = 1.0
        k = np.arange(200) / FS
        ring = np.exp(-k * 1500) * np.sin(2 * np.pi * 3000 * k)
        x += (0.3 + 3 * sev) * np.convolve(h, ring)[:N]
    return x


def process_tags(sev, rng):
    return 65 + 25 * sev + rng.normal(0, 1), 4.0 - 1.2 * sev + rng.normal(0, 0.05)


def features(x, temp=65.0, oil=4.0):
    x = np.asarray(x, float)
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fr = np.fft.rfftfreq(len(x), 1 / FS)
    tot = sp.sum() + 1e-12
    b = [sp[(fr >= lo) & (fr < hi)].sum() / tot for lo, hi in ((0, 200), (200, 2000), (2000, 8000))]
    e = np.abs(hilbert(x))
    es = np.abs(np.fft.rfft(e - e.mean()))
    m = (fr >= 40) & (fr <= 400)
    env = fr[m][es[m].argmax()]
    pk = np.abs(x).max()
    return [rms, pk, kurtosis(x), skew(x), pk / rms, *b, env, temp, oil]


def predict(b, x, temp, oil):
    f = np.array([features(x, temp, oil)])
    s = -b["iso"].score_samples(b["scaler"].transform(f))[0]
    anomaly = float(np.clip((s - b["lo"]) / (b["hi"] - b["lo"]), 0, 1))
    proba = b["clf"].predict_proba(f)[0]
    classes = list(b["clf"].classes_)
    i = int(proba.argmax())
    rul = float(np.clip(b["rul"].predict(f)[0], 0, HORIZON))
    hw = 1.64 * b["sigma"]
    health = 100 * (0.2 * anomaly + 0.2 * (1 - float(proba[classes.index("normal")])) + 0.6 * (1 - rul / HORIZON))
    stage = int(health >= 25) + int(health >= 50) + int(health >= 75)
    return dict(anomaly=round(anomaly, 3), fault=str(classes[i]), fault_conf=round(float(proba[i]), 3),
                rul=round(rul, 1), rul_lo=round(max(0.0, rul - hw), 1), rul_hi=round(min(HORIZON, rul + hw), 1),
                health=round(health, 1), stage=stage)
