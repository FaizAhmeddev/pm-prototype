"""Real-data check on NASA IMS bearings. Run: python -m ml.ims_demo <folder with timestamped files>
Fits anomaly detection on the first 20% of files (healthy period) and reports when scores leave that baseline."""
import glob, sys
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from ml.core import N, features

files = sorted(glob.glob(sys.argv[1] + "/20*"))
if len(files) < 50:
    sys.exit("Too few files. Extract the IMS archive (it contains a nested .7z) and pass the test-set folder.")
F = np.array([features(np.loadtxt(f, usecols=0, max_rows=N))[:9] for f in files])   # vibration features only
k = len(F) // 5
sc = StandardScaler().fit(F[:k])
iso = IsolationForest(n_estimators=200, random_state=0).fit(sc.transform(F[:k]))
s = -iso.score_samples(sc.transform(F))
thr = s[:k].max()
hit = next((i for i in range(k, len(s) - 4) if (s[i:i + 5] > thr).all()), None)
print(f"{len(files)} files, baseline threshold {thr:.3f}")
print("Sustained anomaly starts at file index:", hit, "of", len(files) - 1)
