import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


class IoTBalancer:

    def __init__(self, strategy: str = "smote", random_state: int = 42):
        self.strategy     = strategy
        self.random_state = random_state
        self.original_dist = {}
        self.balanced_dist = {}

    def fit_resample(self, X_train, y_train):


        unique, counts = np.unique(y_train, return_counts=True)
        self.original_dist = dict(zip(unique, counts))

        print("=" * 55)
        print("  BALANCING TRAINING DATA")
        print("=" * 55)
        print(f"  Before → Normal: {self.original_dist.get(0,0):,} "
              f"| Attack: {self.original_dist.get(1,0):,}")

        if self.strategy == "smote":
            X_res, y_res = self._smote(X_train, y_train)

        elif self.strategy == "undersample":
            X_res, y_res = self._undersample(X_train, y_train)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")


        unique2, counts2 = np.unique(y_res, return_counts=True)
        self.balanced_dist = dict(zip(unique2, counts2))

        print(f"  After  → Normal: {self.balanced_dist.get(0,0):,} "
              f"| Attack: {self.balanced_dist.get(1,0):,}")
        print(f"  Strategy used   : {self.strategy.upper()}")
        print("=" * 55)

        return X_res, y_res

    def _smote(self, X_train, y_train):
        sm = SMOTE(random_state=self.random_state)
        return sm.fit_resample(X_train, y_train)

    def _undersample(self, X_train, y_train):
        min_count = min(self.original_dist.values())
        idx_keep  = []
        for cls in np.unique(y_train):
            cls_idx = np.where(y_train == cls)[0]
            chosen  = np.random.RandomState(self.random_state).choice(
                cls_idx, size=min_count, replace=False
            )
            idx_keep.extend(chosen)
        idx_keep = np.array(idx_keep)
        return X_train[idx_keep], y_train[idx_keep]

    def summary(self):
        if not self.original_dist or not self.balanced_dist:
            print("No balancing has been performed yet.")
            return
        total_before = sum(self.original_dist.values())
        total_after  = sum(self.balanced_dist.values())
        ratio_before = self.original_dist.get(0,0) / self.original_dist.get(1,1)
        ratio_after  = self.balanced_dist.get(0,0) / self.balanced_dist.get(1,1)
        print(f"  Total samples before : {total_before:,}")
        print(f"  Total samples after  : {total_after:,}")
        print(f"  Normal:Attack ratio before : {ratio_before:.3f}")
        print(f"  Normal:Attack ratio after  : {ratio_after:.3f}")