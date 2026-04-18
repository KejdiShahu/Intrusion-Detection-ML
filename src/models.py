import joblib
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


class IoTModels:
    def __init__(self, save_dir: str = "results/models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.models = {}
        self.results = {}

    def train_all(self, X_train, y_train):
        self._train_decision_tree(X_train, y_train)
        self._train_svm(X_train, y_train)
        self._train_neural_network(X_train, y_train)
        print("\n  ✓ All models trained and saved.")

    def _train_decision_tree(self, X_train, y_train):
        print("[1] Training Decision Tree …")
        model = DecisionTreeClassifier(
            max_depth=15, min_samples_leaf=10, class_weight="balanced", random_state=42
        )
        model.fit(X_train, y_train)
        self.models["Decision Tree"] = model
        joblib.dump(model, f"{self.save_dir}/decision_tree.pkl")
        print("    Saved → decision_tree.pkl")

    def _train_svm(self, X_train, y_train):
        print("[2] Training SVM … (subsampled to 40k)")
        idx = np.random.RandomState(42).choice(
            len(X_train), size=min(40000, len(X_train)), replace=False
        )
        model = SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
        model.fit(X_train[idx], y_train[idx])
        self.models["SVM"] = model
        joblib.dump(model, f"{self.save_dir}/svm.pkl")
        print("    Saved → svm.pkl")

    def _train_neural_network(self, X_train, y_train):
        print("[3] Training Neural Network …")
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate="adaptive",
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
            verbose=False,
        )
        model.fit(X_train, y_train)
        self.models["Neural Network"] = model
        joblib.dump(model, f"{self.save_dir}/neural_network.pkl")
        print("    Saved → neural_network.pkl")

    def load_all(self):
        for name, fname in [
            ("Decision Tree", "decision_tree.pkl"),
            ("SVM", "svm.pkl"),
            ("Neural Network", "neural_network.pkl"),
        ]:
            path = f"{self.save_dir}/{fname}"
            if os.path.exists(path):
                self.models[name] = joblib.load(path)
                print(f"    Loaded ← {fname}")

    def get_model(self, name: str):
        return self.models.get(name)

    def get_all_models(self) -> dict:
        return self.models
