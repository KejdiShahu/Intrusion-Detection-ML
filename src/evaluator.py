import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


class IoTEvaluator:
    def __init__(self, save_dir: str = "results/evaluation"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.results = []

    def evaluate(self, name, model, X_test, y_test):
        y_pred = model.predict(X_test)

        try:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)
            auc = roc_auc_score(y_test, y_score)
        except Exception:
            auc = float("nan")

        result = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "ROC-AUC": auc,
            "y_pred": y_pred,
        }
        self.results.append(result)

        print(f"\n  ── {name} ──")
        print(
            classification_report(
                y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0
            )
        )
        return result

    def evaluate_all(self, models: dict, X_test, y_test):
        for name, model in models.items():
            self.evaluate(name, model, X_test, y_test)
        return self.get_summary()

    def get_summary(self) -> pd.DataFrame:
        rows = [{k: v for k, v in r.items() if k != "y_pred"} for r in self.results]
        df = pd.DataFrame(rows).set_index("Model")
        df.round(4).to_csv(f"{self.save_dir}/summary.csv")
        print(f"\n  Summary saved → {self.save_dir}/summary.csv")
        return df

    def plot(
            self, y_test, feature_importances: pd.Series = None, loss_curve: list = None
    ):

        fig = plt.figure(figsize=(20, 18))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)
        fig.suptitle(
            "Phase 3 — Model Evaluation", fontsize=16, fontweight="bold", y=0.99
        )

        colors = ["#3498db", "#e74c3c", "#2ecc71"]

        for idx, r in enumerate(self.results):
            ax = fig.add_subplot(gs[0, idx])
            cm = confusion_matrix(y_test, r["y_pred"])
            pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
            sns.heatmap(
                pct,
                annot=True,
                fmt=".1f",
                cmap="Blues",
                ax=ax,
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"],
                cbar=False,
                linewidths=0.5,
            )
            ax.set_title(
                f"{r['Model']}\nConfusion Matrix (%)", fontweight="bold", fontsize=11
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
        ax4 = fig.add_subplot(gs[1, :2])
        x = np.arange(len(metrics))
        w = 0.25
        for i, (r, col) in enumerate(zip(self.results, colors)):
            ax4.bar(
                x + i * w,
                [r[m] for m in metrics],
                w,
                label=r["Model"],
                color=col,
                alpha=0.85,
            )
        ax4.set_xticks(x + w)
        ax4.set_xticklabels(metrics)
        ax4.set_ylim(0.7, 1.02)
        ax4.set_title("Metric Comparison", fontweight="bold")
        ax4.set_ylabel("Score")
        ax4.legend()
        ax4.grid(axis="y", alpha=0.3)

        if feature_importances is not None:
            ax5 = fig.add_subplot(gs[1, 2])
            top = feature_importances.head(12)
            ax5.barh(top.index[::-1], top.values[::-1], color="#3498db", alpha=0.8)
            ax5.set_title(
                "Top 12 Features\n(Decision Tree)", fontweight="bold", fontsize=11
            )
            ax5.set_xlabel("Importance")

        if loss_curve is not None:
            ax6 = fig.add_subplot(gs[2, :2])
            ax6.plot(loss_curve, color="#2ecc71", linewidth=2)
            ax6.set_title("Neural Network — Loss Curve", fontweight="bold")
            ax6.set_xlabel("Epoch")
            ax6.set_ylabel("Loss")
            ax6.grid(alpha=0.3)

        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis("off")
        data = [
            [
                r["Model"],
                f"{r['Accuracy']:.4f}",
                f"{r['F1']:.4f}",
                f"{r['ROC-AUC']:.4f}",
            ]
            for r in self.results
        ]
        tbl = ax7.table(
            cellText=data,
            colLabels=["Model", "Acc", "F1", "AUC"],
            cellLoc="center",
            loc="center",
            colWidths=[0.40, 0.20, 0.20, 0.20],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 2)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor("#1F3864")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#EBF3FB")

        path = f"{self.save_dir}/evaluation.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot saved → {path}")
