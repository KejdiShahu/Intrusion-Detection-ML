import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew

warnings.filterwarnings("ignore", category=FutureWarning)


class IoTEda:
    def __init__(self):
        self.label_encoders = {}
        self.feature_cols = None
        self.fitted = False

    def run_integrated_analysis(self, df, output_dir="./results/eda"):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")

        print("\n" + "=" * 60)
        print("PHASE 1: FULL INTEGRATED EXPLORATORY DATA ANALYSIS")
        print("=" * 60)

        normal_classes = ["Thing_Speak", "MQTT_Publish", "Wipro_bulb", "Amazon_Alexa"]
        df["Is_Attack"] = df["Attack_type"].apply(
            lambda x: 0 if x in normal_classes else 1
        )

        counts = df["Attack_type"].value_counts()

        print("\n[STATS] Class Distribution:")
        for cls, cnt in counts.items():
            pct = cnt / len(df) * 100
            label = "← NORMAL" if cls in normal_classes else "← ATTACK"
            print(f"  {cls:<35} {cnt:>6,}  ({pct:5.1f}%) {label}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(
            f"\n[METADATA] Features: {len(numeric_cols)} | Missing: {df.isnull().sum().sum()} | Duplicates: {df.duplicated().sum()}"
        )

        top_features = [
            "flow_duration",
            "fwd_pkts_tot",
            "bwd_pkts_tot",
            "fwd_pkts_payload.avg",
            "bwd_pkts_payload.avg",
            "flow_pkts_payload.avg",
            "fwd_iat.avg",
            "bwd_iat.avg",
            "idle.avg",
            "idle.std",
            "bwd_init_window_size",
        ]


        plt.figure(figsize=(12, 6))
        colors = ["#2ecc71" if x in normal_classes else "#e74c3c" for x in counts.index]
        bars = plt.bar(counts.index, counts.values, color=colors, edgecolor="white")
        plt.title(
            "Detailed Class Distribution\n(Green = Normal | Red = Attack)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xticks(rotation=45)
        plt.ylabel("Count")
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 500,
                f"{int(yval):,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        plt.tight_layout()
        plt.savefig(f"{output_dir}/eda_1_class_dist.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 8))
        normal_count = df[df["Attack_type"].isin(normal_classes)].shape[0]
        attack_count = len(df) - normal_count
        plt.pie(
            [normal_count, attack_count],
            labels=["Normal", "Attack"],
            autopct="%1.1f%%",
            colors=["#2ecc71", "#e74c3c"],
            startangle=90,
            explode=(0.05, 0.05),
        )
        plt.title("Normal vs Attack Traffic Ratio", fontsize=14, fontweight="bold")
        plt.savefig(f"{output_dir}/eda_2_pie_ratio.png", dpi=150)
        plt.close()

        df[top_features[:9]].hist(
            bins=30, figsize=(15, 10), color="#3498db", edgecolor="black"
        )
        plt.suptitle(
            "Univariate Analysis: Feature Distribution Grid",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"{output_dir}/eda_3_hist_grid.png", dpi=150)
        plt.close()

        plt.figure(figsize=(15, 12))
        for i, col in enumerate(top_features[:6], 1):
            plt.subplot(3, 2, i)
            sns.boxplot(
                x="Is_Attack",
                y=col,
                data=df,
                palette="Set2",
                hue="Is_Attack",
                legend=False,
            )
            plt.title(f"{col} vs Traffic Type", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/eda_4_boxplots.png", dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x="fwd_pkts_payload.avg",
            y="bwd_pkts_payload.avg",
            hue="Is_Attack",
            alpha=0.4,
            palette="viridis",
        )
        plt.title(
            "Scatter Plot: Forward vs Backward Payload (Traffic Asymmetry)", fontsize=14
        )
        plt.savefig(f"{output_dir}/eda_5_scatter.png", dpi=150)
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.violinplot(
            x="Attack_type",
            y="flow_duration",
            data=df,
            palette="muted",
            hue="Attack_type",
            legend=False,
        )
        plt.yscale("log")
        plt.xticks(rotation=45)
        plt.title("Outlier & Density Detection (Log Scale Duration)", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/eda_6_outliers.png", dpi=150)
        plt.close()

        plt.figure(figsize=(14, 10))
        corr = df[top_features].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1
        )
        plt.title(
            "Correlation Heatmap — Key Network Features", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(f"{output_dir}/eda_7_correlation.png", dpi=150)
        plt.close()

        skew_vals = (
            df[top_features]
            .apply(lambda x: skew(x.dropna()))
            .sort_values(ascending=False)
        )
        print("\n[REPORT] Skewness Values:")
        print(skew_vals)

        with open(f"{output_dir}/skewness_report.txt", "w") as f:
            f.write("SKEWNESS ANALYSIS REPORT\n" + "=" * 25 + "\n")
            f.write(skew_vals.to_string())

        print(f"\n[DONE] All 8 pages saved in {output_dir}/")
