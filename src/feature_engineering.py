import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PAPER_FEATURES = [
    "bwd_pkts_payload.avg",
    "flow_pkts_payload.avg",
    "bwd_iat.avg",
    "flow_iat.max",
    "bwd_init_window_size",
    "idle.avg",
    "idle.std",
    "fwd_last_window_size",
    "down_up_ratio",
    "fwd_header_size_max",
    "flow_ECE_flag_count",
    "fwd_pkts_payload.std",
    "bwd_pkts_payload.std",
    "flow_RST_flag_count",
    "bwd_pkts_payload.max",
    "flow_pkts_payload.max",
    "flow_pkts_payload.std",
    "flow_iat.std",
    "payload_bytes_per_second",
    "fwd_subflow_bytes",
    "bwd_subflow_bytes",
    "proto",
    "service",
]


class IoTFeatureEngineer:
    def __init__(self, paper_features: list = None):
        self.paper_features = (
            paper_features if paper_features is not None else PAPER_FEATURES
        )
        self.scaler = StandardScaler()
        self.final_features: list = []
        self.fitted = False

        self.group1: list = []
        self.group2: list = []
        self.group3: list = []
        self.group4: list = []
        self.group5: list = []
        self.group6: list = []
        self.group7: list = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        fe = self._engineer(df)
        self._resolve_final_features(fe)
        X = self._clip(fe[self.final_features].copy())
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), columns=self.final_features, index=df.index
        )
        self.fitted = True
        return X_scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Call fit_transform() before transform().")

        fe = self._engineer(df)
        X = self._clip(fe[self.final_features].copy())
        X_scaled = pd.DataFrame(
            self.scaler.transform(X), columns=self.final_features, index=df.index
        )
        return X_scaled

    def fit_transform_with_labels(
        self, df: pd.DataFrame, label_col: str = "is_attack"
    ) -> tuple:
        y = df[label_col].reset_index(drop=True) if label_col in df.columns else None
        X_scaled = self.fit_transform(df)
        return X_scaled, y

    def get_feature_names(self) -> list:
        return list(self.final_features)

    def summary(self) -> None:
        groups = [
            ("Group 1 — Packet Ratios", self.group1),
            ("Group 2 — Payload", self.group2),
            ("Group 3 — IAT Timing", self.group3),
            ("Group 4 — TCP Flags", self.group4),
            ("Group 5 — Header Size", self.group5),
            ("Group 6 — Active/Idle", self.group6),
            ("Group 7 — Bulk/Subflow", self.group7),
        ]
        all_engineered = sum((g for _, g in groups), [])
        print("=" * 65)
        print("  FEATURE ENGINEERING SUMMARY")
        print("=" * 65)
        for name, feats in groups:
            print(f"  {name:<30} → {len(feats)} features")
        print(f"\n  Total engineered features    : {len(all_engineered)}")
        print(f"  + Paper features             : {len(self.paper_features)}")
        print(f"  = Final feature set          : {len(self.final_features)}")
        print("=" * 65)

    def plot(self, df: pd.DataFrame, save_path: str = None) -> None:
        if save_path:
            path = Path(save_path)

        path.parent.mkdir(parents=True, exist_ok=True)

        fe = self._engineer(df)

        if "is_attack" in fe.columns:
            label_col = "is_attack"
        elif "label" in fe.columns:
            label_col = "label"
        elif "Attack_type" in fe.columns:
            normal_classes = ["Thing_Speak", "MQTT_Publish", "Wipro_bulb"]
            fe["is_attack"] = fe["Attack_type"].apply(
                lambda x: 0 if x in normal_classes else 1
            )
            label_col = "is_attack"
        else:
            raise ValueError(
                "DataFrame must contain 'is_attack', 'label', or 'Attack_type' column."
            )

        COLORS = {"normal": "#2ecc71", "attack": "#e74c3c"}

        def split(col):
            return fe[col][fe[label_col] == 0], fe[col][fe[label_col] == 1]

        def clipped(series):
            return series.clip(upper=series.quantile(0.99))

        fig = plt.figure(figsize=(20, 24))
        gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.5, wspace=0.35)
        fig.suptitle(
            "RT-IoT2022 — Feature Engineering Analysis",
            fontsize=16,
            fontweight="bold",
            y=0.99,
        )

        plots = [
            (gs[0, 0], "pkt_ratio", "Packet Ratio (fwd/bwd)"),
            (gs[0, 1], "syn_ack_ratio", "SYN/ACK Ratio (DDoS Indicator)"),
            (gs[1, 0], "iat_jitter", "IAT Jitter (Timing Irregularity)"),
            (gs[1, 1], "active_idle_ratio", "Active/Idle Ratio (Slowloris Indicator)"),
            (gs[2, 0], "payload_variability", "Payload Variability (Std/Avg)"),
            (gs[2, 1], "rst_ratio", "RST Flag Ratio (Port Scan Indicator)"),
        ]

        for spec, col, title in plots:
            ax = fig.add_subplot(spec)
            n, a = split(col)
            ax.hist(
                clipped(n),
                bins=50,
                alpha=0.6,
                color=COLORS["normal"],
                label="Normal",
                density=True,
            )
            ax.hist(
                clipped(a),
                bins=50,
                alpha=0.6,
                color=COLORS["attack"],
                label="Attack",
                density=True,
            )
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.legend()

        ax = fig.add_subplot(gs[3, :])
        ax.axis("off")
        groups = [
            ("Group 1 — Packet Ratios", self.group1),
            ("Group 2 — Payload", self.group2),
            ("Group 3 — IAT", self.group3),
            ("Group 4 — TCP Flags", self.group4),
            ("Group 5 — Header Size", self.group5),
            ("Group 6 — Active/Idle", self.group6),
            ("Group 7 — Bulk/Subflow", self.group7),
        ]
        table_data = [[g, str(len(f)), ", ".join(f)] for g, f in groups]
        table = ax.table(
            cellText=table_data,
            colLabels=["Feature Group", "Count", "Features"],
            cellLoc="left",
            loc="center",
            colWidths=[0.22, 0.06, 0.72],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.8)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#1F3864")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#EBF3FB")
        ax.set_title(
            "Engineered Feature Groups Summary",
            fontweight="bold",
            fontsize=12,
            pad=10,
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved → {save_path}")
        else:
            plt.show()
        plt.close()

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        fe = df.copy()

        fe["pkt_ratio"] = fe["fwd_pkts_tot"] / (fe["bwd_pkts_tot"] + 1)
        fe["data_pkt_ratio"] = fe["fwd_data_pkts_tot"] / (fe["bwd_data_pkts_tot"] + 1)
        fe["pkt_rate_ratio"] = fe["fwd_pkts_per_sec"] / (fe["bwd_pkts_per_sec"] + 1)
        fe["total_pkts"] = fe["fwd_pkts_tot"] + fe["bwd_pkts_tot"]
        fe["pkt_imbalance"] = abs(fe["fwd_pkts_tot"] - fe["bwd_pkts_tot"]) / (
            fe["total_pkts"] + 1
        )
        self.group1 = [
            "pkt_ratio",
            "data_pkt_ratio",
            "pkt_rate_ratio",
            "total_pkts",
            "pkt_imbalance",
        ]

        fe["payload_ratio"] = fe["fwd_pkts_payload.avg"] / (
            fe["bwd_pkts_payload.avg"] + 1
        )
        fe["total_payload_bytes"] = (
            fe["fwd_pkts_payload.tot"] + fe["bwd_pkts_payload.tot"]
        )
        fe["payload_imbalance"] = abs(
            fe["fwd_pkts_payload.avg"] - fe["bwd_pkts_payload.avg"]
        ) / (fe["flow_pkts_payload.avg"] + 1)
        fe["payload_variability"] = fe["flow_pkts_payload.std"] / (
            fe["flow_pkts_payload.avg"] + 1
        )
        fe["bytes_per_pkt"] = fe["total_payload_bytes"] / (fe["total_pkts"] + 1)
        self.group2 = [
            "payload_ratio",
            "total_payload_bytes",
            "payload_imbalance",
            "payload_variability",
            "bytes_per_pkt",
        ]

        fe["iat_ratio"] = fe["fwd_iat.avg"] / (fe["bwd_iat.avg"] + 1)
        fe["iat_jitter"] = fe["flow_iat.std"] / (fe["flow_iat.avg"] + 1)
        fe["fwd_iat_jitter"] = fe["fwd_iat.std"] / (fe["fwd_iat.avg"] + 1)
        fe["bwd_iat_jitter"] = fe["bwd_iat.std"] / (fe["bwd_iat.avg"] + 1)
        fe["iat_range"] = fe["flow_iat.max"] - fe["flow_iat.min"]
        self.group3 = [
            "iat_ratio",
            "iat_jitter",
            "fwd_iat_jitter",
            "bwd_iat_jitter",
            "iat_range",
        ]

        fe["syn_ack_ratio"] = fe["flow_SYN_flag_count"] / (
            fe["flow_ACK_flag_count"] + 1
        )
        fe["rst_ratio"] = fe["flow_RST_flag_count"] / (fe["total_pkts"] + 1)
        fe["fin_ratio"] = fe["flow_FIN_flag_count"] / (fe["total_pkts"] + 1)
        fe["total_flags"] = (
            fe["flow_SYN_flag_count"]
            + fe["flow_ACK_flag_count"]
            + fe["flow_FIN_flag_count"]
            + fe["flow_RST_flag_count"]
            + fe["flow_CWR_flag_count"]
            + fe["flow_ECE_flag_count"]
        )
        fe["flag_density"] = fe["total_flags"] / (fe["total_pkts"] + 1)
        self.group4 = [
            "syn_ack_ratio",
            "rst_ratio",
            "fin_ratio",
            "total_flags",
            "flag_density",
        ]

        fe["fwd_header_ratio"] = fe["fwd_header_size_tot"] / (fe["fwd_pkts_tot"] + 1)
        fe["bwd_header_ratio"] = fe["bwd_header_size_tot"] / (fe["bwd_pkts_tot"] + 1)
        fe["header_size_diff"] = abs(
            fe["fwd_header_size_avg"]
            if "fwd_header_size_avg" in fe.columns
            else fe["fwd_header_size_max"] - fe["bwd_header_size_max"]
        )
        fe["total_header_bytes"] = fe["fwd_header_size_tot"] + fe["bwd_header_size_tot"]
        self.group5 = [
            "fwd_header_ratio",
            "bwd_header_ratio",
            "header_size_diff",
            "total_header_bytes",
        ]

        fe["active_idle_ratio"] = fe["active.avg"] / (fe["idle.avg"] + 1)
        fe["idle_range"] = fe["idle.max"] - fe["idle.min"]
        fe["active_range"] = fe["active.max"] - fe["active.min"]
        fe["idle_variability"] = fe["idle.std"] / (fe["idle.avg"] + 1)
        fe["flow_efficiency"] = fe["active.tot"] / (fe["flow_duration"] + 1)
        self.group6 = [
            "active_idle_ratio",
            "idle_range",
            "active_range",
            "idle_variability",
            "flow_efficiency",
        ]

        fe["bulk_ratio"] = fe["fwd_bulk_bytes"] / (fe["bwd_bulk_bytes"] + 1)
        fe["bulk_pkt_ratio"] = fe["fwd_bulk_packets"] / (fe["bwd_bulk_packets"] + 1)
        fe["total_bulk_bytes"] = fe["fwd_bulk_bytes"] + fe["bwd_bulk_bytes"]
        fe["subflow_pkt_ratio"] = fe["fwd_subflow_pkts"] / (fe["bwd_subflow_pkts"] + 1)
        fe["subflow_byte_ratio"] = fe["fwd_subflow_bytes"] / (
            fe["bwd_subflow_bytes"] + 1
        )
        self.group7 = [
            "bulk_ratio",
            "bulk_pkt_ratio",
            "total_bulk_bytes",
            "subflow_pkt_ratio",
            "subflow_byte_ratio",
        ]

        return fe

    def _resolve_final_features(self, fe: pd.DataFrame) -> None:
        all_engineered = (
            self.group1
            + self.group2
            + self.group3
            + self.group4
            + self.group5
            + self.group6
            + self.group7
        )
        combined = list(dict.fromkeys(self.paper_features + all_engineered))
        self.final_features = [f for f in combined if f in fe.columns]

    @staticmethod
    def _clip(X: pd.DataFrame) -> pd.DataFrame:
        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].clip(
                lower=X[col].quantile(0.01),
                upper=X[col].quantile(0.99),
            )
        X.fillna(0, inplace=True)
        return X
