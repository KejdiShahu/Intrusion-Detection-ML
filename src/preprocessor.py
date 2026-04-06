import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


class IoTPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_cols = None
        self.fitted = False

    def clean(self, df: pd.DataFrame):
        df = df.copy()

        df = df.drop_duplicates()

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        cols_to_drop = [df.columns[0], "id.orig_p", "id.resp_p"]
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        return df

    def treat_outliers(self, df: pd.DataFrame):
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper)

        return df

    def create_target(self, df: pd.DataFrame):
        df = df.copy()

        normal_classes = ["Thing_Speak", "MQTT_Publish", "Wipro_bulb"]
        df["is_attack"] = df["Attack_type"].apply(
            lambda x: 0 if x in normal_classes else 1
        )

        return df

    def encode(self, df: pd.DataFrame, fit=True):
        df = df.copy()

        for col in ["proto", "service"]:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))

        return df

    def fit(self, df: pd.DataFrame):
        df = self.clean(df)
        df = self.treat_outliers(df)
        df = self.create_target(df)
        df = self.encode(df, fit=True)

        X = df.drop(columns=["is_attack", "Attack_type"], errors="ignore")

        self.feature_cols = X.columns
        self.scaler.fit(X)
        self.fitted = True

        return self

    def transform(self, df: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("Call fit() before transform().")

        df = self.clean(df)
        df = self.treat_outliers(df)
        df = self.create_target(df)
        df = self.encode(df, fit=False)

        X = df.drop(columns=["is_attack", "Attack_type"], errors="ignore")
        y = df["is_attack"]

        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_cols)

        return X_scaled, y

    def fit_transform_split(self, df: pd.DataFrame, test_size=0.2):
        self.fit(df)

        df = self.clean(df)
        df = self.treat_outliers(df)
        df = self.create_target(df)
        df = self.encode(df, fit=False)

        X = df.drop(columns=["is_attack", "Attack_type"], errors="ignore")
        y = df["is_attack"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train), columns=self.feature_cols
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), columns=self.feature_cols
        )

        return X_train_scaled, X_test_scaled, y_train, y_test
