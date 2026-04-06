import pandas as pd
from sklearn.preprocessing import StandardScaler


class IoTDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_data(self):
        """Read the CSV and perform initial cleanup."""

        self.df = pd.read_csv(self.file_path)

        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df
