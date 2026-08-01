import os
from pathlib import Path
import pandas as pd


class Olist:
    def get_data(self):
        """
        Returns a dict: keys = 'orders', 'order_items'... , values = DataFrames
        """
        # Denenecek olası csv klasörleri (ilk var olanı kullan)
        candidates = [
            Path.home() / ".workintech" / "olist" / "data" / "csv",
            Path(__file__).resolve().parents[1] / "data" / "csv",
        ]

        csv_path = None
        for c in candidates:
            if c.is_dir() and any(c.glob("*.csv")):
                csv_path = c
                break

        if csv_path is None:
            tried = "\n  ".join(str(c) for c in candidates)
            raise FileNotFoundError(
                "CSV klasörü bulunamadı. Denenen yollar:\n  " + tried
            )

        file_names = [f for f in os.listdir(csv_path) if f.endswith(".csv")]

        key_names = [
            f.replace("olist_", "")
             .replace("_dataset", "")
             .replace(".csv", "")
            for f in file_names
        ]

        data = {
            key: pd.read_csv(os.path.join(csv_path, file))
            for key, file in zip(key_names, file_names)
        }

        return data

    def ping(self):
        print("pong")