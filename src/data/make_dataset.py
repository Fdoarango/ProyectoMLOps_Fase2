import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class MiBiciDataLoader:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        logging.info(f"Cargando datos desde: {filepath}")
        df = pd.read_csv(filepath)
        cifras_iniciales = len(df)
        
        # Parseo de fechas
        date_col = self.config["data"]["datetime_column"]
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
        
        # Eliminación de duplicados y valores irrelevantes
        df = df.drop_duplicates()
        cifras_finales = len(df)
        logging.info(f"Registros antes: {cifras_iniciales} | Registros después: {cifras_finales}")
        return df

    def split_temporal_windows(self, df: pd.DataFrame):
        """Genera particiones de validación temporal mediante ventanas sucesivas."""
        date_col = self.config["data"]["datetime_column"]
        df = df.sort_values(by=date_col).reset_index(drop=True)
        
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        
        train_months = self.config["temporal_validation"]["train_window_months"]
        test_months = self.config["temporal_validation"]["test_window_months"]
        step_months = self.config["temporal_validation"]["step_months"]
        
        current_cutoff = min_date + pd.DateOffset(months=train_months)
        windows = []
        
        while current_cutoff + pd.DateOffset(months=test_months) <= max_date:
            train_start = current_cutoff - pd.DateOffset(months=train_months)
            test_end = current_cutoff + pd.DateOffset(months=test_months)
            
            train_mask = (df[date_col] >= train_start) & (df[date_col] < current_cutoff)
            test_mask = (df[date_col] >= current_cutoff) & (df[date_col] < test_end)
            
            windows.append((df[train_mask].copy(), df[test_mask].copy(), current_cutoff))
            current_cutoff += pd.DateOffset(months=step_months)
            
        logging.info(f"Se crearon {len(windows)} ventanas temporales sucesivas.")
        return windows