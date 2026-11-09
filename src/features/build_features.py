import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FeatureEngineer:
    """
    Clase modular para la ingeniería de características y reducción de dimensionalidad,
    parametrizada mediante config.yaml para evitar valores fijos (hardcoding).
    """
    def __init__(self, config_path="config.yaml"):
        # Cargar parámetros desde el archivo de configuración
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        self.pca_comp = self.config['model']['pca_components']
        self.random_state = self.config['model']['random_state']
        self.col_id = self.config['model']['columnas_id']
        self.col_eliminar = self.config['model']['columnas_eliminar']
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_comp, random_state=self.random_state)

    def limpiar_columnas_inutiles(self, df):
        """Elimina columnas vacías o no deseadas definidas en el YAML."""
        return df.drop(columns=[col for col in self.col_eliminar if col in df.columns])

    def aplicar_pca(self, df):
        """Aplica escalado y PCA separando los identificadores de las matemáticas."""
        df_features = df.drop(columns=[col for col in self.col_id if col in df.columns])
        # Imputación de nulos numéricos (protección contra datos crudos)
        df_features = df_features.fillna(df_features.median(numeric_only=True))
        
        datos_escalados = self.scaler.fit_transform(df_features.select_dtypes(include='number'))
        datos_pca = self.pca.fit_transform(datos_escalados)
        
        df_final = pd.DataFrame(datos_pca, columns=[f"PC{i+1}" for i in range(datos_pca.shape[1])])
        
        for col in self.col_id:
            if col in df.columns:
                df_final[col] = df[col].values
                
        return df_final