import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import itertools
from pathlib import Path

class MiBiciPipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file) or {}
            
        self.random_state = (
            self.config.get('random_state') or 
            self.config.get('project', {}).get('random_state') or 
            self.config.get('model', {}).get('random_state', 42)
        )
        self.pca_comp = self.config.get('model', {}).get('pca_components', 2)
        self.mejor_pipeline = None
        
    def crear_pipeline(self, pca_components=2, n_estimators=100, contamination='auto'):
        return Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_components, random_state=self.random_state)),
            ('anomaly_detector', IsolationForest(
                n_estimators=n_estimators, 
                contamination=contamination, 
                random_state=self.random_state))
        ])

    def optimizar_hiperparametros(self, X):
        print("Iniciando ajuste de hiperparámetros...")
        
        tuning_config = self.config.get('tuning', {})
        pca_list = tuning_config.get('pca_components_list', [self.pca_comp])
        n_est_list = tuning_config.get('n_estimators_list', [100])
        cont_list = tuning_config.get('contamination_list', ['auto'])
        
        mejor_score = -999.0
        mejores_params = {}
        
        for pca_c, n_est, cont in itertools.product(pca_list, n_est_list, cont_list):
            pipeline_temp = self.crear_pipeline(pca_components=pca_c, n_estimators=n_est, contamination=cont)
            pipeline_temp.fit(X)
            predicciones = pipeline_temp.predict(X)
            
            score = silhouette_score(X, predicciones, sample_size=3000, random_state=self.random_state)
            anomalias_temp = (predicciones == -1).sum()
            
            print(f"Test -> PCA: {pca_c} | n_est: {n_est} | cont: {cont} => Silhouette: {score:.4f} | Anomalías: {anomalias_temp}", flush=True)
            
            if score > mejor_score:
                mejor_score = score
                mejores_params = {'pca_components': pca_c, 'n_estimators': n_est, 'contamination': cont}
                self.mejor_pipeline = pipeline_temp
                
        print(f"\nMejores hiperparámetros encontrados: {mejores_params} con Score: {mejor_score:.4f}")
        return self.mejor_pipeline, mejores_params

    def evaluar_modelo(self, X):
        if not self.mejor_pipeline:
            raise ValueError("Primero debes ejecutar optimizar_hiperparametros().")
            
        predicciones = self.mejor_pipeline.predict(X)
        score_final = silhouette_score(X, predicciones, sample_size=5000, random_state=self.random_state)
        anomalias = (predicciones == -1).sum()
        
        print("\n--- REPORTE DE EVALUACIÓN ---")
        print(f"Silhouette Score: {score_final:.4f}")
        print(f"Anomalías detectadas: {anomalias}")
        return score_final, anomalias

    def guardar_modelo(self, ruta_guardado="models/pipeline_optimizado_mibici.pkl"):
        Path(ruta_guardado).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.mejor_pipeline, ruta_guardado)
        print(f"\nPipeline optimizado guardado en: {ruta_guardado}")


if __name__ == "__main__":
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Proyecto_MiBici_Deteccion_Anomalias_v1")

    print("Cargando datos para el entrenamiento...")
    
    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "raw" / "datos_abiertos_2015_01.csv"
    
    df = pd.read_csv(data_path)
    
    # Selección robusta de columnas numéricas para evitar errores de PCA
    columnas_a_usar = ["duracion_minutos", "distancia_km", "edad_usuario"]
    existentes = [col for col in columnas_a_usar if col in df.columns]
    
    if len(existentes) < 2:
        X = df.select_dtypes(include=['float64', 'int64'])
    else:
        X = df[existentes]
        
    X = X.fillna(X.median())
    
    if X.shape[1] < 2:
        raise ValueError(f"Se detectaron solo {X.shape[1]} columnas numéricas. El PCA requiere al menos 2 características.")

    with mlflow.start_run(run_name="Run_Entrenamiento_Script") as run:
        trainer = MiBiciPipeline(config_path=str(project_root / "config.yaml"))
        
        mejor_pipe, mejores_params = trainer.optimizar_hiperparametros(X)
        score_eval, total_anomalias = trainer.evaluar_modelo(X)
        
        mlflow.log_params(mejores_params)
        mlflow.log_metric("silhouette_score", score_eval)
        mlflow.log_metric("anomalias_detectadas", total_anomalias)
        
        mlflow.sklearn.log_model(
            sk_model=mejor_pipe,
            artifact_path="modelo_isolation_forest",
            registered_model_name="MiBici_IsolationForest_Model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        
        model_output_path = project_root / "models" / "pipeline_optimizado_mibici.pkl"
        trainer.guardar_modelo(ruta_guardado=str(model_output_path))
        print(f"¡Entrenamiento registrado con éxito en MLflow! Run ID: {run.info.run_id}")