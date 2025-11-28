import json
from src.data.database import DatabaseManager, Experiment
from datetime import datetime

class ExperimentLogger:
    def __init__(self):
        self.db_manager = DatabaseManager()

    def log_experiment(self, model_type: str, features: list, hyperparams: dict, metrics: dict, notes: str = ""):
        """
        Logs a training experiment to the database.
        """
        session = self.db_manager.get_session()
        try:
            experiment = Experiment(
                timestamp=datetime.now(),
                model_type=model_type,
                features_list=json.dumps(features),
                hyperparameters=json.dumps(hyperparams),
                accuracy=metrics.get('accuracy'),
                log_loss=metrics.get('log_loss'),
                latency_avg_ms=metrics.get('latency_ms'),
                notes=notes
            )
            session.add(experiment)
            session.commit()
            print(f"Logged experiment: {model_type} (Acc: {metrics.get('accuracy'):.4f})")
        except Exception as e:
            session.rollback()
            print(f"Failed to log experiment: {e}")
        finally:
            session.close()
