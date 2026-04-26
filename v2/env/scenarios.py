import random
from typing import Dict, Tuple

class ScenarioGenerator:
    def __init__(self, config_path="v2/config/env_config.yaml"):
        # For simplicity in this script without yaml dependency, we hardcode the topologies
        self.services = ["api_gateway", "auth_service", "user_db", "frontend_service", "product_db", "cache_service"]
        
        self.scenarios = [
            {"root_cause": "db_connection_leak", "target": "user_db"},
            {"root_cause": "memory_leak", "target": "auth_service"},
            {"root_cause": "cascading_failure", "target": "cache_service"},
            {"root_cause": "cpu_saturation", "target": "api_gateway"}
        ]

    def generate(self, difficulty="medium") -> Tuple[str, str, Dict, Dict]:
        scenario = random.choice(self.scenarios)
        rc = scenario["root_cause"]
        target = scenario["target"]
        
        logs = {}
        metrics = {}
        
        # Populate realistic baseline metrics for all services
        for svc in self.services:
            metrics[svc] = {"cpu": random.uniform(0.1, 0.4), "latency": random.uniform(0.01, 0.05), "error_rate": 0.0}
            logs[svc] = "INFO: Service running normally."

        # Apply failure effects
        if rc == "db_connection_leak":
            metrics[target]["latency"] = 1.5
            metrics[target]["error_rate"] = 0.8
            logs[target] = "ERROR: connection pool exhausted. Waiting for connection..."
            if difficulty in ["medium", "hard"]:
                # Cascading to auth
                metrics["auth_service"]["latency"] = 1.2
                logs["auth_service"] = "WARN: Timeout connecting to user_db"
                
        elif rc == "memory_leak":
            metrics[target]["cpu"] = 0.95
            logs[target] = "FATAL: OutOfMemoryError in auth_service"
            
        elif rc == "cascading_failure":
            metrics[target]["error_rate"] = 1.0
            logs[target] = "ERROR: Cache eviction failed"
            metrics["frontend_service"]["latency"] = 2.0
            logs["frontend_service"] = "ERROR: Cache miss spike, database overwhelmed"
            metrics["product_db"]["cpu"] = 0.99
            
        elif rc == "cpu_saturation":
            metrics[target]["cpu"] = 0.99
            metrics[target]["latency"] = 0.8
            logs[target] = "WARN: Thread starvation detected"

        return rc, target, logs, metrics
