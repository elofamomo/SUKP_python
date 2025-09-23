import numpy as np
import yaml
from pathlib import Path
import os
class SUKPLoader:
    def __init__(self, yaml_path):
        """
        Loads SUKP instance data from a file.
        
        :param file_path: Path to the instance file.
        """
        self.yaml_path = yaml_path
        self.m = None
        self.n = None
        self.capacity = None
        self.item_profits = None
        self.element_weights = None
        self.file_name = None
        self.item_subsets = None  # List of lists: for each item, list of element indices it covers
        file_path = self._load_sukp_config(self.yaml_path)
        self._load(file_path)
    
    def _load(self, file_path):
        if not os.path.exists(file_path):
            raise ValueError(f"File path {file_path} is not ready")
        with open(file_path, 'r') as f:
            f.readline().strip()
            f.readline().strip()
            line = f.readline().strip()
            parts = line.split()
            self.m = int(parts[0].split('=')[1])
            self.n = int(parts[1].split('=')[1])

            self.capacity = int(parts[3].split('=')[1])

            f.readline().strip()
            profit_header = f.readline().strip()
            if not profit_header.startswith("The profit of"):
                raise ValueError("Invalid profit header")
            
            # Read profits
            profits = []
            while len(profits) < self.m:
                line = f.readline().strip()
                if line:
                    profits.extend([int(x) for x in line.split()])
            if len(profits) != self.m:
                raise ValueError(f"Expected {self.m} profits, got {len(profits)}")
            self.item_profits = np.array(profits, dtype=int)  # Use float for generality
            
            # Read weights
            f.readline().strip()
            weight_header = f.readline().strip()
            if not weight_header.startswith("The weight of"):
                raise ValueError("Invalid weight header")
            
            # Read weights
            weights = []
            while len(weights) < self.n:
                line = f.readline().strip()
                if line:
                    weights.extend([int(x) for x in line.split()])
            if len(weights) != self.n:
                raise ValueError(f"Expected {self.n} weights, got {len(weights)}")
            self.element_weights = np.array(weights, dtype=float)
            
            line = f.readline().strip()
            while not line:
                line = f.readline().strip()
            
            # Read relation matrix header: "Relation matrix"
            if line != "Relation matrix":
                raise ValueError("Invalid relation matrix header")
            
            # Read the matrix data (all remaining, split into tokens)
            matrix_data = []
            for line in f:
                matrix_data.extend([int(x) for x in line.strip().split() if x])
            
            expected_size = self.m * self.n
            if len(matrix_data) != expected_size:
                raise ValueError(f"Expected {expected_size} matrix entries, got {len(matrix_data)}")
            
            # Reshape into m x n array
            relation_matrix = np.array(matrix_data).reshape(self.m, self.n)
            
            # Convert to item_subsets: list of element indices per item
            self.item_subsets = []
            for row in relation_matrix:
                subset = np.where(row == 1)[0].tolist()
                self.item_subsets.append(subset)
    
    def get_data(self):
        """
        Returns the loaded data as a dictionary for easy access.
        """
        return {
            'm': self.m,
            'n': self.n,
            'capacity': self.capacity,
            'item_profits': self.item_profits,
            'element_weights': self.element_weights,
            'item_subsets': self.item_subsets
        }
    
    def _load_sukp_config(self, yaml_path='config.yaml'):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        data_config = config.get('sukp_data', {})
        n = data_config.get('n')
        m = data_config.get('m')
        alpha = data_config.get('alpha')
        beta = data_config.get('beta')
        instance = data_config.get('instance')
        missing_params = []
        if n is None:
            missing_params.append('n')
        if m is None:
            missing_params.append('m')
        if alpha is None:
            missing_params.append('alpha')
        if beta is None:
            missing_params.append('beta')
        if instance is None:
            missing_params.append('instance')
        if missing_params:
            raise ValueError(f"Missing required parameters in YAML: {', '.join(missing_params)}")
        if instance not in (1,2):
            raise ValueError("Instance must be 1 or 2")
        base_path = "SUKP_instances_60"
        if instance == 1:
            instance_path = "Instances of Set I"
        elif instance == 2:
            instance_path = "Instances of Set II"
        else:
            raise ValueError("Instance value must be 1 or 2")
        
        alpha_str = f"{alpha:.2f}"
        beta_str = f"{beta:.2f}"
        file_name = f"sukp_{m}_{n}_{alpha_str}_{beta_str}"
        file_name_txt = f"sukp_{m}_{n}_{alpha_str}_{beta_str}.txt"
        file_path = f"{base_path}/{instance_path}/{file_name_txt}"
        self.file_name = file_name
        print(f"Working on {file_name}")
        return file_path
    
    def get_param(self):
        with open(self.yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        param = config.get('agent_param', {})
        return param
    
    def load_general_config(self):
        with open(self.yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        load_checkpoint = config.get('load_checkpoint', bool)
        save_checkpoint = config.get('save_checkpoint', bool)
        interval_update = config.get('update_interval', int)
        episode = config.get('episode', int)
        batch_size = config.get('batch_size', int)
        return {
            'load_checkpoint': load_checkpoint,
            'save_checkpoint': save_checkpoint,
            'episodes': episode,
            'batch_size': batch_size,
            'update_interval': interval_update
        }
    def get_filename(self):
        return self.file_name