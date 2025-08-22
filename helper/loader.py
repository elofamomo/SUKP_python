import numpy as np
class SUKPLoader:
    def __init__(self, file_path):
        """
        Loads SUKP instance data from a file.
        
        :param file_path: Path to the instance file.
        """
        self.m = None
        self.n = None
        self.capacity = None
        self.item_profits = None
        self.element_weights = None
        self.item_subsets = None  # List of lists: for each item, list of element indices it covers

        self._load(file_path)
    
    def _load(self, file_path):
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