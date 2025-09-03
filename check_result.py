import numpy as np
from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
import os

def main():
    yaml_path = "helper/config.yaml"
    loader = SUKPLoader(yaml_path)
    data = loader.get_data()
    param = loader.get_param()
    suk = SetUnionHandler(data, param)
    result_dir = 'result'
    file_name = loader.get_filename()
    npy_path = os.path.join(result_dir, f'{file_name}.npy')
    
    if not os.path.exists(npy_path):
        print(f"File {npy_path} does not exist.")
        return
    
    best_sol = np.load(npy_path)
    
    if len(best_sol) != data['m']:
        print("Loaded best_sol has incorrect length.")
        return
    
    
    # Add items based on best_sol
    for i in range(len(best_sol)):
        if best_sol[i] > 0.5:  # Since it's floats 0.0 or 1.0
            suk.add_item(i)
    
    result_str = ' '.join(['1' if x > 0.5 else '0' for x in best_sol])
    print(f"Result: {result_str}")
    print(f"Total weight: {suk.get_weight()}, capacity: {loader.capacity}")
    
    # Get and print total profit
    total_profit = suk.get_profit()
    print(f"Total profit: {total_profit}")

if __name__ == "__main__":
    main()