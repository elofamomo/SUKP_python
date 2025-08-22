from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
import os
def main():
    path = "SUKP_instances_60/Instances of Set II/sukp_600_585_0.10_0.75.txt"
    loader = SUKPLoader(path)
    data = loader.get_data()
    element_weights = data["element_weights"]
    sum = 0
    for i in data["item_subsets"][2]:
        sum += element_weights[i]
    print(sum) 
    print(len(data["item_subsets"][2]))
    suk = SetUnionHandler(data)
    suk.add_item(2)
    print(suk.get_value())
    print(suk.get_weight())
    print(suk.get_current_union())


if __name__ == "__main__":
    main()