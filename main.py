from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
import os
def main():
    yaml_path = "helper/config.yaml"
    loader = SUKPLoader(yaml_path)
    data = loader.get_data()
    suk = SetUnionHandler(data)
    


if __name__ == "__main__":
    main()