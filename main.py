from helper.loader import SUKPLoader
from helper.set_handler import SetUnionHandler
from solver.agent import DQNAgent
import os
def main():
    yaml_path = "helper/config.yaml"
    loader = SUKPLoader(yaml_path)
    data = loader.get_data()
    param = loader.get_param()
    suk = SetUnionHandler(data, param)
    agent = DQNAgent(suk)


if __name__ == "__main__":
    main()