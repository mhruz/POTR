import yaml
import sys
import os

if __name__ == "__main__":

    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = "python main.py "

    for key, value in config.items():
        command += "--{} {} ".format(key, value)

    print(command)

    os.system(command)
