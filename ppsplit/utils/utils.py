
import os

def create_dir(dir_route):
    if not os.path.exists(dir_route):
        os.makedirs(dir_route)
    return