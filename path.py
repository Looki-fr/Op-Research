import os

current_path = os.path.dirname(__file__)
data_path=os.path.join(current_path, 'data')

get_path_file = lambda filename: os.path.join(data_path, filename)