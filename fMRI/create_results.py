import numpy as np
import pandas as pd
import sys

if len(sys.argv) != 3:
    print("Usage: python script.py <class1> <class2>")
    sys.exit(1)

class1 = sys.argv[1]
class2 = sys.argv[2]


algos = ["lasso" , 'enet', 'omp', 'iht', 'ksn', 'irksn', 'ircr', 'irosr', 'srdi']

comp_alg = np.load(f'encludl_beta_{class1}_{class2}.npy')

distances = {'Algorithm': [], 'Distance': []}

for algo in algos:
    curr_alg = np.load(f"{algo}_beta_{class1}_{class2}.npy")
    distance = np.linalg.norm(curr_alg - comp_alg)
    distances['Algorithm'].append(algo)
    distances['Distance'].append(distance)

df = pd.DataFrame(distances)
df.to_csv(f'./distances_{class1}_{class2}.csv', index=False)
