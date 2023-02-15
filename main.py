import sys
sys.path.append("..")

from A2_ComputacionEvolutiva import *
from ParamScheduler import *
from actividad2Funcs import *

import argparse

def run_algorithm(alg_name):
    params = {
        # Population-based
        "popSize": 100,

        # Genetic algorithm
        "pmut": 0.2,
        "pcross":0.9,

        # Evolution strategy
        "offspringSize":500,
        "sigma_type":"nstepsize",
        #"tau":1/np.sqrt(10),

        # General
        "stop_cond": "ngen",
        "time_limit": 20.0,
        "Ngen": 300,
        "Neval": 1e5,
        "fit_target": 1000,

        "verbose": True,
        "v_timer": 0.5,

        # Metrics
        "success":15
    }

    operators = [
        OperatorReal("Multipoint"),
        #OperatorReal("DE/best/1", {"F":0.7, "Cr":0.8}),
        OperatorReal("Gauss", {"F":0.001}),
        OperatorReal("Cauchy", {"F":0.005}),
    ]

    objfunc = SumPowell(10)

    mutation_op = OperatorReal("Gauss", {"F": 0.001})
    # mutation_op = OperatorReal("Gauss", ParamScheduler("Lineal", {"F":[0.1, 0.001]}))
    cross_op = OperatorReal("Multipoint")
    # parent_select_op = ParentSelection("Tournament", {"amount": 3, "p":0.1})
    parent_select_op = ParentSelection("Tournament", ParamScheduler("Lineal", {"amount": [2, 7], "p":0.1}))
    replace_op = SurvivorSelection("(m+n)")

    if alg_name == "ES":
        alg = ES(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
    elif alg_name == "DE":
        alg = DE(objfunc, OperatorReal("DE/current-to-best/1", {"F":0.8, "Cr":0.9}), SurvivorSelection("One-to-one"), params)
    else:
        print(f"Error: Algorithm \"{alg_name}\" doesn't exist.")
        exit()
        

    ind, fit = alg.optimize()
    print('Best individual: ',ind)
    list_history = np.array(alg.history)
    success_idx = np.where(list_history < params["success"])
    print('Aprox. PEX: ', success_idx[0][0] * alg.time_spent/len(list_history))
    alg.display_report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest='alg', help='Specify an algorithm')
    args = parser.parse_args()

    algorithm_name = "ES"
    if args.alg:
        algorithm_name = args.alg
   
    run_algorithm(alg_name = algorithm_name)

if __name__ == "__main__":
    main()