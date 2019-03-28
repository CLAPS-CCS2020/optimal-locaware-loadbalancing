
import sys, os
import argparse
from slim_ases import *
from pulp import *
import numpy as np
import pickle
from tor_users_per_country import get_network_state
from process_ases import GETAS_URL
import requests
import random
import pdb
from stem import Flag

"""
This script receives data relative to Client-to-country distribution, AS information and vulnerable
paths information 

apply a minmax optimiztion problem and ouputs an allocation of scores for each
location which should satify the constraints of our problem (see README.md, procedure B)

"""

parser = argparse.ArgumentParser(description="")

parser.add_argument("--tor_users_to_country", help="path to the pickle file containing the distribution of Tor users per country")
parser.add_argument("--cust_ases", help="path to the pickle file containing the distribution of IPs per customer AS")
parser.add_argument("--obj_function", type=int, help="Choice of objective function")
parser.add_argument("--load_problem", help="filepth with problem to solve if already computed")
parser.add_argument("--out_dir", help="out dir to save the .lp file")
## Note: simple at first. Later we may try to solve the problem for following network states and initialize variables of the n+1 state with
## the solution computed for state n
parser.add_argument("--network_state", help="filepath to the network state containing Tor network's data")


def load_and_compute_W(tor_users_to_country_file, cust_ases_file):
    with open(tor_users_to_country_file, 'rb') as f:
        tor_users_to_country = pickle.load(f)
    with open(cust_ases_file, 'rb') as f:
        cust_ases = pickle.load(f)
    print("Files have been loaded...")
    ## Now, compute the distributions of Tor users per ASes
    tor_users_per_as = {}
    for countrycode in tor_users_to_country:
        # get the specific list of all Ases for this country
        #Improve it to get time too0000
        r = requests.get(GETAS_URL+countrycode+"&lod=1")
        if r.status_code == 200:
            allAses = r.json()
            for asn in allAses['data']['countries'][0]['routed']:
                if asn in cust_ases:
                    tor_users_per_as[asn] = tor_users_to_country[countrycode]*cust_ases[asn].num_ipv4_addresses
                else:
                    print("AS{} from country {} not in our dataset".format(asn, countrycode))

        else:
            raise ValueError("Something wrong happened: {}", r.json())
    # Normalize
    tot = sum(tor_users_per_as.values())
    W = {}
    for asn, value in tor_users_per_as.items():
        W[asn] = value/tot
    return W

def build_fake_vuln_profile(guards, W):
    """
        build a bivariate dicrete distribution for AS n and Guard i
        with a vulnerability score

        This function builds a fake one for test purpose 
    """
    Vuln = {}
    tot = 0
    for guard in guards:
        if guard not in Vuln:
            Vuln[guard] = {}
        for asn in W:
            Vuln[guard][asn] = random.randint(0,1000)
        tot += sum(Vuln[guard].values())
    for guard in guards:
        for asn in W:
            Vuln[guard][asn] = Vuln[guard][asn]/tot
    return Vuln


def modelize_opt_problem(W, ns_file, obj_function, out_dir=None):
    network_state = get_network_state(ns_file)
    guardsfp = [relay for relay in network_state.cons_rel_stats if Flag.GUARD in network_state.cons_rel_stats[relay].flags and
            not Flag.EXIT in network_state.cons_rel_stats[relay].flags]
    R = {}
    #Compute total G bandwidth
    G = 0
    #max_cons_weight is going to be used as an upper bound of our
    #objective function
    max_cons_weight = 0
    for guard in guardsfp:
        G += network_state.cons_rel_stats[guard].consweight
        if network_state.cons_rel_stats[guard].consweight > max_cons_weight:
            max_cons_weight = network_state.cons_rel_stats[guard].consweight

    #Normalize Wgg
    Wgg = network_state.cons_bw_weights['Wgg']/network_state.cons_bwweightscale
    
    #Modelize the problem
    location_aware = LpProblem("Location aware selection", LpMinimize)
    
    #Vuln is a discrete bivariate distribution [guard][client_asn] which
    #gives a high score if the path between client_asn and guard is bad
    
    Vuln = build_fake_vuln_profile(guardsfp, W)

    for asn in W:
        R[asn] = LpVariable.dicts(asn, guardsfp, lowBound = 0,
                upBound=Wgg*G)
    
    location_aware = LpProblem("Optimal location-aware path selection", LpMinimize)
    # Write a minmax problem as a min of a upper bound
    #
    # The upBound value matches the worst-case scenario, where the matrix Vuln has only
    # one non-negative real number on the guard with the highest consensus weight
    objective = LpVariable("L_upper_bound", lowBound = 0, upBound=max_cons_weight)
    # Compute L as affine expressions involving LpVariables
    print("Computng Affine Expressions for L, i.e., \sum W_iR_i")
    L = {}
    for guard in guardsfp:
        L[guard] = LpAffineExpression([(R[asn][guard], W[asn]) for asn in W])
    print("Done.")
    ##  min_R max_j ( [\sum_{j} L(i)*Vuln(i)(j)  for i in all guards])
    if obj_function == 1:
        location_aware += objective, "Z" #set objective function
        #Trick to avoid complexity explosion of PuLP
        Intermediate = {}
        print("Computing Intermediate var")
        for guard in guardsfp:
            Intermediate[guard] = LpVariable("Intermediate guard var {}".format(guard), lowBound = 0, upBound=max_cons_weight)
            location_aware += Intermediate[guard] == L[guard], "Intermediate on {}".format(guard)
        #print("Done.")
        print("Computing the objective Z with linked constraints")
        #min max L*Vuln is equal to min Z with Z >= L[guard_i]*Vu
        for guard in guardsfp:
            location_aware += objective >=\
                LpAffineExpression([(Intermediate[guard], Vuln[guard][asn]) for asn in W], name="Intermediate  \sum L[{}]*vuln[{}][asn]".format(guard, guard),\
                "Added constraint Z >= \sum L[{}]*vuln[{}][asn] forall asn".format(guard, guard)
                #lpSum([Intermediate[guard]*Vuln[guard][asn] for asn in W])
            print("Added constraint Z >= \sum L[{}]*vuln[{}][asn] forall asn".format(guard, guard))
    ##   min_R max_j ([\sum_{i} W(j)*R(i,j)*Vuln(i)(j)  for j in all locations])
    elif obj_function == 2:
        location_aware += objective, "Z" #set objective function
        for asn in W:
            location_aware += objective >= \
                LpAffineExpression([(R[asn][guard], W[asn]*Vuln[guard][asn]) for guard in guardsfp])
            print("Added constraint Z >= \sum_i W({})*R[{}][{}]*Vuln[{}][{}]".format(asn, asn, guard, guard, asn))
    ##   min_R (\sum_i \sum_j W(j)*R(i,j)*Vuln(i,j))
    elif obj_function == 3:
        print("Computing the lpSum of LpAffineExpressions as an objective function... (this can take time)")
        location_aware += lpSum([LpAffineExpression([(W[asn]*Vuln[guard][asn], R[asn][guard]) for asn in W]) for guard in guardsfp]), "Z"
    ##   min max_j (\sum_i R(i,j)*Vuln(i,j)) 
    elif obj_function == 4:
        location_aware += objective, "Z" #set objective function
        for asn in W:
            location_aware += objective >= \
                    LpAffineExpression([(R[asn][guard], Vuln[guard][asn]) for guard in guardsfp])
            print("Added constraint Z >= R[{}][{}]*Vuln[{}][{}]".format(asn, guard, guard, asn))
    print("Done.")
    # Now set of constraints:
    # Location scores must distribute G*Wgg quantity
    print("Computing constraints \sum R_l(i) == G*Wgg")
    for asn in W:
        location_aware += lpSum([R[asn][guard] for guard in guardsfp]) == G*Wgg, "\sum R(i) == G*Wgg for asn {}".format(asn)
    print("Done.")
    print("Computing constraints L(i) <= BW_i")
    for guard in guardsfp:
        location_aware += L[guard] <= network_state.cons_rel_stats[guard].consweight, "L(i) <= BW_i for guard {}".format(guard)
    print("Done. Writting out .lp file")

    ## Missing constraint for theta-GP-Secure TODO

    # Write problem out:
    if out_dir:
        outpath = os.path.join(out_dir, "location_aware_with_obj_{}.lp".format(obj_function))
    else:
        outpath = "location_aware.lp"
    #location_aware.writeLP(outpath)
    location_aware.solve()

    
if __name__ == "__main__":
    args = parser.parse_args()
    ## Compute appropritate values and modelize the optimization problem
    if args.tor_users_to_country and args.cust_ases:
        W = load_and_compute_W(args.tor_users_to_country, args.cust_ases)
        modelize_opt_problem(W, args.network_state, args.obj_function, args.out_dir)
    ## Load the problem and solve() it
    elif args.load_problem:
        pass
