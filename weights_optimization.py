
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
import json

"""
This script receives data relative to Client-to-country distribution, AS information and pmatrixerable
paths information 

apply a minmax optimiztion problem and ouputs an allocation of scores for each
location which should satify the constraints of our problem (see README.md, procedure B)

"""

parser = argparse.ArgumentParser(description="")

parser.add_argument("--tor_users_to_location", help="path to the pickle file containing the distribution of Tor users per country")
parser.add_argument("--pickle", action="store_true", default=False)
parser.add_argument("--json", action="store_true", default=False)
parser.add_argument("--cust_locations", help="path to the file containing the distribution of IPs per customer AS")
parser.add_argument("--obj_function", type=int, help="Choice of objective function")
parser.add_argument("--cluster_file", type=str, required=True, help="Pickle file of clustered guards")
parser.add_argument("--pmatrix", type=str, help="Penalty matrix")
parser.add_argument("--reduced_as_to", type=int, help="for test purpose, gives the number of ASes to keep")
parser.add_argument("--reduced_guards_to", type=int, help="for test purpose, gives the number of guards to keep")
parser.add_argument("--load_problem", help="filepth with problem to solve if already computed")
parser.add_argument("--out_dir", help="out dir to save the .lp file")
## Note: simple at first. Later we may try to solve the problem for following network states and initialize variables of the n+1 state with
## the solution computed for state n
parser.add_argument("--network_state", help="filepath to the network state containing Tor network's data")


def load_and_compute_W_from_citymap(tor_users_to_location_file):
    with open(tor_users_to_location_file, 'r') as f:
        tor_users_to_location = json.load(f)
    #normalize
    W = {}
    tot = sum(tor_users_to_location.values())
    for location, value in tor_users_to_location.items():
        W[location] = value/tot
    return W

def load_and_compute_W(tor_users_to_location_file, cust_locations_file, reduced_as_to=None):
    with open(tor_users_to_location_file, 'rb') as f:
            tor_users_to_location = pickle.load(f)
    with open(cust_locations_file, 'rb') as f:
        cust_locations = pickle.load(f)
    print("Files have been loaded...")
    ## Now, compute the distributions of Tor users per ASes
    tor_users_per_as = {}
    for countrycode in tor_users_to_location:
        # get the specific list of all Ases for this country
        #Improve it to get time too0000
        #r = requests.get(GETAS_URL+countrycode+"&lod=1")
        #if r.status_code == 200:
            #allAses = r.json()
            #for loc in allAses['data']['countries'][0]['routed']:
        for loc in cust_locations:
                #if loc in cust_locations:
            tor_users_per_as[loc] = tor_users_to_location[countrycode]*cust_locations[loc].num_ipv4_addresses
                #else:
                    #print("AS{} from country {} not in our dataset".format(loc, countrycode))

        #else:
            #raise ValueError("Something wrong happened: {}", r.json())
    W = {}
    if reduced_as_to:
        allAses = list(tor_users_per_as.keys())
        to_delete = len(allAses) - reduced_as_to
        print("deleting {} elemets".format(to_delete))
        for loc in allAses:
            del tor_users_per_as[loc]
            to_delete -= 1
            if to_delete == 0:
                break
    # Normalize
    tot = sum(tor_users_per_as.values())
    for loc, value in tor_users_per_as.items():
        W[loc] = value/tot
    return W

def build_fake_pmatrix_profile(guards, W):
    """
        build a bivariate dicrete distribution for AS n and Guard i
        with a pmatrixerability score

        This function builds a fake one for test purpose 
    """
    pmatrix = {}
    tot = 0
    for loc in W:
        if loc not in pmatrix:
            pmatrix[loc] = {}
        for guard in guards:
            #pmatrix[guard][loc] = random.randint(0,1000)
            pmatrix[loc][guards] = 1000 #uniform pmatrixerability
        tot += sum(pmatrix[loc].values())
    for guard in guards:
        for loc in W:
            pmatrix[loc][guard] = pmatrix[guard][loc]/tot
    return pmatrix


def modelize_opt_problem(W, ns_file, obj_function, cluster_file, out_dir=None, pmatrix_file=None, reduced_as_to=None, reduced_guards_to=None):
    network_state = get_network_state(ns_file)
    
    with open(cluster_file, "rb") as f:
        clusters = pickle.load(f)
    # guardsfp = [relay for relay in network_state.cons_rel_stats if Flag.GUARD in network_state.cons_rel_stats[relay].flags and
            # not Flag.EXIT in network_state.cons_rel_stats[relay].flags]
    if reduced_guards_to:
        guardsfp = guardsfp[0:reduced_guards_to]
    

    R = {}
    #Compute total G bandwidth
    G = 0
    #max_cons_weight is going to be used as an upper bound of our
    #objective function
    max_cons_weight = 0
    for prefix in clusters:
        G += clusters[prefix].tot_consweight
        if clusters[prefix].tot_consweight > max_cons_weight:
            max_cons_weight = clusters[prefix].tot_consweight
    
    prefixes = list(clusters.keys())
    #pmatrix is a discrete bivariate distribution [guard][location] which
    #gives a high score if the path between location and guard is bad
    if not pmatrix_file:
        pmatrix = build_fake_pmatrix_profile(prefixes, W)
    else:
        with open(pmatrix_file, 'r') as f:
            pmatrix = json.load(f)
    
    #Normalize Wgg
    Wgg = network_state.cons_bw_weights['Wgg']/network_state.cons_bwweightscale
    
    #Modelize the problem
    location_aware = LpProblem("Location aware selection", LpMinimize)
    
    for loc in W:
        R[loc] = LpVariable.dicts(loc, prefixes, lowBound = 0,
                upBound=Wgg*G)
    
    location_aware = LpProblem("Optimal location-aware path selection", LpMinimize)
    # Write a minmax problem as a min of a upper bound
    #
    # The upBound value matches the worst-case scenario, where the matrix pmatrix has only
    # one non-negative real number on the guard with the highest consensus weight
    objective = LpVariable("L_upper_bound", lowBound = 0, upBound=max_cons_weight)
    # Compute L as affine expressions involving LpVariables
    print("Computing Affine Expressions for L, i.e., \sum W_iR_i")
    L = {}
    for prefix_guard in prefixes:
        L[prefix_guard] = LpAffineExpression([(R[loc][prefix_guard], W[loc]) for loc in W], name="L({})".format(prefix_guard))
    print("Done.")
    ##  min_R max_j ( [\sum_{j} L(i)*pmatrix(j)(i)  for i in all guards])
    if obj_function == 1:
        location_aware += objective, "Z" #set objective function
        #Trick to avoid complexity explosion of PuLP
        Intermediate = {}
        print("Computing Intermediate var")
        for prefix_guard in prefixes:
            Intermediate[prefix_guard] = LpVariable("Intermediate guard var {}".format(prefix_guard), lowBound = 0, upBound=max_cons_weight)
            location_aware += Intermediate[prefix_guard] == L[prefix_guard], "Intermediate on {}".format(prefix_guard)
        #print("Done.")
        print("Computing the objective Z with linked constraints")
        #min max L*pmatrix is equal to min Z with Z >= L[guard_i]*Vu
        for prefix_guard in prefixes:
            location_aware += objective >=\
                LpAffineExpression([(Intermediate[prefix_guard], pmatrix[loc][prefix_guard]) for loc in W], name="Intermediate  \sum L[{}]*pmatrix[loc][{}]".format(prefix_guard, prefix_guard)),\
                "Added constraint Z >= \sum L[{}]*pmatrix[loc][{}] forall loc".format(prefix_guard, prefix_guard)
                #lpSum([Intermediate[guard]*pmatrix[guard][loc] for loc in W])
            print("Added constraint Z >= \sum L[{}]*pmatrix[loc][{}] forall loc".format(prefix_guard, prefix_guard))
    ##   min_R max_j ([\sum_{i} W(j)*R(j,i)*pmatrix(j)(i)  for j in all locations])
    elif obj_function == 2:
        location_aware += objective, "Z" #set objective function
        for loc in W:
            location_aware += objective >= \
                LpAffineExpression([(R[loc][prefix_guard], W[loc]*pmatrix[loc][prefix_guard]) for prefix_guard in prefixes])
            print("Added constraint Z >= \sum_i W({})*R[{}][prefix]*pmatrix[prefix][{}] forall prefix".format(loc, loc, loc))
    ##   min_R (\sum_i \sum_j W(j)*R(i,j)*pmatrix(i,j))
    elif obj_function == 3:
        print("Computing the lpSum of LpAffineExpressions as an objective function... (this can take time)")
        location_aware += lpSum([LpAffineExpression([(R[loc][prefix_guard], W[loc]*pmatrix[loc][prefix_guard]) for asn in W]) for prefix_guard in prefixes]), "Z"
    ##   min max_j (\sum_i R(i,j)*pmatrix(i,j)) 
    elif obj_function == 4:
        location_aware += objective, "Z" #set objective function
        for loc in W:
            location_aware += objective >= \
                    LpAffineExpression([(R[loc][prefix_guard], pmatrix[loc][prefix_guard]) for prefix_guard in prefixes])
            print("Added constraint Z >= R[{}][prefix_guard]*pmatrix[prefix_guard][{}] forall prefixes".format(loc, loc))
    print("Done.")
    # Now set of constraints:
    # Location scores must distribute G*Wgg quantity
    print("Computing constraints \sum R_l(i) == G*Wgg")
    for loc in W:
        location_aware += lpSum([R[loc][prefix_guard] for prefix_guard in prefixes]) == G*Wgg, "\sum R(i) == G*Wgg for loc {}".format(loc)
    print("Done.")
    print("Computing constraints L(i) <= BW_i")
    for prefix_guard in prefixes:
        location_aware += L[prefix_guard] <= clusters[prefix_guard].tot_consweight, "L(i) <= BW_i for prefix_guard {}".format(prefix_guard)

    ## Missing constraint for theta-GP-Secure TODO
        
    #Temporally wating for theta-GP-secure stuffs
    #No relay gets more than 2 times its original selection probability
    print("No relay gets more than 2 times its original selection probability:")
    for loc in W:
        for prefix_guard in prefixes:
            location_aware += R[loc][prefix_guard] <= 2*clusters[prefix_guard].tot_consweight

    print("Done. Writting out pickle file")

    # Write problem out:
    if out_dir:
        if reduced_as_to or reduced_guards_to:
            outpath = os.path.join(out_dir, "location_aware_with_obj_{}_reducedas_{}_reducedguard_{}".format(obj_function, reduced_as_to,reduced_guards_to))
        else:
            outpath = os.path.join(out_dir, "location_aware_with_obj_{}".format(obj_function))
    else:
        outpath = "location_aware.pickle"
    with open(outpath+".pickle", "wb") as f:
        pickle.dump(location_aware, f, pickle.HIGHEST_PROTOCOL)

    print("Done. Writting out .lp file")
    location_aware.writeLP(outpath+".lp")
    print("Done. Writtin out .mps file")
    location_aware.writeMPS(outpath+".mps")
    #location_aware.solve()

    
if __name__ == "__main__":
    args = parser.parse_args()
    ## Compute appropritate values and modelize the optimization problem
    if args.tor_users_to_location:
        if args.pickle and args.cust_locations:
            W = load_and_compute_W(args.tor_users_to_location, args.cust_locations, args.reduced_as_to)
        elif args.json:
            W = load_and_compute_W_from_citymap(args.tor_users_to_location)
        modelize_opt_problem(W, args.network_state, args.obj_function, args.cluster_file, args.out_dir, args.pmatrix, args.reduced_as_to, args.reduced_guards_to)
    ## Load the problem and solve() it
    elif args.load_problem:
        with open(args.load_problem, "rb") as f:
            location_aware = pickle.load(f)
            location_aware.solve(pulp.PULP_CBC_CMD(msg=1, threads=4))
            pdb.set_trace()
            for v in location_aware.variables():
                print(v.name, "=", v.varValue)
