
import sys, os
import argparse
from slim_ases import *
from pulp import *
import numpy as np
import pickle
from tor_users_per_country import get_network_state
from process_ases import GETAS_URL
import requests

"""
This script receives data relative to Client-to-country distribution, AS information and vulnerable
paths information 

apply a minmax optimiztion problem and ouputs an allocation of scores for each
location which should satify the constraints of our problem (see README.md, procedure B)

"""

parser = argparse.ArgumentParser(description="")

parser.add_argument("--tor_users_to_country", help="path to the pickle file containing the distribution of Tor users per country")
parser.add_argument("--cust_ases", help="path to the pickle file containing the distribution of IPs per customer AS")
parser.add_argument("--load_problem", help="filepth with problem to solve if already computed")
## Note: simple at first. Later we may try to solve the problem for following network states and initialize variables of the n+1 state with
## the solution computed for state n
parser.add_argument("--network_state", help="filepath to the network state containing Tor network's data")


def load_and_compute_W(tor_users_to_country_file, curt_ases_file):
    with open(tor_users_to_country_file, 'rb'):
        tor_users_to_country = pickle.load(tor_users_to_country_file)
    with open(cust_ases_file, 'rb'):
        cust_ases = pickle.load(cust_ases_file)
    print("Files have been loaded...")
    ## Now, compute the distributions of Tor users per ASes
    tor_users_per_as = {}
    for countrycode in tor_users_to_country:
        # get the specific list of all Ases for this country
        #Improve it to get time too0000
        r = requests.get(GETAS_URL+countrycode+"&lod=1")
        if r.status_code == 200:
            allAses = r.json()
            for asn in allAses['data'][0]['routed']:
                if asn in cust_ases:
                    tor_users_per_as[asn] = tor_users_to_country[countrycode]*cust_ases[asn]
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

def build_fake_vuln_profile(network_state)
    pass

def modelize_opt_problem(W, ns_file):
    network_state = get_network_state(ns_file)
    guardsfp = [relay in network_state.cons_rel_stats if Flag.GUARD in network_state.cons_rel_stats[relay].flags and
            not Flag.EXIT in network_state.cons_rel_stats[relay].flags]
    R = {}
    #Compute total G bandwidth
    G = 0
    for guard in guardsfp:
        G += network_state.cons_rel_stats[guard].consweight
    #Normalize Wgg
    Wgg = network_state.cons_bw_weights['Wgg']/network_state.cons_bwweightscale
    
    #Modelize the problem
    location_aware = LpProblem("Location aware selection", LpMinimize)
    
    #Vuln is a discrete bivariate distribution [guard][client_asn] which
    #gives a high score if the path between client_asn and guard is bad
    
    Vuln = build_fake_vuln_profile(network_state)

    for asn in W:
        R[asn] = LpVariable.dicts(asn, guardsfp, lowBound = 0,
                upBound=(network_state.cons_bw_weights['Wgg']/network_state.cons_bwweightscale)*G)
    
    location_aware = LpProblem("Optimal location-aware path selection", LpMinimize)
    # Write a minmax problem as a min of a upper bound
    #TODO careful, the upBound depends on the scale of Vuln
    objective = LpVariable("L_upper_bound", lowBound = 0, upBound=Wgg*G)
    # Compute L as affine expressions involving LpVariables
    L = {}
    for guard in guardsfp:
        L[guard] = LpAffineExpression([(R[asn][guard], W[asn]) for asn in W])
    
    #min max L*Vuln is equal to min Z with Z >= L[guard_i]*Vu
    location_aware += objective #set objective function
    for guard in guardsfp:
        location_aware += objective >=
            LpAffineExpression([(L[guard], Vuln[guard][asn]) for asn in W])
    # Now set of constraints:
    # Location scores must distribute G*Wgg quantity
    for asn in W:
        location_aware += lpSum([R[asn][guard] for guard in guardsfp]) == G*Wgg
    for guard in guardsfp:
        location_aware += L[guard] <= network_state.cons_rel_stats[guard].consweight

    ## Missing constraint for theta-GP-Secure TODO

    # Write problem out:
    location_aware.writeLP("location_aware.lp")




    
if __name__ == "__name__":
    args = parser.parse_args()
    ## Compute appropritate values and modelize the optimization problem
    if args.tor_users_to_country and args.cust_ases:
        W, network_state = load_and_compute_W(args.tor_users_to_country, args.cust_ases)
        
    ## Load the problem and solve() it
    elif args.load_problem:
        pass
