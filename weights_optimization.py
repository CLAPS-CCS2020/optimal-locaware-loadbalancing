import sys, os
import argparse
from slim_ases import *
from pulp import *
import numpy as np
import pickle
from util import get_network_state
from process_ases import GETAS_URL
import requests
import random
import pdb
from stem import Flag
import json
import math
from subprocess import Popen, PIPE

"""
This script receives data relative to Client-to-country distribution, AS information and penalty
paths information 

apply a minmax optimiztion problem and ouputs an allocation of scores for each
location which should satify the constraints of our problem (see README.md, procedure B)

"""

parser = argparse.ArgumentParser(description="")

parser.add_argument("--tor_users_to_location", help="path to the pickle file containing the distribution of Tor users per country")
parser.add_argument("--pickle", action="store_true", default=False)
parser.add_argument("--json", action="store_true", default=False)
parser.add_argument("--disable_SWgg", action="store_true", default=False)
parser.add_argument("--in_shadow", action="store_true", default=False)
parser.add_argument("--cust_locations", help="path to the file containing the distribution of IPs per customer AS")
parser.add_argument("--obj_function", type=int, help="Choice of objective function")
parser.add_argument("--cluster_file", type=str, help="Pickle file of clustered guards")
parser.add_argument("--client_clust_representative", type=str, help="Ryan's clusterization file for ASes in one AS representative")
parser.add_argument("--pmatrix", type=str, help="Penalty matrix")
parser.add_argument("--penalty_vanilla", type=str, help="Vanilla penalty vector for each location")
parser.add_argument("--reduced_as_to", type=int, help="for test purpose, gives the number of ASes to keep")
parser.add_argument("--reduced_guards_to", type=int, help="for test purpose, gives the number of guards to keep")
parser.add_argument("--load_problem", help="filepth with problem to solve if already computed")
parser.add_argument("--out_dir", help="out dir to save the .lp file")
parser.add_argument("--binary_search_theta", action='store_true', default=False)
parser.add_argument("--theta", type=float, help="set theta value for gpa", default=2.0)
## Note: simple at first. Later we may try to solve the problem for following network states and initialize variables of the n+1 state with
## the solution computed for state n
parser.add_argument("--network_state", help="filepath to the network state containing Tor network's data")

ELASTICITY = 0.001

class Relay():
    """
    from shadow-plugin-tor/tools/generate.py (duplicates to avoid dependencies)
    """
    def __init__(self, ip, bw, isExit=False, isGuard=False):
        self.ip = ip
        self.bwconsensus = int(bw) # in KiB, from consensus
        self.isExit = isExit
        self.isGuard = isGuard
        self.code = None
        self.name = None

        self.bwrate = 0 # in bytes
        self.bwburst = 0 # in bytes
        self.bwtstamp = 0

        self.maxobserved = 0 # in bytes
        self.maxread = 0 # in bytes
        self.maxwrite = 0 # in bytes

        self.upload = 0 # in KiB
        self.download = 0 # in KiB

        self.rates = [] # list of bytes/s histories


def load_and_compute_W_from_shadowcityinfo(tor_users_to_location, cityinfo):
    pass


def load_and_compute_W_from_clusterinfo(asn_to_users_file, clusterinfo):
    """
        Compute densities of users per cluster and returns
        both the density and the list of location per cluster
        representative
    """
    with open(asn_to_users_file) as f:
        asn_to_users = json.load(f)
    W = {}
    tot = sum(asn_to_users.values())
    repre = {}
    with open(clusterinfo) as f:
        for line in f:
            tab = line.split('\t')
            W[tab[0]] = 0
            repre[tab[0]] = tab[1][:-1].split(',')
            for asn in tab[1].split(',')[:-1]:
                W[tab[0]] += asn_to_users[asn]
            W[tab[0]] /= tot
    #Are we using the right files? =)
    nbr_repre =  sum([len(repre[x]) for x in repre if isinstance(repre[x], list)])
    assert nbr_repre == len(asn_to_users),\
    "looks like we don't have the same number of ASes within cluster representatives and the asn_to_json info: {} vs {}"\
    .format(nbr_repre, len(asn_to_users))
    return W, repre


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

def produce_clustered_pmatrix(pmatrix, repre, asn_to_users, gclusters):
    """
    Computer a pmatrix cluster -> guard -> pvalue as the weighted sum
    of penalties given all ases of a given cluster, according to their number
    of users
    """
    pmatrix_clustered = {}
    for representative, ases in repre.items():
        pmatrix_clustered[representative] = {}
        for gclusterid in gclusters:
            tot = 0
            tot_users = 0
            for asn in ases:
                tot += pmatrix[asn][gclusterid]*asn_to_users[asn]
                tot_users += asn_to_users[asn]
            pmatrix_clustered[representative][gclusterid] = tot/tot_users

    return pmatrix_clustered


            



def build_fake_pmatrix_profile(guards, W):
    """
        build a bivariate dicrete distribution for AS n and Guard i
        with a penalty score

        This function builds a fake one for test purpose 
    """
    pmatrix = {}
    tot = 0
    for loc in W:
        if loc not in pmatrix:
            pmatrix[loc] = {}
        for guard in guards:
            #pmatrix[guard][loc] = random.randint(0,1000)
            pmatrix[loc][guard] = 1000 #uniform penalty
        tot += sum(pmatrix[loc].values())
    for guard in guards:
        for loc in W:
            pmatrix[loc][guard] = pmatrix[loc][guard]/tot
    return pmatrix

def model_opt_problem_lastor_shadow(shadow_relay_info, obj_function, out_dir=None, pmatrix_file=None,
        theta=2.0, disable_SWgg=False):
    pass
    with open(shadow_relay_info, "rb") as picklef:
        exitguards_nodes = pickle.load(picklef)
        guards_nodes = pickle.load(picklef)
        exits_nodes = pickle.load(picklef)
        middles_nodes = pickle.load(picklef)
    
    ## Compute G, Wgg, etc.
    G, E, M, D = 0, 0, 0, 0
    G = sum(guard.bwconsensus for guard in guards_nodes)
    E = sum(exit.bwconsensus for exit in exits_nodes)
    D = sum(exitguard.bwconsensus for exitguard in exitguards_nodes)
    M = sum(middle.bwconsensus for middle in middles_nodes)

    SWgg = (E + D)/G #SWgg for Scarce Wgg

    ## Load shadow's penalty matrix
    
    with open(pmatrix_file, "r") as f:
        pmatrix = json.load(f)

    ## model opt problem

    location_aware = LpProblem("Location aware selection", LpMinimize)
    #todo

    objective = LpVariable("L_upper_bound", lowBound = 0)
    # Compute L as affine expressions involving LpVariables
    print("Computing Affine Expressions for L, i.e., \sum W_iR_i")
    L = {}
    for guard in guards_nodes:
        L[guard.name] = LpAffineExpression([(R[loc][guard.name], W[loc]) for loc in W], name="L({})".format(guard.name))

    print("Computing the lpSum of LpAffineExpressions as an objective function... (this can take time)")
    # client-guard relation
    part_one = lpsum([LpAffineExpression([(R[loc][guard.name], W[loc]*pmatrix[loc][guard.name]) for loc in W]) for guard in guards_nodes])
    # guard-guard at the middle position relationship
    part_two = lpsum([LpAffineExpression([(guard.bwconsenus - L[guard.name], pmatrix[guard.name][guard2.name]) for guard2 in guards_nodes if  guard2.name != guard.name]) for guard in guards_nodes])
    # guard-middle relationship
    part_three = lpsum([LpAffineExpression([(guard.bwconsenus - L[guard.name], pmatrix[guard.name][middle.name]) for middle in middles_nodes]) for guard in guards_nodes])
    # middle as guard - exit relationship
    part_four = lpsum([LpAffineExpression([(guard.bwconsenus - L[guard.name], pmatrix[guard.name][exit.name]) for exit in exits_nodes]) for guard in guards_nodes])
    # middle as guard - guardexits relationship
    part_five = lpsum([LpAffineExpression([(guard.bwconsensus - L[guard.name], pmatrix[guard.name][guardexit.name]) for exitguard in exitguards_nodes]) for guard in guards_nodes])
    location_aware += lpSum([part_one, part_two, part_three, part_four, part_five]), "Z"
    print("Done.")
    # Now set of constraints:
    # Location scores must distribute G*Wgg quantity
    print("Computing constraints \sum R_l(i) == G*Wgg")
    for loc in W:
        sum_R = lpSum([R[loc][guard.name] for guard in guards_nodes])
        location_aware += sum_R == G*Wgg, "\sum R(i) == G*Wgg for loc {}".format(loc)
    print("Done.")
    print("Computing constraints L(i) <= BW_i")
    for guard in guards_nodes:
        location_aware += L[guard.name] <= guard.bwconsensus, "L(i) <= BW_i for {}".format(guard.name)

    #No relay gets more than theta times its original selection probability
    print("GPA constraint, using theta = {} and relCost(i) = BW_i/sum_j(BW_j)".format(theta))
    for loc in W:
        for guard in guards_nodes:
            location_aware += R[loc][guard.name] <= theta*guard.bwconsensus*SWgg


def model_opt_problem(W, repre, asn_to_users_file, penalty_vanilla, ns_file, obj_function, cluster_file=None, out_dir=None, pmatrix_file=None,
        theta=2.0, reduced_as_to=None, reduced_guards_to=None, disable_SWgg=False):
    
    network_state = get_network_state(ns_file)
    
    with open(cluster_file, "rb") as f:
        gclusters = pickle.load(f)
    # guardsfp = [relay for relay in network_state.cons_rel_stats if Flag.GUARD in network_state.cons_rel_stats[relay].flags and
    # not Flag.EXIT in network_state.cons_rel_stats[relay].flags]
    
    with open(asn_to_users_file) as f:
        asn_to_users = json.load(f)

    if reduced_guards_to:
        guardsfp = guardsfp[0:reduced_guards_to]
    
    with open(penalty_vanilla) as f:
        penalty_vanilla = json.load(f)
    R = {}
    #Compute total G bandwidth
    G = 0
    #max_cons_weight is going to be used as an upper bound of our
    #objective function
    max_cons_weight = 0
    for prefix in gclusters:
        G += gclusters[prefix].tot_consweight
        if gclusters[prefix].tot_consweight > max_cons_weight:
            max_cons_weight = gclusters[prefix].tot_consweight
    print("Total guard consensus weight: {0}, max observed consenus weight: {1}".format(G, max_cons_weight))
    # Computing E and D for the new Wgg (see Section 4.3 of Waterfilling paper).
    E = 0
    D = 0
    for relay in network_state.cons_rel_stats:
        rel_stat = network_state.cons_rel_stats[relay]
        if Flag.RUNNING not in rel_stat.flags or Flag.BADEXIT in rel_stat.flags\
          or Flag.VALID not in rel_stat.flags:
            continue
        if Flag.EXIT in rel_stat.flags and Flag.GUARD in rel_stat.flags:
            D += rel_stat.consweight
        elif Flag.EXIT in rel_stat.flags:
            E += rel_stat.consweight
    print("E:{}, D:{} and G:{}".format(E,D,G))
    SWgg = (E + D)/G #SWgg for Scarce Wgg
    print("New Wgg value from same strategy as Waterfillign Section 4.3 is: {}".format(SWgg))

    gclustersids = list(gclusters.keys())
    #pmatrix is a discrete bivariate distribution [guard][location] which
    #gives a high score if the path between location and guard is bad
    if not pmatrix_file:
        pmatrix = build_fake_pmatrix_profile(gclustersids, W)
    else:
        print("Loading Penalty matrix")
        with open(pmatrix_file, 'r') as f:
            pmatrix_unclustered = json.load(f)
            #for loc in pmatrix:
            #    for guard in pmatrix[loc]:
            #        #only right for lastor
            #        if pmatrix[loc][guard] == math.inf:
            #            pmatrix[loc][guard] = 3.14*6378137
                        #pmatrix[loc][guard] = sys.maxsize
        pmatrix = produce_clustered_pmatrix(pmatrix_unclustered, repre, asn_to_users, gclusters)
    #Normalize Wgg
    Wgg = network_state.cons_bw_weights['Wgg']/network_state.cons_bwweightscale
    if not disable_SWgg:
        Wgg = SWgg
    print("Wgg={}".format(Wgg))
    #model the problem
    location_aware = LpProblem("Location aware selection", LpMinimize)
    
    for loc in W:
        R[loc] = LpVariable.dicts(loc, gclustersids, lowBound = 0,
                upBound=Wgg*G)
    
    location_aware = LpProblem("Optimal location-aware path selection", LpMinimize)
    # Write a minmax problem as a min of a upper bound
    #
    objective = LpVariable("L_upper_bound", lowBound = 0)
    # Compute L as affine expressions involving LpVariables
    print("Computing Affine Expressions for L, i.e., \sum W_iR_i")
    L = {}
    for gclusterid in gclustersids:
        L[gclusterid] = LpAffineExpression([(R[loc][gclusterid], W[loc]) for loc in W], name="L({})".format(gclusterid))
    print("Done.")
    ##  min_R max_j ( [\sum_{j} L(i)*pmatrix(j)(i)  for i in all guards])
    if obj_function == 1:
        location_aware += objective, "Z" #set objective function
        print("Computing the objective Z with linked constraints")
        #min max L*pmatrix is equal to min Z with Z >= L[guard_i]*Vu
        for gclusterid in gclustersids:
            location_aware += objective >=\
                LpAffineExpression([(R[loc][gclusterid], W[loc]*pmatrix[loc][gclusterid]) for loc in W], name="\sum L[{}]*pmatrix[loc][{}]".format(gclusterid, gclusterid)),\
                "Added constraint Z >= \sum L[{}]*pmatrix[loc][{}] forall loc".format(gclusterid, gclusterid)
                #lpSum([Intermediate[guard]*pmatrix[guard][loc] for loc in W])
            print("Added constraint Z >= \sum L[{}]*pmatrix[loc][{}] forall loc".format(gclusterid, gclusterid))
    ##   min_R max_j ([\sum_{i} W(j)*R(j,i)*pmatrix(j)(i)  for j in all locations])
    elif obj_function == 2:
        location_aware += objective, "Z" #set objective function
        for loc in W:
            location_aware += objective >= \
                LpAffineExpression([(R[loc][gclusterid], W[loc]*pmatrix[loc][gclusterid]) for gclusterid in gclustersids])
            print("Added constraint Z >= \sum_i W({})*R[{}][prefix]*pmatrix[prefix][{}] forall prefix".format(loc, loc, loc))
    ##   min_R (\sum_i \sum_j W(j)*R(i,j)*pmatrix(i,j))
    elif obj_function == 3:
        print("Computing the lpSum of LpAffineExpressions as an objective function... (this can take time)")
        location_aware += lpSum([LpAffineExpression([(R[loc][gclusterid], W[loc]*pmatrix[loc][gclusterid]) for loc in W]) for gclusterid in gclustersids]), "Z"
    ##   min max_j (\sum_i R(i,j)*pmatrix(i,j)) 
    elif obj_function == 4:
        location_aware += objective, "Z" #set objective function
        for loc in W:
            location_aware += objective >= \
                    LpAffineExpression([(R[loc][gclusterid], pmatrix[loc][gclusterid]) for gclusterid in gclustersids])
            print("Added constraint Z >= R[{}][gclusterid]*pmatrix[gclusterid][{}] forall gclustersids".format(loc, loc))
    print("Done.")
    # Now set of constraints:
    # Location scores must distribute G*Wgg quantity
    print("Computing constraints \sum R_l(i) == G*Wgg")
    for loc in W:
        #constraint = LpConstraint(name="\sum R(i) == G*Wgg for loc {}".format(loc), e = lpSum([R[loc][gclusterid] for gclusterid in gclustersids]),
        #        sense=0, rhs=G*Wgg)
        #prob_extension = constraint.makeElasticSubProblem(penalty=100, proportionFreeBound=0.001)
        #location_aware.extend(prob_extension)
        sum_R = lpSum([R[loc][gclusterid] for gclusterid in gclustersids])
        location_aware += sum_R == G*Wgg, "\sum R(i) == G*Wgg for loc {}".format(loc)
        #location_aware += sum_R >= G*Wgg-G*Wgg*ELASTICITY, "\sum R(i) >= G*Wgg-G*Wgg*elasticity for loc {}".format(loc)
        #location_aware += sum_R <= G*Wgg+G*Wgg*ELASTICITY, "\sum R(i) <= G*Wgg-G*Wgg*elasticity for loc {}".format(loc)
    print("Done.")
    print("Computing constraints L(i) <= BW_i")
    for gclusterid in gclustersids:
        location_aware += L[gclusterid] <= gclusters[gclusterid].tot_consweight, "L(i) <= BW_i for gclusterid {}".format(gclusterid)

    ## Missing constraint for theta-GP-Secure TODO
        
    #Temporally wating for theta-GP-secure stuffs
    #No relay gets more than theta times its original selection probability
    print("GPA constraint, using theta = {} and relCost(i) = BW_i/sum_j(BW_j)".format(theta))
    for loc in W:
        for gclusterid in gclustersids:
            location_aware += R[loc][gclusterid] <= theta*gclusters[gclusterid].tot_consweight*Wgg
    print("Done.")
    print("Adding 'no worse than vanilla constraint'")
    for loc in W:
        for ori_loc in repre[loc]:
            location_aware += LpAffineExpression([(R[loc][gclusterid], pmatrix_unclustered[ori_loc][gclusterid]) for gclusterid in gclustersids]) <= penalty_vanilla[ori_loc] * G


    print("Done. Writting ouut")

    # Write problem out:
    if out_dir:
        if reduced_as_to or reduced_guards_to:
            outpath = os.path.join(out_dir, "location_aware_with_obj_{}_reducedas_{}_reducedguard_{}".format(obj_function, reduced_as_to,reduced_guards_to))
        else:
            outpath = os.path.join(out_dir, "location_aware_with_obj_{}_theta_{}".format(obj_function, theta))
    else:
        outpath = "location_aware.pickle"
    #with open(outpath+".pickle", "wb") as f:
        #pickle.dump(location_aware, f, pickle.HIGHEST_PROTOCOL)

    #print("Done. Writting out .lp file")
    #location_aware.writeLP(outpath+".lp")
    print("Done. Writtin out .mps file")
    location_aware.writeMPS(outpath+".mps")
    #location_aware.solve()

    
if __name__ == "__main__":
    args = parser.parse_args()
    ## Compute appropritate values and model the optimization problem
    if args.tor_users_to_location:
        if args.pickle and args.cust_locations:
            W = load_and_compute_W(args.tor_users_to_location, args.cust_locations, args.reduced_as_to)
        elif args.json and args.client_clust_representative:
            W, repre = load_and_compute_W_from_clusterinfo(args.tor_users_to_location, args.client_clust_representative)
        elif args.json:
            W = load_and_compute_W_from_citymap(args.tor_users_to_location)
        elif args.in_shadow:
            ## this W is computed for shadow simulations, not real world.
            W = load_and_compute_W_from_shadowcityinfo(args.tor_users_to_location, args.cityinfo)
        if args.binary_search_theta:
            cur_theta = 1.25
            up_theta = 2
            down_theta = 0.5
            last_positive = 2
            for _ in range(0, 10):
                model_opt_problem(W, repre, args.penalty_vanilla, args.network_state, args.obj_function, theta = cur_theta,
                    cluster_file=args.cluster_file, out_dir=args.out_dir, pmatrix_file=args.pmatrix,
                    reduced_as_to=args.reduced_as_to, reduced_guards_to=args.reduced_guards_to,
                    disable_SWgg=args.disable_SWgg)
                    
                process = Popen(["./check_model", os.path.join(args.out_dir, "location_aware_with_obj_{}.mps".format(args.obj_function))], stdout=PIPE)
                output, err = process.communicate()
                exit_code = process.wait()
                if exit_code == 0:
                    up_theta = cur_theta
                    last_postive = cur_theta
                elif exit_code == 1:
                    down_theta = cur_theta
                else:
                    print("check_model returned something else than -1, 0: {}".format(exit_code))
                cur_theta = (up_theta+down_theta)/2
                print("Next theta tested value: {}".format(cur_theta))

        elif args.in_shadow:
            model_opt_problem_lastor_shadow(W, args.shadow_relay_info, args.obj_function, out_dir=args.out_dir,
                    pmatrix_file=args.pmatrix, theta=args.theta, disable_SWgg=False)
                    
        else:
            model_opt_problem(W, repre, args.tor_users_to_location, args.penalty_vanilla, args.network_state, args.obj_function, theta=args.theta,
                cluster_file=args.cluster_file, out_dir=args.out_dir, pmatrix_file=args.pmatrix,
                reduced_as_to=args.reduced_as_to, reduced_guards_to=args.reduced_guards_to,
                disable_SWgg=args.disable_SWgg)
    ## Load the problem and solve() it
    elif args.load_problem:
        with open(args.load_problem, "rb") as f:
            location_aware = pickle.load(f)
            location_aware.solve(pulp.PULP_CPLEX_CMD(msg=1, path="/home/frochet/.cplex/cplex/bin/x86-64_linux/cplex"))
            pdb.set_trace()
            for v in location_aware.variables():
                print(v.name, "=", v.varValue)
