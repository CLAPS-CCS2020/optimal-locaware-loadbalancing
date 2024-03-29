import sys, os
import argparse
from slim_ases import *
from pulp import *
import numpy as np
import pickle
from util import get_network_state, produce_clustered_pmatrix, produce_clustered_pmatrix_for_denasa, parse_sol_file, produce_clustered_pmatrix_for_shadow_denasa
from process_ases import GETAS_URL
import requests
import random
import pdb
from stem import Flag
import json
import math
from subprocess import Popen, PIPE
from bandwidth_weights import *

"""
This script receives data relative to Client-to-country distribution, AS information and penalty
paths information 

apply a minmax optimiztion problem and ouputs an allocation of scores for each
location which should satify the constraints of our problem (see README.md, procedure B)

"""

common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument("--tor_users_to_location", help="path to the pickle file containing the distribution of Tor users per country")
# parser.add_argument("--pickle", action="store_true", default=False)
# parser.add_argument("--json", action="store_true", default=False)
common_parser.add_argument("--disable_SWgg", action="store_true", default=False)
# parser.add_argument("--in_shadow", action="store_true", default=False)
# parser.add_argument("--cust_locations", help="path to the file containing the distribution of IPs per customer AS")
common_parser.add_argument("--obj_function", type=int, help="Choice of objective function")
common_parser.add_argument("--client_clust_representative", type=str, help="Ryan's clusterization file for ASes in one AS representative")
common_parser.add_argument("--pmatrix", type=str, default=None, help="Penalty matrix")
common_parser.add_argument("--penalty_vanilla", default=None, type=str, help="Vanilla penalty vector for each location")
# parser.add_argument("--load_problem", help="filepth with problem to solve if already computed")
common_parser.add_argument("--out_dir", help="out dir to save the .lp file")
# parser.add_argument("--binary_search_theta", action='store_true', default=False)
common_parser.add_argument("--theta", type=float, help="set theta value for gpa", default=5.0)

##Parse instances
parser = argparse.ArgumentParser(description="Model a LP problem and output a "
                                 ".mps representation of it")
sub = parser.add_subparsers(dest="sub")

cr_parser = sub.add_parser("CR", parents=[common_parser], help="For Counter-Raptor security analysis")
cr_shadow_parser = sub.add_parser("CR_SHADOW", parents=[common_parser], help="For Counter-Raptor shadow "
                                  "simulation")

denasa_parser = sub.add_parser("DeNASA", parents=[common_parser], add_help=False) #help="For DeNASA security analysis")
denasa_shadow_parser = sub.add_parser("DeNASA_SHADOW", parents=[common_parser], help="For DeNASA shadow "
                               "simulations")


## For Counter Raptor security analysis
cr_parser.add_argument("--network_state", help="filepath to the network state containing Tor network's data (shadow_relay_dump in case of shadow simulation")
cr_parser.add_argument("--cluster_file", type=str, help="Pickle file of clustered guards")
cr_parser.add_argument("--reduced_as_to", type=int, help="for test purpose, gives the number of ASes to keep")
cr_parser.add_argument("--reduced_guards_to", type=int, help="for test purpose, gives the number of guards to keep")

## For Counter Raptor Shadow simulations
cr_shadow_parser.add_argument("--shadow_relay_dump", help="Path to a .json file"
                              " containing information about the network")

## For DeNASA security analysis
denasa_parser.add_argument("--network_state", help="filepath to the network state containing Tor network's data (shadow_relay_dump in case of shadow simulation")
denasa_parser.add_argument("--cluster_file", type=str, help="Pickle file of clustered guards")
## For DeNASA Shadow simulation
denasa_shadow_parser.add_argument("--shadow_relay_dump", help="Path to a .json file"
                              " containing information about the network")
## For DeNASA g&e security analysis
denasa_exit_parser = sub.add_parser("DeNASA_EXIT", parents=[denasa_parser],
                                    help="For DeNASA g&e security analysis")
denasa_exit_parser.add_argument("--deNasa_sol_guards", help="filepath to the solver"
                                " output file for denasa guard weight LP"
                                " problem")
## For DeNASA g&e shadow 
denasa_shadow_exit_parser = sub.add_parser("DeNASA_SHADOW_EXIT", parents=[denasa_parser],
                                            help="For DeNASA g&e shadow sims")
denasa_shadow_exit_parser.add_argument("--relay_to_asn", help="filepath to a json dict "
                                "that contains a map from fingerprints to asn")
denasa_shadow_exit_parser.add_argument("--deNasa_sol_guards", help="filepath to the solver"
                                " output file for denasa guard weight LP"
                                " problem")
denasa_shadow_exit_parser.add_argument("--shadow_relay_dump", help="Path to a .json file"
                              " containing information about the network")

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


def load_and_compute_W_from_denasa_clusterinfo(city_to_users_file, clusterinfo, asn_to_city):
    with open(city_to_users_file) as f:
        city_to_users = json.load(f)
    W = {}
    tot = sum(city_to_users.values())
    repre = {}
    with open(clusterinfo) as f:
        for line in f:
            tab = line.split('\t')
            cur_city = asn_to_city[tab[0]][0]
            W[cur_city] = 0
            repre[cur_city] = [asn_to_city[x][0] for x in tab[1][:-1].split(',') if x in asn_to_city]
            for city in repre[cur_city]:
                W[cur_city] += city_to_users[city]
            W[cur_city] /= tot
    return W, repre 

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
    if (nbr_repre != len(asn_to_users)):
        print("looks like we don't have the same number of locations within cluster representatives and the location info for the simulation: {} vs {}.\
               So it is possible that we are not able to cluster some city appearing in Shadow simulations for example :( "\
        .format(nbr_repre, len(asn_to_users)))
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

def load_and_compute_from_solfile(W, guards, pathsolfile):
    """
        L(i) = \sum W_j*R_ij for each guard i
    """
    parsed_weights = parse_sol_file(pathsolfile)
    if len(parsed_weights) != len(W):
        print("We do not have the same AS set size "
    "between ASes parsed from the DeNASA guard sol file and W?")
    L = {}
    for  guard in guards:
        L[guard] = 0
        for loc in W:
            if guard in parsed_weights[loc]:
                L[guard] += W[loc]*parsed_weights[loc][guard]
    return L

def build_fake_fp_to_asn(network_state, repre):
    fp_to_asn = {}
    keys = repre.keys()
    for relay in network_state.cons_rel_stats:
        fp_to_asn[relay] = random.randint(1, 65536)
    return fp_to_asn


def build_fake_pmatrix_profile(nodes, locs, randrange):
    """
        build a bivariate dicrete distribution for AS n and Guard i
        with a penalty score

        This function builds a fake one for test purpose 
    """
    pmatrix = {}
    for loc in locs:
        if loc not in pmatrix:
            pmatrix[loc] = {}
        for node in nodes:
            pmatrix[loc][node] = random.randint(randrange[0], randrange[1])
    return pmatrix


def model_opt_problem_for_shadow(W, repre, client_distribution,
        penalty_vanilla, shadow_relay_dump, obj_function, out_dir=None,
        pmatrix_file=None, theta=5.0, disable_SWgg=False):
    with open(client_distribution) as f:
        client_distribution = json.load(f)

    with open(shadow_relay_dump) as f:
        relays = json.load(f)
    with open(penalty_vanilla) as f:
        penalty_vanilla = json.load(f)

    guards = {}
    exits = {}
    guardexits = {}
    middles = {}
    G, E, D, M = 0, 0, 0, 0
    for name, relinfo in relays.items():
        if relinfo[4] and relinfo[5]:
            guardexits[name] = relinfo
            D+=relinfo[6]
        elif relinfo[4]:
            guards[name] = relinfo
            G+=relinfo[6]
        elif relinfo[5]:
            exits[name] = relinfo
            E += relinfo[6]
        else:
            middles[name] = relinfo
            M+=relinfo[6]
    R = {}
    max_cons_weight = 0.0
    for name, relinfo in guards.items():
        if relinfo[6] > max_cons_weight:
            max_cons_weight = relinfo[6]
    print("Total guard consensus weight: {0}, max observed consenus weight: {1}".format(G, max_cons_weight))

    print("E:{}, D:{} and G:{}".format(E,D,G))
    bwweight = BandwidthWeights()
    casename, Wgg, Wgd, Wee, Wed, Wmg, Wme, Wmd = bwweight.recompute_bwweights(G, M, D, E, G+M+D+E)
    print("casename: {}, Wgg:{}, Wgd:{}, Wee:{}, Wmg:{}, Wme:{}, Wmd:{}".format(casename, Wgg,
        Wgd, Wee, Wed, Wmg, Wme, Wmd))
    Wgg = Wgg/10000.0
    print("Wgg is {}".format(Wgg))
    SWgg = (E + D)/G #SWgg for Scarce Wgg
    print("New Wgg value from same strategy as Waterfillign Section 4.3 is: {}".format(SWgg))

    with open(pmatrix_file, 'r') as f:
        pmatrix_unclustered = json.load(f)
    pmatrix = produce_clustered_pmatrix(pmatrix_unclustered, repre, client_distribution, guards)
    
    if not disable_SWgg:
        Wgg = SWgg
    print("Using Wgg={}".format(Wgg))
    #model the problem
    location_aware = LpProblem("Location aware selection", LpMinimize)
    
    for loc in W:
        R[loc] = LpVariable.dicts(loc, list(guards.keys()), lowBound = 0,
                upBound=Wgg*G)
    
    location_aware = LpProblem("Optimal location-aware path selection", LpMinimize)
    # Write a minmax problem as a min of a upper bound
    #
    objective = LpVariable("L_upper_bound", lowBound = 0)
    # Compute L as affine expressions involving LpVariables
    print("Computing Affine Expressions for L, i.e., \sum W_iR_i")
    L = {}
    for guard in guards.keys():
        L[guard] = LpAffineExpression([(R[loc][guard], W[loc]) for loc in W],
                                      name="L({})".format(guard))
    print("Done.")
    ##  min_R max_j ( [\sum_{j} L(i)*pmatrix(j)(i)  for i in all guards])
    if obj_function == 3:
        print("Computing the lpSum of LpAffineExpressions as an objective function... (this can take time)")
        location_aware += lpSum([LpAffineExpression([(R[loc][guard], W[loc]*pmatrix[loc][guard]) for loc in W]) for guard in guards]), "Z"
    ##   min max_j (\sum_i R(i,j)*pmatrix(i,j)) 
    else:
        print("Objective function unsupported")
        sys.exit(1)

    print("Done.")
    # Now set of constraints:
    # Location scores must distribute G*Wgg quantity
    print("Computing constraints \sum R_l(i) == G*Wgg")
    for loc in W:
        sum_R = lpSum([R[loc][guard] for guard in guards])
        location_aware += sum_R == G*Wgg, "\sum R(i) == G*Wgg for loc {}".format(loc)
        #location_aware += sum_R >= G*Wgg-G*Wgg*ELASTICITY, "\sum R(i) >= G*Wgg-G*Wgg*elasticity for loc {}".format(loc)
        #location_aware += sum_R <= G*Wgg+G*Wgg*ELASTICITY, "\sum R(i) <= G*Wgg-G*Wgg*elasticity for loc {}".format(loc)
    print("Done.")
    print("Computing constraints L(i) <= BW_i")
    for guard in guards:
        location_aware += L[guard] <= guards[guard][6], "L(i) <= BW_i for guard {}".format(guard)

    ## Missing constraint for theta-GP-Secure TODO
        
    #Temporally wating for theta-GP-secure stuffs
    #No relay gets more than theta times its original selection probability
    print("GPA constraint, using theta = {} and relCost(i) = BW_i/sum_j(BW_j)".format(theta))
    for loc in W:
        for guard in guards:
            location_aware += R[loc][guard] <= theta*guards[guard][6]*Wgg
    print("Done.")
    print("Adding 'no worse than vanilla constraint'")
    for loc in W:
        for ori_loc in repre[loc]:
            location_aware += LpAffineExpression([(R[loc][guard], pmatrix_unclustered[ori_loc][guard]) for guard in guards]) <= penalty_vanilla[ori_loc] * G

    print("Done. Writting ouut")

    # Write problem out:
    if out_dir:
        outpath = os.path.join(out_dir, "location_aware_with_obj_{}_theta_{}".format(obj_function, theta))
    else:
        outpath = "location_aware.pickle"
    #location_aware.writeLP(outpath+".lp")
    print("Done. Writtin out .mps file")
    location_aware.writeMPS(outpath+".mps")
    #location_aware.solve()

def model_opt_problem_shadow_for_denasa_exit(W, repre, L, client_distribution,
        penalty_vanilla, shadow_relay_dump, obj_function, relay_to_asn_file, out_dir=None,
        pmatrix_file=None, theta=5.0, disable_SWgg=False):

    with open(relay_to_asn_file) as f:
        relay_to_asn = json.load(f)
        
    with open(client_distribution) as f:
        client_distribution = json.load(f)

    with open(shadow_relay_dump) as f:
        relays = json.load(f)
    with open(penalty_vanilla) as f:
        penalty_vanilla = json.load(f)

    guards = {}
    exits_only = {}
    exits = {}
    guardexits = {}
    middles = {}
    G, E, D, M = 0, 0, 0, 0
    for name, relinfo in relays.items():
        if relinfo[4] and relinfo[5]:
            guardexits[name] = relinfo
            exits[name] = relinfo
            D+=relinfo[6]
        elif relinfo[4]:
            guards[name] = relinfo
            G+=relinfo[6]
        elif relinfo[5]:
            exits[name] = relinfo
            exits_only[name] = relinfo
            E += relinfo[6]
        else:
            middles[name] = relinfo
            M+=relinfo[6]
    R = {}
    #max_cons_weight = 0.0
    #for name, relinfo in guards.items():
    #    if relinfo[6] > max_cons_weight:

    location_aware = LpProblem("Optimal location-aware path selection for exit deNasa weights", LpMinimize)
    #cli_loc,guardname -> {exitname : penalty}
    join_W = {}
    for loc_cli in W:
        for guard in guards:
            join_W["{},{}".format(loc_cli, guard)] = W[loc_cli]*L[guard]
    tot = sum(join_W.values())
    for loc in join_W:
        join_W[loc] = join_W[loc]/tot

    R = {}
    for loc in join_W:
        R[loc] = LpVariable.dicts(loc, list(exits.keys()), lowBound = 0,
                                  upBound=E+D)

    if not pmatrix_file:
        print("WARN: building fake pmatrix")
        pmatrix = build_fake_pmatrix_profile([relay_to_asn[exitid] for exitid in exits.keys()], join_W, [0, 8])
    else:
        with open(pmatrix_file, 'r') as f:
            pmatrix_unclustered = json.load(f)
        print("Unclustered pmatrix loaded... Now clustering that matrix for"
              " representative.. Ugly O(m*n*q*r)")
        pmatrix = produce_clustered_pmatrix_for_shadow_denasa(pmatrix_unclustered,
                                                       repre, relay_to_asn, client_distribution,
                                                       guards, exits)
    LE = {}
    for exit in exits:
        LE[exit] = LpAffineExpression([(R[loc][exit], join_W[loc]) for loc in join_W], name="LE({})".format(exit))
    
    objective = LpVariable("L_upper_bound", lowBound = 0)
    print("Computing the lpSum of LpAffineExpressions as an objective function... ")
    location_aware += lpSum([LpAffineExpression([(R[loc][exit],
                                                  join_W[loc]*pmatrix["{}, {}".format(loc.split(',')[0], relay_to_asn[loc.split(',')[1]])][relay_to_asn[exit]]) for loc in join_W]) for exit in exits]), "Z"

    ## Computing constraints
    print("Computing constraints \sum R_l(i) == E+D (under the assumption E+D is scarce, this is the right way to load balance)")
    for loc in join_W:
        #location_aware.extend(prob_extension)
        sum_R = lpSum([R[loc][exitid] for exitid in exits])
        location_aware += sum_R == E+D, "\sum R(i) == E+D for loc {}".format(loc)

    print("Computing constraints LE(i) <= BW_i")
    for exitid in exits:
        location_aware += LE[exitid] <= exits[exitid][6]

    print("GPA constraint, using theta = {} and relCost(i) = BW_i/sum_j(BW_j)".format(theta))
    for loc in join_W:
        for exitid in exits:
            # Assuming exit relays are used at 100% in exit position
            location_aware += R[loc][exitid] <= theta*exits[exitid][6]
    print("Done.")
    # print("Adding 'no worse than vanilla constraint'")
    # for loc in W:
        # for ori_loc in repre[loc]:
            # location_aware += LpAffineExpression([(R[loc][exitid], pmatrix_unclustered[ori_loc][relay_to_asn[exitid]]) for exitid in exitids]) <= penalty_vanilla[ori_loc] * (E+D)
    
    write_to_mps_file(location_aware, out_dir, obj_function, theta)

def model_opt_problem_for_denasa_exit(W, repre, L, asn_to_users_file,
                                      penalty_vanilla, ns_file,
                                      fp_to_asn_file,
                                      obj_function, cluster_file=None,
                                      out_dir=None, pmatrix_file=None,
                                      theta=5.0, reduced_as_to=None,
                                      reduced_guards_to=None,
                                      disable_SWgg=False):
    
    network_state = get_network_state(ns_file)
    
    with open(asn_to_users_file) as f:
        asn_to_users = json.load(f)
    
    E, D = compute_tot_pos_bandwidths(network_state)
    
    #We take all exits and guard+exits relays
    exitids = []
    for relay in network_state.cons_rel_stats:
        rel_stats = network_state.cons_rel_stats[relay]
        if Flag.EXIT in rel_stats.flags:
            exitids.append(relay)
    
    if fp_to_asn_file:
        with open(fp_to_asn_file) as f:
            fp_to_asn = json.load(f)
    else:
        print("WARN: using a fake fp_to_asn generated info")
        fp_to_asn = build_fake_fp_to_asn(network_state, repre)

    #LP variables
    
    # Affine expression LE = \sum WGE_i*R_i ; will be used in the constraint to
    # ensure load balancing of exit relays.
    # TO BE DECIDED: Per client Cluster Exit distribution or Per client
    # Cluster-guard, exit pairs distribution? The later would offer much better
    # network level end-to-end security but would theoritically allows guard discovery attacks, which
    # kinda sucks. One option would be to cluster guard relays in the same logic
    # we initially clustered clients
    
    ##L_norm will be used for re-computing the penalty matrix
    location_aware = LpProblem("Optimal location-aware path selection for exit deNasa weights", LpMinimize)
    guard_ases = {} 
    print("Computing the list of all guard ases")
    for guard in guards:
        if fp_to_asn[guard] not in guard_ases:
            guard_ases[fp_to_asn[guard]] = [guard]
        else:
            guard_ases[fp_to_asn[guard]].append(guard)

    join_location = {} 
    for loc_cli in W:
        for loc_guard in guard_ases:
            join_location["{}, {}".format(loc_cli, loc_guard)] = W[loc_cli]*sum([L[guard] for guard in guard_ases[loc_guard]])
    ## Norm join_location:
    tot = sum(join_location.values())
    for loc in join_location:
        join_location[loc] = join_location[loc]/tot

    #distribution of client weights and denasa Guard weights
    R = {}
    for loc in join_location:
        R[loc] = LpVariable.dicts(loc, exitids, lowBound = 0,
                                  upBound=E+D)
    if not pmatrix_file:
        print("WARN: building fake pmatrix")
        pmatrix = build_fake_pmatrix_profile([fp_to_asn[exitid] for exitid in exitids], join_location, [0, 1])
    else:
        with open(pmatrix_file, 'r') as f:
            pmatrix_unclustered = json.load(f)
        print("Unclustered pmatrix loaded... Now clustering that matrix for"
              " representative.. Ugly O(m*n*q*r)")
        pmatrix = produce_clustered_pmatrix_for_denasa(pmatrix_unclustered,
                                                       repre, asn_to_users,
                                                       guard_ases, exitids)

    # pmatrix should now be the form of "Client cluster AS, Guard AS", exit AS => penalty
    # We now want Client Cluster AS, exit AS => penalty by summing over all
    # guard AS with the p
    LE = {}
    for exitid in exitids:
        LE[exitid] = LpAffineExpression([(R[loc][exitid], join_location[loc]) for loc in join_location], name="LE({})".format(exitid))
    
    objective = LpVariable("L_upper_bound", lowBound = 0)
    print("Computing the lpSum of LpAffineExpressions as an objective function... ")
    location_aware += lpSum([LpAffineExpression([(R[loc][exitid],
                                                  join_location[loc]*pmatrix[loc][fp_to_asn[exitid]]) for loc in join_location]) for exitid in exitids]), "Z"

    ## Computing constraints
    print("Computing constraints \sum R_l(i) == E+D (under the assumption E+D is scarce, this is the right way to load balance)")
    for loc in join_location:
        #location_aware.extend(prob_extension)
        sum_R = lpSum([R[loc][exitid] for exitid in exitids])
        location_aware += sum_R == E+D, "\sum R(i) == E+D for loc {}".format(loc)

    print("Computing constraints LE(i) <= BW_i")
    
    for exitid in exitids:
        location_aware += LE[exitid] <= network_state.cons_rel_stats[exitid].consweight

    print("GPA constraint, using theta = {} and relCost(i) = BW_i/sum_j(BW_j)".format(theta))
    for loc in join_location:
        for exitid in exitids:
            # Assuming exit relays are used at 100% in exit position
            location_aware += R[loc][exitid] <= theta*network_state.cons_rel_stats[exitid].consweight
    print("Done.")
    # print("Adding 'no worse than vanilla constraint'")
    # for loc in W:
        # for ori_loc in repre[loc]:
            # location_aware += LpAffineExpression([(R[loc][exitid], pmatrix_unclustered[ori_loc][exitid]) for exitid in exitids]) <= penalty_vanilla[ori_loc] * (E+D)
    
    write_to_mps_file(location_aware, out_dir, obj_function, theta)

def model_opt_problem(W, repre, asn_to_users_file, penalty_vanilla, ns_file, obj_function, cluster_file=None, out_dir=None, pmatrix_file=None,
        theta=5.0, reduced_as_to=None, reduced_guards_to=None, disable_SWgg=False):
    
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
    E, D = compute_tot_pos_bandwidths(network_state)
    print("E:{}, D:{} and G:{}".format(E,D,G))
    SWgg = (E + D)/G #SWgg for Scarce Wgg
    print("New Wgg value from same strategy as Waterfillign Section 4.3 is: {}".format(SWgg))

    gclustersids = list(gclusters.keys())
    #pmatrix is a discrete bivariate distribution [guard][location] which
    #gives a high score if the path between location and guard is bad
    if not pmatrix_file:
        pmatrix = build_fake_pmatrix_profile(gclustersids, W, [0, 1000])
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
    #for loc in W:
    #    for ori_loc in repre[loc]:
    #       location_aware += LpAffineExpression([(R[loc][gclusterid], pmatrix_unclustered[ori_loc][gclusterid]) for gclusterid in gclustersids]) <= penalty_vanilla[ori_loc] * G

    write_to_mps_file(location_aware, out_dir, obj_function, theta)

def compute_tot_pos_bandwidths(network_state):
    E, D = 0, 0
    for relay in network_state.cons_rel_stats:
        rel_stat = network_state.cons_rel_stats[relay]
        if Flag.RUNNING not in rel_stat.flags or Flag.BADEXIT in rel_stat.flags\
          or Flag.VALID not in rel_stat.flags:
            continue
        if Flag.EXIT in rel_stat.flags and Flag.GUARD in rel_stat.flags:
            D += rel_stat.consweight
        elif Flag.EXIT in rel_stat.flags:
            E += rel_stat.consweight
    return E, D

def write_to_mps_file(location_aware, out_dir, obj_function, theta,
                      reduced_guards_to=None, reduced_as_to=None):

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

    if args.sub == "CR":
        W, repre = load_and_compute_W_from_clusterinfo(args.tor_users_to_location, args.client_clust_representative)
        model_opt_problem(W, repre, args.tor_users_to_location, args.penalty_vanilla, args.network_state, args.obj_function, theta=args.theta,
            cluster_file=args.cluster_file, out_dir=args.out_dir, pmatrix_file=args.pmatrix,
            reduced_as_to=args.reduced_as_to, reduced_guards_to=args.reduced_guards_to,
            disable_SWgg=args.disable_SWgg)
    elif args.sub == "CR_SHADOW":
        W, repre = load_and_compute_W_from_clusterinfo(args.tor_users_to_location, args.client_clust_representative)
        model_opt_problem_for_shadow(W, repre, args.tor_users_to_location,
                                     args.penalty_vanilla,
                                     args.shadow_relay_dump, args.obj_function,
                                     theta=args.theta,
                                     pmatrix_file=args.pmatrix,
                                     out_dir=args.out_dir,
                                     disable_SWgg=args.disable_SWgg)
        
    elif args.sub == "DeNASA":
        W, repre = load_and_compute_W_from_clusterinfo(args.tor_users_to_location, args.client_clust_representative)
        model_opt_problem(W, repre, args.tor_users_to_location, args.penalty_vanilla, args.network_state, args.obj_function, theta=args.theta,
            cluster_file=args.cluster_file, out_dir=args.out_dir, pmatrix_file=args.pmatrix,
            disable_SWgg=args.disable_SWgg)
    elif args.sub == "DeNASA_EXIT":
        ##
        print("Computing Client representative weighting ...")
        W, repre =\
        load_and_compute_W_from_clusterinfo(args.tor_users_to_location,
                                            args.client_clust_representative)
        #extract guards
        network_state = get_network_state(args.network_state)
        guards = {}
        for relay in network_state.cons_rel_stats:
            rel_stat = network_state.cons_rel_stats[relay]
            if Flag.GUARD in rel_stat.flags and Flag.EXIT not in rel_stat.flags:
                guards[relay] = rel_stat
        print("Compute total guard usage in entry and middle, for each guard (known from previous "
              "CLAPS deNasa results) ...")
        L = load_and_compute_from_solfile(W, guards, args.deNasa_sol_guards)
        model_opt_problem_for_denasa_exit(W, repre, L,
                                          args.tor_users_to_location,
                                          args.penalty_vanilla,
                                          args.network_state, args.fp_to_asn,
                                          args.obj_function, theta=args.theta,
                                          cluster_file=args.cluster_file,
                                          out_dir=args.out_dir,
                                          pmatrix_file=args.pmatrix,
                                          disable_SWgg=args.disable_SWgg)
    elif args.sub == "DeNASA_SHADOW":
        W, repre = load_and_compute_W_from_clusterinfo(args.tor_users_to_location, args.client_clust_representative)
        model_opt_problem_for_shadow(W, repre, args.tor_users_to_location,
                                     args.penalty_vanilla,
                                     args.shadow_relay_dump,
                                     args.obj_function,
                                     theta=args.theta,
                                     pmatrix_file=args.pmatrix,
                                     out_dir=args.out_dir,
                                     disable_SWgg=args.disable_SWgg)
    elif args.sub == "DeNASA_SHADOW_EXIT":
        print("Computing Client representative weighting ...")
        W, repre =\
        load_and_compute_W_from_clusterinfo(args.tor_users_to_location,
                                            args.client_clust_representative)
        #extract guards
        guards = {}
        with open(args.shadow_relay_dump) as f:
            relays = json.load(f)
            for name, relinfo in relays.items():
                if relinfo[4] and not relinfo[5]:
                    guards[name] = relinfo
        L = load_and_compute_from_solfile(W, guards, args.deNasa_sol_guards)
        model_opt_problem_shadow_for_denasa_exit(W, repre, L,
                                                args.tor_users_to_location,
                                                args.penalty_vanilla,
                                                args.shadow_relay_dump,
                                                args.obj_function,
                                                args.relay_to_asn,
                                                theta=args.theta,
                                                pmatrix_file=args.pmatrix,
                                                out_dir=args.out_dir,
                                                disable_SWgg=args.disable_SWgg)
        
    else:
        sys.exit(-1)

