"""

Takes either as input a .sol file produced by the linear program resolver and
consensus information, or information regarding an existing selection algorithm
(such as Counter-RAPTOR, DeNASA, LASTor, etc) and produce the set of weights
for each clients to choose from.

OUTPUT to a file: lines of:

    locationid relay_name weight_for_guard weight_for_middle weight_for_exit

"""
import os, sys
import argparse
import json, pandas
from util import parse_client_cluster, produce_clustered_pmatrix, tille_pr
from bandwidth_weights import *
from weights_optimization import load_and_compute_W_from_clusterinfo
import pdb

parser_c = argparse.ArgumentParser(description="Produce a file containing weights\
        which can be loaded by clients within our shadow simulations")

sub = parser_c.add_subparsers(help="Conversion type", dest="sub")

##Counter-Raptor
cr_parser = sub.add_parser("CR", help="Convert to Counter-Raptor wooh!")
cr_parser.add_argument('--relchoice', help="path to file containing choice of\
        relays for the simulation we expect to run")
cr_parser.add_argument('--penalties', help="Contains CR's shadow penalties\
        information for each of Shadow's location. This penalties file is\
        bounded to relchoice, since it must be computed for this shadow topology")

cr_parser.add_argument('--client_distribution', help="path to file containing\
        shadow locations {'citycode':weight}")
cr_parser.add_argument('--alpha', type=float, default=0.5)
cr_parser.add_argument('--outpath')
cr_parser.add_argument('--outname')

claps_cr_parser = sub.add_parser("CLAPS_CR", help="Convert to CLAPS Counter-Raptort")
claps_cr_parser.add_argument('--relchoice', help="path to file containing choice of\
        relays for the simulation we expect to run")
claps_cr_parser.add_argument('--cluster_repre', help="path to the file containing cluster representative")
claps_cr_parser.add_argument('--client_distribution', help="path to file containing\
        shadow locations {'citycode':weight}")
claps_cr_parser.add_argument('--sol_file', help="path to the solver solution file of the CLAPS counter raptor")
claps_cr_parser.add_argument('--outpath', help="")
claps_cr_parser.add_argument('--outname', help="")

##DeNASA
denasa_g_parser = sub.add_parser("DeNASA_G", help="Convert to DeNASA G weights")
denasa_g_parser.add_argument('--relchoice', help="path to file containing choice of\
        relays for the simulation we expect to run")

denasa_ge_parser = sub.add_parser("CLAPS_DeNASA_GE", help="Convert to DeNASA GE weights")
denasa_ge_parser.add_argument('--relchoice', help="path to file containing choice of\
        relays for the simulation we expect to run")
denasa_ge_parser.add_argument('--sol_file_g', help="path to the solver solution file of denasa g select")
denasa_ge_parser.add_argument('--sol_file_ge', help="")
denasa_ge_parser.add_argument('--client_distribution', help="path to file containing\
        shadow locations {'citycode':weight}")
denasa_ge_parser.add_argument('--cluster_repre', help="path to the file containing cluster representative")
denasa_ge_parser.add_argument('--outpath')
denasa_ge_parser.add_argument('--outname', default="alternative_weights")


##LASTor
lastor_parser = sub.add_parser("LASTor", help="Convert to LASTor weights")
#TODO

_TESTING = True

def parse_relaychoice(relchoice):
    """
        input: path to csv file containg al relays
    """
    return pandas.read_csv(relchoice)

def output(locationsinfos, sub,  outpath, outname):
    """
        locationsinfo is dict {k:v} with k locationid and v a list containing
        relay_name, weight_for_guard, weight_for_middle and weight_for_exit to
        output
    """
    def _write_out(locationsinfos, outpath, outname):
        with open(os.path.join(outpath, outname), "w") as f:
            for location in locationsinfos:
                for value in locationsinfos[location]:
                    f.write("{0}_{1}\n".format(location, locationsinfos[location][value]))

    if sub == "CLAPS_DeNASA_GE":
        _write_out(locationsinfos[0], outpath, outname+"_g")
        _write_out(locationsinfos[1], outpath, outname+"_ge")
    else:
        _write_out(locationsinfos, outpath, outname)


def _get_max_guardconsweight(relays):
    max_guard_consensus_weight = 0
    guards = {}
    for Index, row in relays.iterrows():
        #Pick guard-only relays
        if row['IsGuard'] is False:
            continue
        if row['IsGuard'] is True and row['IsExit'] is True:
            continue
        guards[row['Name']] = row
        if row['ConsensusWeight'] > max_guard_consensus_weight:
            max_guard_consensus_weight = row['ConsensusWeight']
    return max_guard_consensus_weight, guards

def _parse_solution(solfile):
    solinfo = {}
    with open(solfile) as f:
        f.readline() #skip header
        for line in f:
            loc, relayname = line.split()[1].split("_")
            if loc not in solinfo:
                solinfo[loc] = {}
            weight = int(round(float(line.split()[2])))
            solinfo[loc][relayname] = weight
    return solinfo

def compute_cr_weights(args):
    
    locationsinfo = {}
    relays = parse_relaychoice(args.relchoice)
    with open(args.penalties) as f:
        penalties = json.load(f)
    with open(args.client_distribution) as f:
        locations = json.load(f)
    ## compute the max guard consenus weights, as done in the
    ## official counter-raptor C implementation
    max_guard_consensus_weight, guards = _get_max_guardconsweight(relays)
    print("Max guard value is {}".format(max_guard_consensus_weight))
    print("Number of guards: {}".format(len(guards)))
    
    # Compute blending first, then tillé, then we put the weithts in the right format
    weights = {}
    for location in locations:
        weights[location] = {}
        for guard in guards.values():
            thisguard_res = (1-penalties[location][guard['Name']])*max_guard_consensus_weight
            weights[location][guard['Name']] = int(round(args.alpha*thisguard_res + (guard['ConsensusWeight'])*(1-args.alpha)))
        min_weight = min(weights[location].values())
        tsize = int(round(len(guards)*0.1))
        tille_probs = tille_pr(weights[location].keys(), weights[location], tsize)
        min_prob = min(tille_probs.values())
        
        for guard in guards.values():
            weights[location][guard['Name']] = int(round(tille_probs[guard['Name']]/min_prob * min_weight))

    for location in locations:
        locationsinfo[location] = {}
        for guard in guards.values():
            locationsinfo[location][guard['Name']] = "{0} {1} {2} {3}".format(
                    guard['Name'],
                    weights[location][guard['Name']],
                    -1,
                    -1)
    return locationsinfo

def compute_claps_g_weights(args, solfile):
    locationsinfo = {}
    relays = parse_relaychoice(args.relchoice)
    max_guard_consensus_weight, guards = _get_max_guardconsweight(relays)
    #parse information from the solution file
    solinfo = _parse_solution(solfile)
    W, repre = load_and_compute_W_from_clusterinfo(args.client_distribution, args.cluster_repre)
    #Re-compute L! TODO need cluster location distribution
    print("DEBUG: Should be one: {}".format(sum(W.values())))
    L = {}
    for relayname in guards:
        L[relayname] = 0
        for loc in W:
            if relayname in solinfo[loc]: #is not inside if the value is 0
                L[relayname] += W[loc]*solinfo[loc][relayname]
        print("L[{}] is {}".format(relayname, L[relayname]))

    for location in solinfo:
        locationsinfo[location] = {}
        for guard in guards.values():
            if guard['Name'] in solinfo[location]:
                thisguard_weight = solinfo[location][guard['Name']]
            else:
                thisguard_weight = 0
            ## look into parsed sol file
            locationsinfo[location][guard['Name']] = "{0} {1} {2} {3}".format(
                guard['Name'],
                thisguard_weight,
                int(round(guard['ConsensusWeight']-L[guard['Name']])),
                -1)
    return locationsinfo

def compute_claps_denasa_ge_weights(args):
    locationinfo_ge = {}
    relays = parse_relaychoice(args.relchoice)
    solinfo_denasa_ge = _parse_solution(args.sol_file_ge)
    # sol file does not necessary have all guards and all exits
    guards = {}
    exits = {}
    for Index, row in relays.iterrows():
        #Pick guard-only relays
        if row['IsExit'] is True:
            exits[row['Name']] = row['ConsensusWeight']
        elif row['IsGuard'] is True:
            guards[row['Name']] = row['ConsensusWeight']
    for location in solinfo_denasa_ge:
        locationinfo_ge[location] = {}
        for exit in exits:
            if exit in solinfo_denasa_ge[location]:
                thisexit_weight = solinfo_denasa_ge[location][exit]
            else:
                thisexit_weight = 0
            locationinfo_ge[location][exit] = "{0} {1} {2} {3}".format(exit, 
                    -1,
                    -1,
                    int(round(thisexit_weight)))
    return locationinfo_ge

if __name__ == "__main__":
    args = parser_c.parse_args()
    
    if args.sub == "CR":
        locationsinfo = compute_cr_weights(args)
    elif args.sub == "CLAPS_CR":
        locationsinfo = compute_claps_g_weights(args, args.sol_file)
    elif args.sub == "CLAPS_DeNASA_GE":
        locationsinfo_g = compute_claps_g_weights(args, args.sol_file_g)
        locationsinfo_ge = compute_claps_denasa_ge_weights(args)
        locationsinfo = [locationsinfo_g, locationsinfo_ge]
    else:
        print("Not Implemented")
    output(locationsinfo, args.sub, args.outpath, args.outname)

