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
from util import parse_client_cluster, produce_clustered_pmatrix
from bandwidth_weights import *
from weights_optimization import load_and_compute_W_from_clusterinfo

parser = argparse.ArgumentParser(description="Produce a file containing weights\
        which can be loaded by clients within our shadow simulations")

sub = parser.add_subparsers(help="Conversion type", dest="sub")

##Counter-Raptor
cr_parser = sub.add_parser("CR", help="Convert to Counter-Raptor wooh!")
cr_parser.add_argument('--relchoice', help="path to file containing choice of\
        relays for the simulation we expect to run")
cr_parser.add_argument('--resilience', help="Contains CR's resilience\
        information for each of Shadow's location. This resilience file is\
        bounded to relchoice, since it must be computed for this shadow topology")

cr_parser.add_argument('--client_distribution', help="path to file containing\
        shadow locations {'citycode':weight}")
cr_parser.add_argument('--alpha', type=float, default=0.5)
cr_parser.add_argument('--outpath')
cr_parser.add_argument('--outname')

claps_cr_parser = sub.add_parser("CLAPS_CR", help="Convert to CLAPS Counter-Raptort")
claps_cr_parser.add_argument('--resilience', help="Contains the path to the penalty matrix")
claps_cr_parser.add_argument('--relchoice', help="path to file containing choice of\
        relays for the simulation we expect to run")
claps_cr_parser.add_argument('--cluster_repre', help="path to the file containing cluster representative")
claps_cr_parser.add_argument('--client_distribution', help="path to file containing\
        shadow locations {'citycode':weight}")
claps_cr_parser.add_argument('--sol_file', help="path to the solver solution file of the CLAPS counter raptor")
claps_cr_parser.add_argument('--outpath', help="")
claps_cr_parser.add_argument('--outname', help="")
claps_cr_parser.add_argument('--alpha', type=float, default=0.5)

##DeNASA
denasa_parser = sub.add_parser("DeNASA", help="Convert to DeNASA weights")
denasa_parser.add_argument('--relchoice', help="path to file containing choice of\
        relays for the simulation we expect to run")
#TODO

##LASTor
lastor_parser = sub.add_parser("LASTor", help="Convert to LASTor weights")
#TODO

_TESTING = True

def parse_relaychoice(relchoice):
    """
        input: path to csv file containg al relays
    """
    return pandas.read_csv(relchoice)

def output(locationsinfo, outpath, outname):
    """
        locationsinfo is dict {k:v} with k locationid and v a list containing
        relay_name, weight_for_guard, weight_for_middle and weight_for_exit to
        output
    """
    with open(os.path.join(outpath, outname), "w") as f:
        for location in locationsinfo:
            for value in locationsinfo[location]:
                f.write("{0}_{1}\n".format(location, locationsinfo[location][value]))

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
        if row['Consensus(KB/s)'] > max_guard_consensus_weight:
            max_guard_consensus_weight = row['Consensus(KB/s)']
    return max_guard_consensus_weight, guards

def compute_cr_weights(args):
    
    locationsinfo = {}
    relays = parse_relaychoice(args.relchoice)
    with open(args.resilience) as f:
        resilience = json.load(f)
    with open(args.client_distribution) as f:
        locations = json.load(f)
    ## compute the max guard consenus weights, as done in the
    ## official counter-raptor C implementation
    max_guard_consensus_weight, guards = _get_max_guardconsweight(relays)
    print("Max guard value is {}".format(max_guard_consensus_weight))
    print("Number of guards: {}".format(len(guards)))
    ## pick random resilience to compute weights
    for location in locations:
        locationsinfo[location] = {}
        for guard in guards.values():
            thisguard_res = (1-resilience[location][guard['Name']])*max_guard_consensus_weight
            locationsinfo[location][guard['Name']] = "{0} {1} {2} {3}".format(
                    guard['Name'],
                    int(round(args.alpha*thisguard_res +
                    (guard['Consensus(KB/s)'])*(1-args.alpha))),
                    -1,
                    -1)
    return locationsinfo

def compute_claps_cr_weights(args):
    locationsinfo = {}
    relays = parse_relaychoice(args.relchoice)
    with open(args.resilience) as f:
        penalties = json.load(f)
    with open(args.client_distribution) as f:
        locations = json.load(f)
    
    solinfo = {}
    #parse information from the solution file
    with open(args.sol_file) as f:
        for line in f:
            loc, relayname = line.split(" ")[0].split("_")
            if loc not in solinfo:
                solinfo[loc] = {}
            weight = int(line.split(" ")[1])
            solinfo[loc][relayname] = weight

    W = load_and_compute_W_from_clusterinfo(locations, args.cluster_file)
    #Re-compute L! TODO need cluster location distribution
    

    for location in solinfo
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
                int(round(Wmg*guard['Consensus(KB/s)']/10000.0)),
                -1)
    return locationsinfo
if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.sub == "CR":
        locationsinfo = compute_cr_weights(args)
        output(locationsinfo, args.outpath, args.outname)
    elif args.sub == "CLAPS_CR":
        locationsinfo = compute_claps_cr_weights(args)
        output(locationsinfo, args.outpath, args.outname)
    

