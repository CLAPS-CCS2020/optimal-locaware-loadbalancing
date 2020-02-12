import sys, os
import argparse
import json
import pandas
from util import parse_alternative_weights, parse_alternative_weights_ge
import math

parser = argparse.ArgumentParser(description="")
parser.add_argument("alternative_weights", help="alternative_weights file")
parser.add_argument("--alternative_weights_ge", help="path to alternative_weights_ge if needed")
parser.add_argument("relchoice", help="csv file containing relay info")
parser.add_argument("city_trait", help="json file for city information")


def parse_relaychoice(relchoice):
    """
        input: path to csv file containg al relays
    """
    return pandas.read_csv(relchoice)

def compute_pr(weights):
    tot = sum(weights.values())
    return {k:v/tot for k, v in weights.items()}

def compute_distance(unbalanced, balanced):
    distance = 0
    for guard in unbalanced:
        distance += abs(unbalanced[guard]-balanced[guard])
    return distance

if __name__ == "__main__":
    args = parser.parse_args()
    relays = parse_relaychoice(args.relchoice)
    guards, exits = {}, {}
    for Index, row in relays.iterrows():
        if row['IsGuard'] and not row['IsExit']:
            guards[row['Name']] = row['ConsensusWeight']
        elif row['IsExit']:
            exits[row['Name']] = row['ConsensusWeight']

    if args.alternative_weights_ge:
        balanced_pr_ge = compute_pr(exits)
    balanced_pr = compute_pr(guards)

    alt_weights = parse_alternative_weights(args.alternative_weights)
    if args.alternative_weights_ge:
        alt_weights_ge = parse_alternative_weights_ge(args.atlernative_weights_ge)
        alt_weights_mixed = {}
        for location in alt_weights_ge:
            tot_weights[location] = 0
            alt_weights_mixed[location] = {}
            for guard in alt_weights_ge[location]:
                tot_weights[location] += alt_weights[location][guard]
                for exit in alt_weights_ge[location][guard]:
                    if exit not in alt_weights_mixed[location]:
                        alt_weights_mixed[location][exit] = 0
                    else:
                        alt_weights_mixed[location][exit] += alt_weights[location][guard] * alt_weights_ge[location][guard][exit] 
        for location in alt_weights_mixed:
            for exit in alt_weights_mixed[location]:
                alt_weights_mixed[location][exit] /= tot_weights[location]



    with open(args.city_trait) as f:
        cityinfo = json.load(f)

    loc_distance = {}
    for loc in alt_weights:
        if args.alternative_weights_ge:
            pr_relays_ge = compute_pr(alt_weights_mixed[loc])
        pr_relays = compute_pr(alt_weights[loc])
        loc_distance[loc] = compute_distance(pr_relays, balanced_pr)
        if args.alternative_weights_ge:
            loc_distance_ge[loc] = compute_distance(pr_relays_ge,
                                                    balanced_pr_ge)
   
    sorted_distlist = [(k, loc_distance[k]) for k in sorted(loc_distance,
                                                            key=loc_distance.get,
                                                            reverse=True)]
    if args.alternative_weights_ge:
        sorted_distlist_ge = [(k, loc_distance_ge[k] for k in
                               sorted(loc_distance_ge, key=loc_distance_ge.get,
                                   reverse=True)]
    print("Looking bad regarding guards, first is worst")
    for i in range(0,20):
        print("city: {0}, info:{1}".format(sorted_distlist[i][0], cityinfo[sorted_distlist[i][0]]))
    print("Looking bad regarding exits, first is worst")
        print("city: {0}, info:{1}".format(sorted_distlist_ge[i][0],
                                           cityinfo[sorted_distlist[i][0]]))



