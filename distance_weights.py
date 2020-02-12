import sys, os
import argparse
import json
import pandas
from util import parse_alternative_weights
import math

parser = argparse.ArgumentParser(description="")
parser.add_argument("alternative_weights", help="alternative_weights file")
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
    guards = {}
    for Index, row in relays.iterrows():
        if row['IsGuard'] and not row['IsExit']:
            guards[row['Name']] = row['ConsensusWeight']
    balanced_pr = compute_pr(guards)

    alt_weights = parse_alternative_weights(args.alternative_weights)
    with open(args.city_trait) as f:
        cityinfo = json.load(f)

    loc_distance = {}
    for loc in alt_weights:
        pr_relays = compute_pr(alt_weights[loc])
        loc_distance[loc] = compute_distance(pr_relays, balanced_pr)
    
    sorted_distlist = [(k, loc_distance[k]) for k in sorted(loc_distance,
                                                            key=loc_distance.get,
                                                            reverse=True)]
    for i in range(0,20):
        print("city: {0}, info:{1}".format(sorted_distlist[i][0], cityinfo[sorted_distlist[i][0]]))

