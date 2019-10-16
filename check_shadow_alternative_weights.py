"""
    Check whether the information we end up to give to Tor is correct

"""

import argparse
import pandas
from util import parse_client_cluster
from bandwidth_weights import *

parser = argparse.ArgumentParser()

parser.add_argument("Relchoice")
parser.add_argument("Client distribution")
parser.add_argument("Representative")

def parse_relaychoice(relchoice)
    return pandas.read_csv(relchoice)

def parse_alt_weights(alt_weights_file):
    alt_weights = {}
    with open(alt_weights_file) as f:
        for line in f:
            location = int(line.split("_")[0])
            guard = line.split(" ")[0].split("_")[1]
            weight = int(line.split(" ")[1])
            if location not in alt_weights:
                alt_weights[location] = {}
            alt_weights[location][guard] = weight


if __name__ == "__main__":

    args = parser.parse_args()

    relays = parse_relaychoice(args.relchoice)
    alt_weights = parse_alt_weights(args.alt_weights)

    with open(args.client_distribution) as f:
        W = json.load(f)

    representatives = parse_client_cluster(args.cluster_file)
    

    G,M,E,D = 0, 0, 0, 0
    for relay in relays.values():
        if "relayexitguard" in relay['Name']:
            D += relay['ConsensusWeight']
        elif "relayguard" in relay['Name']:
            G += relay['ConsensusWeight']
        elif "relayexit" in relay['Name']:
            E += relay['ConsensusWeight']
        else:
            M += relay['ConsensusWeight']

    ## Compute bandwidth weights
    bw_weights =  BandwidthWeights()
    cons_bw_weights = bw_weights.recompute_bwweights(G, M, E, D, G+M+E+D)
    print(f"bw_weights: {cons_bw_weights}")
    G_cr = 0
    for relay in relays.values():
        if "relayguard" in relay['Name']:
            ## Let's check whether L[i] makes sense
            guard = relay['name']
            consweight = relay['ConsensusWeight']
            prop_cli_weight = 0
            for location in alt_weights:
                guard_weight = alt_weights[location][guard]
                location_weight = 0
                for clust_loc in representative[location]:
                    location_weight += W[clust_loc]
                prop_cli_weight += guard_weight*location_weight
            print(f'cons_weight:{consweight}; L[{guard}]={prop_cli_weight}')
            G_cr += prop_cli_weight

    print("Looking at tot bandwidth:")
    print("Vanilla Network: G:{G*cons_bw_weights['Wgg']/10000.0}, "
          "M:{M+G*cons_bw_weights['Wmg']/10000.0}, E+D:{E+D}")
    print("This network: G:{G_cr}, M: {M+G-G_cr}, E+D:{E+D}")


