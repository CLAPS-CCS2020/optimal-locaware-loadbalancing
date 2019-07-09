import sys, os
import argparse
import pdb
import matplotlib.pyplot as plt
from util import *
import scipy
import seaborn as sns

parser = argparse.ArgumentParser("")
parser.add_argument("--solfile", type=str, help="path to .sol file containing the result of the solution of the lp problem")
parser.add_argument("--pmatrix", type=str, help="path to the pmatrix json file")
parser.add_argument("--asn_to_users", type=str, help="path to the asn_to_users json file")
parser.add_argument("--clusterdescr", type=str, help="path to the file containing the clustering representative with the linked AS")
parser.add_argument("--ns_file", type=str, help="network file info")

def main(args):

    weights = parse_sol_file(args.solfile)
    tot_weight = 0
    for representative, guards in weights.items():
        for guard in guards:
            tot_weight += weights[representative][guard]
        break

    with open(args.pmatrix) as f:
        pmatrix = json.load(f)
    with open(args.asn_to_users) as f:
        asn_to_users = json.load(f)

    cluster_info = parse_client_cluster(args.clusterdescr)
    
    network_state_file = get_network_state(args.ns_file)
    ## Compute the CDF of client resiliency for each ASes
    resilience = []
    for representative in weights.keys():
        for asn in cluster_info[representative]:
            avg_resi = 0
            for guardfp in guards:
                avg_resi += weight[representative][guardfp]*pmatrix[asn][guardf]
            resilience.append(avg_rsi/tot_weight)

    ## Plot CDF
    plt.figure()
    plot_cdf(resilience, args.clusterdescr)
    plt.savefig()
    plt.close()


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))

