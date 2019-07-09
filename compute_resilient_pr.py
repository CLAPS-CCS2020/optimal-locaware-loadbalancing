import sys, os
import argparse
import pdb
import matplotlib.pyplot as plt
from util import *
import json
from stem import Flag
import pickle

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
    print("Tot weight: {}".format(tot_weight))
    with open(args.pmatrix) as f:
        pmatrix = json.load(f)
    with open(args.asn_to_users) as f:
        asn_to_users = json.load(f)

    cluster_info = parse_client_cluster(args.clusterdescr)
    
    network_state_file = get_network_state(args.ns_file)
    print("Loaded all data")
    ## Compute the CDF of client resiliency for each ASes
    guards = [guard for guard in network_state_file.cons_rel_stats if Flag.GUARD in
                network_state_file.cons_rel_stats[guard].flags and Flag.EXIT not
                in network_state_file.cons_rel_stats[guard].flags]
    resilience = []
    for representative in weights.keys():
        for asn in cluster_info[representative]:
            avg_resi = 0
            for guardfp in guards:
                if guardfp in weights[representative]: # if not, it means that this guard a 0 prob
                    avg_resi += weights[representative][guardfp]*(1-pmatrix[asn][guardfp])
            resilience.append((avg_resi/tot_weight, 1))
        print("Computed metric for representative {}".format(representative))
    


    ## Plot CDF
    if os.path.exists("figuredata.pickle"):
        with open("figuredata.pickle", "rb") as f:
            ax = pickle.load(f)
    else:
        ax = plt.subplot(111)
        plt.xlabel("Expected path resilience", fontsize=18)
        plt.ylabel("CDF", fontsize=18)
        ## Computing origin Counter-Raptor weights (only once =)
        alpha = 0.25
        cr_resilience = []
        tot_crweight = 0
        max_bw = 0
        for guardfp in guards:
            if network_state_file.cons_rel_stats[guardfp].consweight > max_bw:
                max_bw = network_state_file.cons_rel_stats[guardfp].consweight
        for guardfp in guards:
            tot_crweight += max_bw*alpha*(1-pmatrix[asn][guardfp])+(1-alpha)*(network_state_file.cons_rel_stats[guard].consweight)
        for representative in weights.keys():
            for asn in cluster_info[representative]:
                avg_rsi = 0
                for guardfp in guards:
                    avg_rsi += (max_bw*alpha*(1-pmatrix[asn][guardfp])+(1-alpha)*(network_state_file.cons_rel_stats[guard].consweight)) * (1-pmatrix[asn][guardfp])
                cr_resilience.append((avg_rsi/tot_crweight, 1))
        plot_cdf(cr_resilience, "Counter-Raptor alpha=0.25")
    plot_cdf(resilience, args.clusterdescr.split("/")[-1])
    plt.legend()
    filename = args.clusterdescr.split("/")[-1]
    plt.savefig("cr_optimized_{}.png".format(filename))
    with open("figuredata.pickle", 'wb') as f:
        pickle.dump(ax, f)
    plt.close()


if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))

