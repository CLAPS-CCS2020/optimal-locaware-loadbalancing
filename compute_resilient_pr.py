import sys, os
import argparse
import pdb
from matplotlib import pyplot as plt
from util import *

parser = argparse.ArgumentParser("")
parser.add_argument("--solfile", type=str, help="path to .sol file containing the result of the solution of the lp problem")
parser.add_argument("--pmatrix", type=str, help="path to the pmatrix json file")
parser.add_argument("--asn_to_users", type=str, help="path to the asn_to_users json file")
parser.add_argument("--clusterdescr", type=str, help="path to the file containing the clustering representative with the linked AS")

def main(args):
    with open(args.solfile) as f:
        ## parse solfile
        pass
    with open(args.pmatrix) as f:
        pmatrix = json.load(f)
    with open(args.asn_to_users) as f:
        asn_to_users = json.load(f)

    cluster_info = parse_client_cluster(args.clusterdescr)
    
    ## Compute the CDF of client resiliency for each ASes
    for as in asn_to_users.keys():
        pass

    ## Plot CDF

if __name__ == "__main__":
    sys.exit(main(parser.parse_args()))

