
import sys, os
import argparse
from slim_ases import *
from pulp import *

"""
This script receives data relative to Client-to-country distribution, AS information and vulnerable
paths information 

apply a minmax optimiztion problem and ouputs an allocation of scores for each
location which should satify the constraints of our problem (see README.md, procedure B)

"""

parser = argparse.ArgumentParser(description="")

parser.add_argument("--tor_users_to_country", help="path to the pickle file containing the distribution of Tor users per country")
parser.add_argument("--cust_ases", help="path to the pickle file containing the distribution of IPs per customer AS")
parser.add_argument("--load_problem", help="filepth with problem to solve if already computed")

if __name__ == "__name__":
    args = parser.parse_args()
    ## Compute appropritate values and modelize the optimization problem
    if args.tor_users_to_country and args.cust_ases:
        pass
    ## Load the problem and solve() it
    elif args.load_problem:
        pass
