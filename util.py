import sys, os
import matplotlib.pyplot as plt
import pickle
from slim_desc import *
import datetime, pytz
from operator import itemgetter
import numpy as np

class TorOptions:
    """Stores some parameters set by Tor."""
    # given by #define ROUTER_MAX_AGE (60*60*48) in or.h
    router_max_age = 60*60*48
    default_bwweightscale = 10000

def parse_sol_file(filename):
    weights = {}
    with open(filename) as f:
        ## parse solfile
        f.readline()#skip header
        line = f.readline() 
        while line:
            tab = line.split()
            col1 = tab[1].split("_")
            representative = col1[0]
            fingerprint = col1[1]
            if representative in weights:
                weights[representative][fingerprint] = float(tab[2])
            else:
                weights[representative] = {}
                weights[representative][fingerprint] = float(tab[2])
            line = f.readline()
    return weights
        
    with open(args.pmatrix) as f:
        pmatrix = json.load(f)
    with open(args.asn_to_users) as f:
        asn_to_users = json.load(f)

    cluster_info = parse_client_cluster(args.clusterdescr)
    

def parse_client_cluster(filename):
    representative = {}
    with open(filename) as f:
        for line in f:
            tab = line.split('\t')
            representative[tab[0]] = tab[1][:-1].split(',')
    return representative

def get_network_states(network_state_files):
    """
        Inspired from github.com/torps/pathsim.py and yields
        slim_desc.NetworkState objects
    """
    for ns_file in network_state_files:
        if (ns_file is not None):
            # get network state variables from file
            network_state = get_network_state(ns_file)
        else:
            network_state = None
        yield network_state

    

def get_network_state(ns_file):
    """Reads in network state file, returns slim_desc.NetworkState object."""
    
    cons_rel_stats = {}
    with open(ns_file, 'rb') as nsf:
        consensus = pickle.load(nsf)
        new_descriptors = pickle.load(nsf)
        hibernating_statuses = pickle.load(nsf)
        
    # set variables from consensus
    cons_valid_after = timestamp(consensus.cons_valid_after)            
    cons_fresh_until = timestamp(consensus.cons_fresh_until)
    cons_bw_weights = consensus.cons_bw_weights
    if (consensus.cons_bwweightscale == None):
        cons_bwweightscale = TorOptions.default_bwweightscale
    else:
        cons_bwweightscale = consensus.cons_bwweightscale
    for relay in consensus.relays:
        if (relay in new_descriptors):
            cons_rel_stats[relay] = consensus.relays[relay]
    
    return NetworkState(cons_valid_after, cons_fresh_until, cons_bw_weights,
        cons_bwweightscale,  hibernating_statuses, cons_rel_stats, new_descriptors)


def plot_cdf(vals, label):
    vals = sorted(vals, key=itemgetter(0))
    ys = list(map(itemgetter(1), vals))
    sum_ys = sum(ys)
    ys = [ y / sum_ys for y in ys ]
    ys = np.cumsum(ys)
    plt.plot(list(map(itemgetter(0), vals)), ys,
            label=label, antialiased=True)

def timestamp(t):
    """Returns UNIX timestamp"""
    td = t - datetime.datetime(1970, 1, 1, tzinfo=pytz.UTC)
    ts = td.days*24*60*60 + td.seconds
    return ts
