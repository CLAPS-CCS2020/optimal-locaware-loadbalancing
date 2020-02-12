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

def parse_client_cluster(filename):
    representative = {}
    with open(filename) as f:
        for line in f:
            tab = line.split('\t')
            representative[tab[0]] = tab[1][:-1].split(',')
    return representative

def parse_alternative_weights(filename):
    alt_weights = {}
    with open(filename) as f:
        for line in f:
            elems = line.split()
            location = elems[0].split("_")[0]
            name = elems[0].split("_")[1]
            if location not in alt_weights:
                alt_weights[location] = {}
            weight = int(elems[1])
            alt_weights[location][name] = weight
    return alt_weights

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


def plot_cdf(vals, label, color=None):
    vals = sorted(vals, key=itemgetter(0))
    ys = list(map(itemgetter(1), vals))
    sum_ys = sum(ys)
    ys = [ y / sum_ys for y in ys ]
    ys = np.cumsum(ys)
    if color:
        plt.plot(list(map(itemgetter(0), vals)), ys,
                label=label, color=color, antialiased=True)
    else:
        plt.plot(list(map(itemgetter(0), vals)), ys,
                label=label, antialiased=True)

def produce_clustered_pmatrix_for_denasa(pmatrix, repre, asn_to_users, guard_ases, exitids):
    """
    Compute a pmatrix  $cluster, $guard_as -> $exit_as -> pvalue
    """
    pmatrix_clustered = {}
    for representative, ases in repre.items():
        for guard_as in guard_ases:
            new_loc = "{}, {}".format(representative, guard_as)
            pmatrix_clustered[new_loc] = {}
            for exit in exitids:
                tot = 0
                tot_users = 0
                for asn in ases:
                    tot += pmatrix["{}, {}".format(asn, guard_as)][exit] * asn_to_users[asn]
                    tot_users += asn_to_users[asn]
                pmatrix_clustered[new_loc][exit] = tot/tot_users
    return pmatrix_clustered
                

def produce_clustered_pmatrix_for_shadow_denasa(pmatrix, repre, relay_to_asn,
        client_distribution, guards, exits):
    pmatrix_clustered = {}
    for representative, cities in repre.items():
        for guard in guards:
            new_W = "{}, {}".format(representative, relay_to_asn[guard])
            pmatrix_clustered[new_W] = {}
            for exit in exits:
                tot = 0
                tot_users = 0
                for city in cities:
                    tot += pmatrix["{}, {}".format(city, relay_to_asn[guard])][relay_to_asn[exit]] * client_distribution[city]
                    tot_users += client_distribution[city]
                pmatrix_clustered[new_W][relay_to_asn[exit]] = tot/tot_users
    return pmatrix_clustered



def produce_clustered_pmatrix(pmatrix, repre, asn_to_users, guards):
    """
    Compute a pmatrix cluster -> guard -> pvalue as the weighted sum
    of penalties given all ases of a given cluster, according to their number
    of users
    """
    pmatrix_clustered = {}
    for representative, ases in repre.items():
        pmatrix_clustered[representative] = {}
        for guard in guards:
            tot = 0
            tot_users = 0
            for asn in ases:
                tot += pmatrix[asn][guard]*asn_to_users[asn]
                tot_users += asn_to_users[asn]
            pmatrix_clustered[representative][guard] = tot/tot_users

    return pmatrix_clustered


def timestamp(t):
    """Returns UNIX timestamp"""
    td = t - datetime.datetime(1970, 1, 1, tzinfo=pytz.UTC)
    ts = td.days*24*60*60 + td.seconds
    return ts

def handle_edge_case(keys, weights, m):

    n = len(keys)
    j = len(list(filter(lambda w: w > 0, [weights[ky] for ky in keys])))

    wprime = dict.fromkeys(keys, 0.0)

    if j < m:
        for ky in keys:
            if weights[ky] > 0:
                wprime[ky] = (1/m)
            else:
                wprime[ky] = (1 / m) * ((m - j) / (n - j))
        return True, wprime
    else:
        return False, wprime

def tille_pr(keys, weights, m):

    is_edge_case, wprime = handle_edge_case(keys, weights, m)
    if is_edge_case:
        return wprime

    k = m

    s = set(keys)
    denom = sum(weights.values())

    def helper():
        wsum = sum([weights[x_i] for x_i in s])

        for x_i in s:
            wprime[x_i] = (weights[x_i] * k / wsum)

    updated = True
    while updated:
        helper()

        updated = False

        remove = []
        for x_i in s:
            if wprime[x_i] > 1:
                wprime[x_i] = 1
                remove.append(x_i)
                updated = True
                k = (k - 1)
        for x in remove:
            s.remove(x)

    sum_wprime = sum(wprime.values())
    for x in keys:
        wprime[x] = wprime[x] / sum_wprime

    return wprime 
