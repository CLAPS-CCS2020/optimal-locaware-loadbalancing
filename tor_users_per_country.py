import argparse
import sys, os
from process_consensuses import timestamp, TorOptions
from slim_desc import *
from stem import Flag
import pickle

"""

Takes as input network states for a given period and output the distribution of observed users
per countries

"""

parser = argparse.ArgumentParser(description="Computes the distribution of client per country")
parser.add_argument('--start_day', type=int, required=True)
parser.add_argument('--end_day', type=int, required=True)
parser.add_argument('--start_year', type=int, required=True)
parser.add_argument('--start_month', type=int, required=True)
parser.add_argument('--end_year', type=int, required=True)
parser.add_argument('--end_month', type=int, required=True)
parser.add_argument('--in_dir', required=True)
parser.add_argument('--out_dir', required=True)

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
    with open(ns_file, 'r') as nsf:
        consensus = pickle.load(nsf)
        new_descriptors = pickle.load(nsf)
        hibernating_statuses = pickle.load(nsf)
        
    # set variables from consensus
    cons_valid_after = timestamp(consensus.valid_after)            
    cons_fresh_until = timestamp(consensus.fresh_until)
    cons_bw_weights = consensus.bandwidth_weights
    if (consensus.bwweightscale == None):
        cons_bwweightscale = TorOptions.default_bwweightscale
    else:
        cons_bwweightscale = consensus.bwweightscale
    for relay in consensus.relays:
        if (relay in new_descriptors):
            cons_rel_stats[relay] = consensus.relays[relay]
    
    return NetworkState(cons_valid_after, cons_fresh_until, cons_bw_weights,
        cons_bwweightscale, cons_rel_stats, hibernating_statuses,
new_descriptors)

def main(ns_files, args):
    """
        Computes the distribution of users per country
    """
    countries = {}
    descriptors = {}

    for network_state in get_network_states(ns_files):
        descriptors.update(network_state.descriptors)
        relays = network_state.cons_rel_stats
        
        for relayfp in relays:
            if Flag.GUARD in relays[relayfp].flags and Flag.EXIT not in relays[relayfp].flags:
                for countrycode, numreqs in descriptors[relayfp].dirreqv2_unique_ips.items():
                    if countrycode not in countries:
                        countries[countrycode] = numreqs
                    else:
                        countries[countrycode] += numreqs
                for countrycode, numreqs in descriptors[relayfp].dirreqv3_unique_ips.items():
                    if countrycode not in countries:
                        countries[countrycode] = numreqs
                    else:
                        countries[countrycode] += numreqs
    # Dumping all country information
    outpath = os.path.join(args.out_dir, 'countries_info_from_{}-{}-{}_to_{}-{}-{}'.format(
        args.start_year, args.start_month, args.start_day, args.end_year, args.end_month,
        args.end_day))
    f = open(outpath, "wb")
    pickle.dump(countries, f, pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
    args = parser.parse_args()
    month = args.start_month
    day = args.start_day
    pathnames = []
    for year in range(args.start_year, args.end_year+1):
        while (year < args.end_year and month <= 12) or (month <= args.end_month):
            prepend_month = '0' if month <= 9 else ''
            while (year < args.end_year and month <= 12 and day <= 31) or\
                  (day <= args.end_day):
                prepend_day = '0' if day <= 9 else ''
                for dirpath, dirnames, fnames in os.walk(args.in_dir):
                    for fname in fnames:
                        if "{0}-{1}{2}-{3}{4}".format(year, prepend_month,
                                                      month, prepend_day, day) in fname:
                            pathnames.append(os.path.join(dirpath, fname))
                day+=1
            day = 1
            month+=1
        month = 1
    pathnames.sort()
    sys.exit(main(ns_files))

