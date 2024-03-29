import argparse
import sys, os
from process_consensuses import timestamp, TorOptions
from slim_desc import *
from stem import Flag
import pickle
import pdb
from util import get_network_states, get_network_state
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

def main(ns_files, args):
    """
        Computes the distribution of users per country
    """
    countries = {}
    descriptors = {}

    for network_state in get_network_states(ns_files):
        descriptors_to_look_at = {}
        for descfp in network_state.descriptors:
            #for any new descriptor:
            if descfp not in descriptors:
                descriptors_to_look_at[descfp] = network_state.descriptors[descfp]
            else:
                ## update the desc if the extra_info_digest is new
                if descriptors[descfp].extra_info_digest !=\
                   network_state.descriptors[descfp].extra_info_digest and\
                   network_state.descriptors[descfp].extra_info_digest is not None:
                    descriptors_to_look_at[descfp] = network_state.descriptors[descfp]
        print("we look at {} descriptors".format(len(descriptors_to_look_at)))
        relays = network_state.cons_rel_stats
        for relayfp in descriptors_to_look_at.keys():
            if relayfp not in relays:
                print("Relays {} not in our list?".format(relayfp))
            if Flag.GUARD in relays[relayfp].flags and Flag.EXIT not in relays[relayfp].flags:
                if descriptors_to_look_at[relayfp].dirreqv2_unique_ips is not None:
                    for countrycode, numreqs in descriptors_to_look_at[relayfp].dirreqv2_unique_ips.items():
                        if countrycode not in countries:
                            countries[countrycode] = numreqs
                        else:
                            countries[countrycode] += numreqs
                if descriptors_to_look_at[relayfp].dirreqv3_unique_ips is not None:
                    for countrycode, numreqs in descriptors_to_look_at[relayfp].dirreqv3_unique_ips.items():
                        if countrycode not in countries:
                            countries[countrycode] = numreqs
                        else:
                            countries[countrycode] += numreqs

        descriptors.update(network_state.descriptors)

    country_percent = {}
    #delete unsupported "Anonymous country" and "Satellite Provider"
    if "a1" in countries:
        del countries["a1"]
    if "a2" in countries:
        del countries["a2"]
    if "??" in countries:
        del countries["??"]
    tot = sum(countries.values())
    for country, value in countries.items():
        country_percent[country] = value/tot
    printable_countries = country_percent.items()
    print(sorted(printable_countries, key=lambda x: x[1], reverse=True))
    # Dumping all country information
    outpath = os.path.join(args.out_dir, 'countries_info_from_{}-{}-{}_to_{}-{}-{}'.format(
        args.start_year, args.start_month, args.start_day, args.end_year, args.end_month,
        args.end_day))
    f = open(outpath, "wb")
    pickle.dump(country_percent, f, pickle.HIGHEST_PROTOCOL)





if __name__ == "__main__":
    args = parser.parse_args()
    month = args.start_month
    day = args.start_day
    pathnames = []
    for year in range(args.start_year, args.end_year+1):
        while (year < args.end_year and month <= 12) or (month <= args.end_month):
            prepend_month = '0' if month <= 9 else ''
            while (year < args.end_year and month <= 12 and day <= args.end_day) or\
                  (year == args.end_year and month == args.end_month and day <= args.end_day):
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
    pdb.set_trace()
    sys.exit(main(pathnames, args))

