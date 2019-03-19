import argparse
import sys, os
import pickle
from slim_ases import *
import requests
import pycountry
from datetime import date
import time as dtime
import pdb

"""
This file processes all custumer ASes for a given period and compute all available IPv4. 
We dump on the disk that information for later use.

Note: remaining problem to solve: filtering out ASes and IPs from hosting companies

"""

GETAS_URL = "https://stat.ripe.net/data/country-asns/data.json?resource="
RIS_ASN_URL = "https://stat.ripe.net/data/ris-asns/data.json?list_asns=true&asn_types=o"
RIS_PREFIX = "https://stat.ripe.net/data/ris-prefixes/data.json?resource="

parser = argparse.ArgumentParser(description="Process AS information from Ripe Atlas databases")

parser.add_argument("--fromdate", help="asks the closest information to date, in yyyy-mm-dd. If not provided, asks the latest")
parser.add_argument("--out", help="store pickle file in directory out")

def main(args):

    if args.fromdate:
        time = args.fromdate
    else:
        time = date.today().isoformat()
    
    AsInfo = {}

    for country in pycountry.countries:
        print("Handling {}-{}".format(country.alpha_2, country.name))
        r = requests.get(GETAS_URL+country.alpha_2+"&lod=1&query_time="+time)
        if r.status_code > 200:
            #try 2 more times:
            for _ in range(0,2):
                print("Status code of previous request: {} - Waiting 5 seconds and retry".format(r.status_code))
                dtime.sleep(5)
                r = requests.get(GETAS_URL+country.alpha_2+"&lod=1&query_time="+time)
                if r.status_code < 300:
                    break
            if r.status_code > 200:
                raise ValueError("Something went wrong when querying the server")
        allAses = r.json()
        print(allAses['messages'])
        # Consider only routed AS
        for as_elem in allAses['data']['countries'][0]['routed']:
            newAs = ASCust(int(as_elem), country.alpha_2, [], time)
            AsInfo[as_elem] = newAs

    # Filter out transiting AS (i.e., ASes that do not announce origin for IPs)
    r = requests.get(RIS_ASN_URL+"&query_time="+time)
    if r.status_code > 200:
        for _ in range(0,2):
            print("Status code of previous request: {} - Waiting 5 seconds and retry".format(r.status_code))
            dtime.sleep(5)
            r = requests.get(RIS_ASN_URL+"&query_time="+time)
            if r.status_code < 300:
                break
        if r.status_code > 200:
            raise ValueError("Something went wrong when querying the server")
    allAses = r.json()
    asn_to_delete = [asn for asn in AsInfo if int(asn) not in allAses['data']['asns']['originating']]
    print("We delete {} ASes which never appears on end-paths".format(len(asn_to_delete)))
    for asn in asn_to_delete:
        del AsInfo[asn]

    ## TODO
    # Find a way to filter out hosting ASes (e.g., AS16276)
    ##

    # Query originating v4 prefixes for all ASes
    # and count total of IPs
    for asn in AsInfo:
        r = requests.get(RIS_PREFIX+asn+"&query_time+"+time+"&list_prefixes=true&types=o&af=v4&noise=filter")
        if r.status_code > 200:
            for _ in range(0,2):
                print("Status code of previous request: {} - Waiting 5 seconds and retry".format(r.status_code))
                dtime.sleep(5)
                r = requests.get(RIS_PREFIX+asn+"&query_time+"+time+"&list_prefixes=true&types=o&af=v4&noise=filter")
                if r.status_code < 300:
                    break
            if r.status_code > 200:
                pdb.set_trace()

        asnInfo = r.json()
        #gives all prefix
        if r.status_code < 300:
            AsInfo[asn].prefixes_originated = asnInfo['data']['prefixes']['v4']['originating']
        print("Fetched prefixes for AS{}".format(asn))
        #count IPs for this AS

        AsInfo[asn].compute_num_ipv4_addresses()
    
    #dump allAses info
    print("Dumping all ASes info")
    outpath  = os.path.join(args.out, "allAses-"+time)
    f = open(outpath, 'wb')
    pickle.dump(AsInfo, f, pickle.HIGHEST_PROTOCOL)
    f.close()

    return 0

if __name__ == "__main__":

    args = parser.parse_args()
    sys.exit(main(args))
