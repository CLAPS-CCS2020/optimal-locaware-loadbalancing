import argparse
import sys, os
import pickle
from slim_ases import *
import requests
import pycountry
from datetime import date

"""
This file processes all custumer ASes and compute all available IPv4  We group the information
per country. 


Note: remaining problem to solve: filtering out ASes and IPs from hosting companies

"""

GETAS_URL = "https://stat.ripe.net/data/country-asns/data.json?resource="
RIS_ASN_URL = "https://stat.ripe.net/data/ris-asns/data.json?list_asns=true&asn_types=o"
PREFIX_ROUTING_CONSISTENCY_URL = ""

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
            raise ValueError("Something went wrong when querying the server")
        allAses = r.json()
        # Consider only routed AS
        for as_elem in allAses['data']['countries'][0]['routed']:
            newAs = ASCust(int(as_elem), country.alpha_2, [], time)
            AsInfo[as_elem] = newAs

    # Filter out only transiting AS
    r = requests.get(RIS_ASN_URL+"&query_time="+time)
    if r.status_code > 200:
        raise ValueError("Something went wrong when querying the server")
    allAses = r.json()
    asn_to_delete = [asn for asn in AsInfo if asn not in allAses['data']['asns']['originating']]
    print("We delete {} ASes which never appears on end-paths".format(len(asn_to_delete)))
    for asn in asn_to_delete:
        del AsInfo[asn]




    return 0

if __name__ == "__main__":

    args = parser.parse_args()
    sys.exit(main(args))
