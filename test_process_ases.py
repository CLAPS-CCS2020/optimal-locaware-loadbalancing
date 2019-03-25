import requests
from slim_ases import *


RIS_PREFIX = "https://stat.ripe.net/data/ris-prefixes/data.json?resource="

def test_compute_num_addresses():
    asn = 2611
    r = requests.get(RIS_PREFIX+str(asn)+"&query_time=2019-03-25&list_prefixes=true&types=o&af=v4&noise=filter")
    asnInfo = r.json()
    newAs = ASCust(asn, "be", asnInfo['data']['prefixes']['v4']['originating'], "2019-03-25")
    newAs.compute_num_ipv4_addresses()
    assert(newAs.num_ipv4_addresses == 995840) #source: https://bgp.he.net/AS2611
    asn = 16276
    r = requests.get(RIS_PREFIX+str(asn)+"&query_time=2019-03-25&list_prefixes=true&types=o&af=v4&noise=filter")
    asnInfo = r.json()
    newAs = ASCust(asn, "fr", asnInfo['data']['prefixes']['v4']['originating'], "2019-03-25")
    newAs.compute_num_ipv4_addresses()
    assert(newAs.num_ipv4_addresses == 2962688) #source: https://bgp.he.net/AS16276
    print("Test Ok.")


if __name__ == "__main__":
    test_compute_num_addresses()


