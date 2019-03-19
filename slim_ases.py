"""

used to store extracted information from Ripe Atlas BGP databases and keep overall memory 
consumption somewhat low

"""

import ipaddress 

class ASCust:
    """
    Contains information about custumer AS for all countries. Custumer ASes are defined
    by the origin of routes, i.e., if they contain routes orignated from this AS.
    """
    def __init__(self, as_name, country, prefixes_originated, date_info):
        self.as_name = as_name
        self.country = country
        self.prefixes_originated = prefixes_originated
        self.date_info = date_info
        self.num_ipv4_addresses = None

    def compute_num_ipv4_addresses(self):
        """
        Computes the number of availale IP addresses from
        origin prefixes
        """
        if self.num_ipv4_addresses is None:
            allIps = []
            self.num_ipv4_addresses = 0
            for prefix in self.prefixes_originated:
                allIps.append(ipaddress.ip_network(prefix))
            for network in allIps:
                countit = True
                # Do not count subnets
                for checkOverlap in allIps:
                    if network != checkOverlap and network.overlaps(checkOverlap) and\
                       network.prefixlen > checkOverlap.prefixlen:
                        countit = False
                        break
                if countit:
                    self.num_ipv4_addresses+=network.num_addresses




