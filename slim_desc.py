"""

Used to extract required information from Tor relay descriptors to keep memory usage somewhat low.

"""
class RouterStatusEntry:

    def __init__(self, fprint, nickname, flags, consweight):
        self.fprint = fprint
        self.nickname = nickname
        self.flags = flags
        self.consweight = consweight

##TODO rewrite ClusterRouter to something more flexible
##
class ClusterRouter:

    def  __init__(self, asn=None, prefix=None):
        self.asn = asn
        self.prefix = prefix
        self.tot_consweight = 0
        self.routerlist = {}

    def addRouter(self, router):
        assert(isinstance(router, RouterStatusEntry))
        if router.fprint not in self.routerlist:
            self.routerlist[router.fprint] = router
        self.tot_consweight += router.consweight



class ServerDescriptor:

    def __init__(self, fprint, hibernating, nickname, family, address,
            avg_bw, burst_bw, observ_bw, extra_info_digest,
            dirreqv2_from_country, dirreqv3_from_country,
            dirreqv2_unique_ips, dirreqv3_unique_ips):
        self.fprint = fprint
        self.hibernating = hibernating
        self.nickname = nickname
        self.address = address
        self.avg_bw = avg_bw
        self.burst_bw = burst_bw
        self.observ_bw = observ_bw
        self.extra_info_digest = extra_info_digest
        #mapping of locales to rounded count of requests
        #dirreq-v3-ips counts each country once for each unique IP in that country,
        #no matter how many requests each IP makes. dirreq-v3-reqs counts the
        #number of requests from each country.
        self.dirreqv2_from_country = dirreqv2_from_country 
        self.dirreqv3_from_country = dirreqv3_from_country 
        self.dirreqv2_unique_ips = dirreqv2_unique_ips
        self.dirreqv3_unique_ips = dirreqv3_unique_ips


class NetworkStatusDocument:

    def __init__(self, cons_valid_after, cons_fresh_until, cons_bw_weights,
            cons_bwweightscale, relays):
        self.cons_bwweightscale = cons_bwweightscale
        self.cons_valid_after = cons_valid_after
        self.cons_fresh_until = cons_fresh_until
        self.cons_bw_weights = cons_bw_weights
        self.relays = relays

class NetworkState:
    def __init__(self, cons_valid_after, cons_fresh_until, cons_bw_weights,
            cons_bwweightscale, hibernating_statuses, cons_rel_stats, descriptors):
        self.cons_valid_after = cons_valid_after
        self.cons_fresh_until = cons_fresh_until
        self.cons_bw_weights = cons_bw_weights
        self.cons_bwweightscale = cons_bwweightscale
        self.hibernating_statuses = hibernating_statuses 
        self.cons_rel_stats = cons_rel_stats
        self.descriptors = descriptors
