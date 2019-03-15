"""

Used to extract required information from Tor relay descriptors to keep memory usage somewhat low.

"""
class RouterStatusEntry:

    def __init__(self, fprint, nickname, flags, consweight):
        self.fprint = fprint
        self.nickname = nickname
        self.flags = flags

class ServerDescriptor:

    def __init__(self, fprint, hibernating, nickname, family, address,
            avg_bw, burst_bw, observ_bw, extra_info_digest):
        self.fprint = fprint
        self.hibernating = hibernating
        self.nickname = nickname
        self.address = address
        self.avg_bw = avg_bw
        self.burst_bw = burst_bw
        self.observ_bw = observ_bw
        self.extra_info_digest = extra_info_digest
        #mapping of locales to rounded count of requests
        self.dirreqv2_from_country = dirreq_from_country 


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
            hibernating_statuses, cons_rel_stats, descriptors):
        self.cons_valid_after = cons_valid_after
        self.cons_fresh_until = cons_fresh_until
        self.cons_bw_weights = cons_bw_weights
        self.hibernating_statuses = hibernating_statuses 
        self.cons_rel_stats = cons_rel_stats
        self.descriptors = descriptors
