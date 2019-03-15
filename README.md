# optimal-locaware-loadbalancing

Set of scripts aimed at deriving scores to select relays in a
location-aware fashion while still preserving the load-balancing of the
Tor network.

## Goals

  - Easily deployable / implementable system; Does not change the
    fundamental logic of Tor's weight selection, but how only *how*
    those weights are computed to offer location-aware protection

  - The location-aware scheme minimizes the probability to end-up with a
    same AS in both Client-Guard IP-level path and Exit-Destination
    IP-level path, for all Tor clients.

  - \theta-GP-secure

  - The network is load-balanced with the same assumptions as Tor today:
    iff the external systems providing information are rights.

## Procedure

  1)Offers an heuristic to evaluate client distribution per AS without
    relying on new measurements (we only use the data which Tor already
    provides - i.e., an estimation of Tor users per country). This heuristic
    is a fundamental block of the load-balancing system, like Torflow
    measurements are.
    To map towards a distribution of clients per AS, we may either:
      - Compute the joint distribution of Tor clients per country with
        the number of available IP addresses per custumer ASes (using
        BGP databases).
      - Compute the joint distribution of Tor clients per country with
        the distribution of Ping-responsive IP addresses per custumer
        ASes (using RIPE atlas probes).

    Estimated work: more than 1 week full time.

  2) Apply the following minmax optimization problem to derive weights for each
     location  

   Let W_l the density of Clients per AS computed from 1), such that
   \sum W_l = 1
   Let R_l a discrete distribution of scores to compute for guard selection
   common to all clients in AS l.
   Let G the total consensus weight of the guard-flagged relays
   Let Wgg the fraction of bandwidth of each guard-flagged relay
   dedicated to the entry position, computed as described in
   dir-spec.txt Section 3.8.4.
   Let L = \sum W_l\*R_l
   Let VULN a discrete distribution of #vulnerable paths for all guards
   computed from the intersection set of ASes in the forward and reverse
   paths between Client_AS <-> Guard and Exit <-> Destination

   We want to find an allocation of weights for each R_l such that:

      min max(L*VULN)
   
   under constraints:

      1) for l in AllLocations:
         \sum R_l(i) = G*Wgg
      
      2) L(i) <= BW_i

      3) max_l max_i R_l(i)/relCost(i) <= \theta
  
  Assuming a solver works on this (should be); then we may know compute
the set of weights for the middle position:

  Wmg_i = 1 - (L(i)/BW_i)

  Each Tor clients selects guard with Pr(G=i) = R(i)/\sum(R(j)) and
middle relay Pr(M=i) = Wmg_i\*BW_i/\sum(Wmg_j\*BW_j)

  This procedure guarantees a load factor of 1 on each relay, with a
minimization of a network adversary's incentives to control a particular
AS, for all Tor users.
