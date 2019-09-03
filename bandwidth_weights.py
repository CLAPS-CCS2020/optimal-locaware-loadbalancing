
class Enum(tuple): __getattr__ = tuple.index

class BandwidthWeights(object):

    def __init__(self):
        self.bww_errors = Enum(("NO_ERROR","SUMG_ERROR", "SUME_ERROR",
            "SUMD_ERROR","BALANCE_MID_ERROR", "BALANCE_EG_ERROR",
            "RANGE_ERROR"))


    def check_weights_errors(self, Wgg, Wgd, Wmg, Wme, Wmd, Wee, Wed,
            weightscale, G, M, E, D, T, margin, do_balance):
        """Verify that our weights satify the formulas from dir-spec.txt"""

        def check_eq(a, b, margin):
            return (a - b) <= margin if (a - b) >= 0 else (b - a) <= margin
        def check_range(a, b, c, d, e, f, g, mx):
            return (a >= 0 and a <= mx and b >= 0 and b <= mx and\
                    c >= 0 and c <= mx and d >= 0 and d <= mx and\
                    e >= 0 and e <= mx and f >= 0 and f <= mx and\
                    g >= 0 and g <= mx)

            # Wed + Wmd + Wgd == weightscale
        if (not check_eq(Wed+Wmd+Wgd, weightscale, margin)):
            return self.bww_errors.SUMD_ERROR
        # Wmg + Wgg == weightscale
        if (not check_eq(Wmg+Wgg, weightscale, margin)):
            return self.bww_errors.SUMG_ERROR
        # Wme + Wee == 1
        if (not check_eq(Wme+Wee, weightscale, margin)):
            return self.bww_errors.SUME_ERROR
        # Verify weights within range 0 -> weightscale
        if (not check_range(Wgg, Wgd, Wmg, Wme, Wmd, Wed, Wee, weightscale)):
            return self.bww_errors.RANGE_ERROR
        if (do_balance):
            #Wgg*G + Wgd*D == Wee*E + Wed*D
            if (not check_eq(Wgg*G+Wgd*D, Wee*E+Wed*D, (margin*T)/3)):
                return self.bww_errors.BALANCE_EG_ERROR
            #Wgg*G+Wgd*D == M*weightscale + Wmd*D + Wme * E + Wmg*G
            if (not check_eq(Wgg*G+Wgd*D, M*weightscale+Wmd*D+Wme*E+Wmg*G,
                (margin*T)/3)):
                return self.bww_errors.BALANCE_MID_ERROR

        return self.bww_errors.NO_ERROR


    def recompute_bwweights(self, G, M, E, D, T, SWgg=False):
        """Detects in which network case load we are according to section 3.8.3
        of dir-spec.txt from Tor' specifications and recompute bandwidth weights
        """
        weightscale = 10000
        if (3*E >= T and 3*G >= T):
            #Case 1: Neither are scarce
            casename = "Case 1 (Wgd=Wmd=Wed)"
            Wgd = Wed = Wmd = weightscale/3
            Wee = (weightscale*(E+G+M))/(3*E)
            Wme = weightscale - Wee
            Wmg = (weightscale*(2*G-E-M))/(3*G)
            Wgg = weightscale - Wmg

            check = self.check_weights_errors(Wgg, Wgd, Wmg, Wme, Wmd, Wee, Wed,
                    weightscale, G, M, E, D, T, 10, True)
            if (check != self.bww_errors.NO_ERROR):
                raise ValueError(\
                        'ERROR: {0}  Wgd={1}, Wed={2}, Wmd={3}, Wee={4},\
                        Wme={5}, Wmg={6}, Wgg={7}'.format(self.bww_errors[check],
                            Wgd, Wed, Wmd, Wee, Wme, Wmg, Wgg))
        elif (3*E < T and 3*G < T):
            #Case 2: Both Guards and Exits are scarce
            #Balance D between E and G, depending upon D capacity and
            #scarcity
            R = min(E, G)
            S = max(E, G)
            if (R+D < S):
                #subcase a
                Wgg = Wee = weightscale
                Wmg = Wme = Wmd = 0
                if (E < G):
                    casename = "Case 2a (E scarce)"
                    Wed = weightscale
                    Wgd = 0
                else: 
                    # E >= G
                    casename = "Case 2a (G scarce)"
                    Wed = 0
                    Wgd = weightscale

            else:
                #subcase b R+D >= S
                casename = "Case 2b1 (Wgg=weightscale, Wmd=Wgd)"
                Wee = (weightscale*(E-G+M))/E
                Wed = (weightscale*(D-2*E+4*G-2*M))/(3*D)
                Wme = (weightscale*(G-M))/E
                Wmg = 0
                Wgg = weightscale
                Wmd = Wgd = (weightscale-Wed)/2

                check = self.check_weights_errors(Wgg, Wgd, Wmg, Wme, Wmd,
                        Wee, Wed, weightscale, G, M, E, D, T, 10, True)
                if (check != self.bww_errors.NO_ERROR):
                    casename = 'Case 2b2 (Wgg=weightscale, Wee=weightscale)'
                    Wgg = Wee = weightscale
                    Wed = (weightscale*(D-2*E+G+M))/(3*D)
                    Wmd = (weightscale*(D-2*M+G+E))/(3*D)
                    Wme = Wmg = 0
                    if (Wmd < 0):
                        #Too much bandwidth at middle position
                        casename = 'case 2b3 (Wmd=0)'
                        Wmd = 0
                    Wgd = weightscale - Wed - Wmd

                    check = self.check_weights_errors(Wgg, Wgd, Wmg, Wme, Wmd,
                            Wee, Wed, weightscale, G, M, E, D, T, 10, True)
                if (check != self.bww_errors.NO_ERROR and check !=\
                            self.bww_errors.BALANCE_MID_ERROR):
                    raise ValueError(\
                            'ERROR: {0}  Wgd={1}, Wed={2}, Wmd={3}, Wee={4},\
                            Wme={5}, Wmg={6}, Wgg={7}'.format(self.bww_errors[check],
                                Wgd, Wed, Wmd, Wee, Wme, Wmg, Wgg))
        else: # if (E < T/3 or G < T/3)
            #Case 3: Guard or Exit is scarce
            S = min(E, G)

            if (not (3*E < T or  3*G < T) or not (3*G >= T or 3*E >= T)):
                raise ValueError(\
                        'ERROR: Bandwidths have inconsistent values \
                        G={0}, M={1}, E={2}, D={3}, T={4}'.format(G,M,E,D,T))

            if (3*(S+D) < T):
                    #subcasea: S+D < T/3
                if (G < E):
                    casename = 'Case 3a (G scarce)'
                    Wgg = Wgd = weightscale
                    Wmd = Wed = Wmg = 0

                    if (E < M): Wme = 0
                    else: Wme = (weightscale*(E-M))/(2*E)
                    Wee = weightscale - Wme
                else:
                    # G >= E
                    casename = "Case 3a (E scarce)"
                    Wee = Wed = weightscale
                    Wmd = Wgd = Wme = 0
                    if (G < M): 
                        Wmg = 0
                    else:
                        if SWgg:
                            # We only care about case 3a in this project 'cause that's the one
                            # we currently always have
                            print("Computing Wgg and Wmg under the condition that the total guard bandwidth is equal t the total exit bandwidth")
                            Wgg = weightscale*(E+D)/G
                            Wmg = weightscale - Wgg
                        else:
                            Wmg = (weightscale*(G-M))/(2*G)
                            Wgg = weightscale - Wmg
            else:
                #subcase S+D >= T/3
                if (G < E):
                    casename = 'Case 3bg (G scarce, Wgg=weightscale, Wmd == Wed'
                    Wgg = weightscale
                    Wgd = (weightscale*(D-2*G+E+M))/(3*D)
                    Wmg = 0
                    Wee = (weightscale*(E+M))/(2*E)
                    Wme = weightscale - Wee
                    Wmd = Wed = (weightscale-Wgd)/2

                    check = self.check_weights_errors(Wgg, Wgd, Wmg, Wme,
                            Wmd, Wee, Wed, weightscale, G, M, E, D, T, 10,
                            True)
                else:
                    # G >= E
                    casename = 'Case 3be (E scarce, Wee=weightscale, Wmd == Wgd'
                    Wee = weightscale
                    Wed = (weightscale*(D-2*E+G+M))/(3*D)
                    Wme = 0
                    Wgg = (weightscale*(G+M))/(2*G)
                    Wmg = weightscale - Wgg
                    Wmd = Wgd = (weightscale-Wed)/2

                    check = self.check_weights_errors(Wgg, Wgd, Wmg, Wme,
                            Wmd, Wee, Wed,  weightscale, G, M, E, D, T, 10,
                            True)

                if (check):
                    raise ValueError(\
                            'ERROR: {0}  Wgd={1}, Wed={2}, Wmd={3}, Wee={4},\
                            Wme={5}, Wmg={6}, Wgg={7}'.format(self.bww_errors[check],
                                Wgd, Wed, Wmd, Wee, Wme, Wmg, Wgg))

        return (casename, Wgg, Wgd, Wee, Wed, Wmg, Wme, Wmd)

if __name__ == "__main__":

    test = BandwidthWeights()
    print("", test.recompute_bwweights(20004302,32832,2032032,232432323,122323224523 ))
