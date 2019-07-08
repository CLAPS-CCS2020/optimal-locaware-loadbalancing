import sys, os

def parse_client_cluster(filename):
    representative = {}
    with open(filename) as f:
        for line in f:
            tab = line.split('\t')
            representative[tab[0]] = tab[1][:-1].split(',')
    return representative
