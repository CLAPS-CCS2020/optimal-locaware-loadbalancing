import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
parser = argparse.ArgumentParser()
parser.add_argument('--tor_user_to_country_file', type=str, required=True)
parser.add_argument('--selfmade_user_to_country_file', type=str, required=True)


#From https://github.com/RWails/tor-location-masking/blob/master/python/tor_stats.py
def parse_userstats(filename): 
    country_to_users = dict()
    with open(filename, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True, delimiter=',')
        #skip header
        for _ in range(0, 6):
            next(reader)
        for row in reader:
            if row[1] not in country_to_users:
                country_to_users[row[1]] = int(row[2])
            else:
                country_to_users[row[1]] += int(row[2])
    #ignore tot, ??, a1 and a2
    if '' in country_to_users:
        del country_to_users['']
    if '??' in country_to_users:
        del country_to_users['??']
    if 'a1' in country_to_users:
        del country_to_users['a1']
    if 'a2' in country_to_users:
        del country_to_users['a2']

    return country_to_users

if __name__ == "__main__":

    args = parser.parse_args()
    tor_country_to_users = parse_userstats(args.tor_user_to_country_file)
    with open(args.selfmade_user_to_country_file, "rb") as f:
        selfmade_country_to_users = pickle.load(f)
    #Normalize
    tot = sum(tor_country_to_users.values())
    tor_country_to_users_norm = [(k, v/tot) for k, v in tor_country_to_users.items()]
    tor_country_to_users_norm.sort(key=lambda x: x[1], reverse=True)
    #Plot the first 20ies

    #assert(len(tor_country_to_users_norm) == len(selfmade_country_to_users_norm))
    #plot bargraphs
    n_groups = 20
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    fig, ax = plt.subplots()
    ax.bar(index, [v for (k, v) in tor_country_to_users_norm[0:n_groups]], bar_width,
            alpha=opacity, color='r', label="Tor's userstats using #requests")
    ax.bar(index+bar_width, [selfmade_country_to_users[k] for (k, v) in tor_country_to_users_norm[0:n_groups]],
            bar_width, alpha=opacity, color='b', label="selfmade userstats using unique ips seen")
    ax.set_xlabel("Countries")
    ax.set_ylabel("%")
    ax.set_xticks(index+bar_width)
    ax.set_xticklabels((k for (k, v) in tor_country_to_users_norm[0:n_groups]))
    ax.legend()
    fig.tight_layout()
    plt.show()
