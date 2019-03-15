from slimDesc import *
import stem.descriptor.reader
from stem.descriptor.reader import FileMissing
import stem.descriptor
import stem
import os
import os.path
import cPickle as pickle
import datetime, time, pytz
import wget
import argparse

"""

To the extent that a federal employee is an author of a portion of
this software or a derivative work thereof, no copyright is claimed by
the United States Government, as represented by the Secretary of the
Navy ("GOVERNMENT") under Title 17, U.S. Code. All Other Rights 
Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following disclaimer
      in the documentation and/or other materials provided with the
      distribution.
    * Neither the names of the copyright owners nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
GOVERNMENT ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS" CONDITION
AND DISCLAIMS ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
RESULTING FROM THE USE OF THIS SOFTWARE.


This is a slightly modified version of github.com/torps process_consensus file.
This version fixes a memory consumption issue and TorPS's issue #8 on
correctness of timestamps. It also add automatics downloads of missing
descriptors from Collector

All credits to its original author

"""

parser = argparse.ArgumentParser(description="Process consensus, descriptors and extra info descriptors to store a consensus-dependant subset of
        descriptor informations")
parser.add_argument('--start_year', type=int)
parser.add_argument('--start_month', type=int)
parser.add_argument('--end_year', type=int)
parser.add_argument('--end_month', type=int)
parser.add_argument('--in_dir_cons', help='directory where are located consensus')
parser.add_argument('--in_dir_desc', help='directory where are located descriptors')
parser.add_argument('--in_dir_extra_desc', help='directory where are located extra escriptors')
parser.add_argument('--out_dir', help='directory where we output our lightweight classes')
parser.add_argument('--initial_descriptor_dir', help='Needed if we look in the beginning of a month, due to the structure of \
        archives files, the descriptors might be in the previous month')
parser.add_argument('--initial_extrainfo_descriptor_dir', help='Need if we look in the beginning of a month, due the structure of \
        archives files, the descriptors might be in the previous month')


class TorOptions:
    """Stores some parameters set by Tor."""
    # given by #define ROUTER_MAX_AGE (60*60*48) in or.h    
    router_max_age = 60*60*48    
    default_bwweightscale = 10000   

def timestamp(t):
    """Returns UNIX timestamp"""
    td = t - datetime.datetime(1970, 1, 1, tzinfo=pytz.UTC)
    ts = td.days*24*60*60 + td.seconds
    return ts


# def read_extra_descriptors(extra_descriptors, extra_descriptor_dir):
    # num_descriptors = 0
    # print ('Reading extra descriptors from: {0}'.format(extra_descriptor_dir))


def read_descriptors(descriptors, end_of_month_desc, descriptor_dir, skip_listener):
	"""Add to descriptors contents of descriptor archive in descriptor_dir."""
        num_descriptors = 0    
        num_relays = 0
        print('Reading descriptors from: {0}'.format(descriptor_dir))
        reader = stem.descriptor.reader.DescriptorReader(descriptor_dir, \
        validate=False)
        reader.register_skip_listener(skip_listener)
        # use read li$stener to store metrics type annotation, whione_annotation = [None]
        cur_type_annotation = [None]
        def read_listener(path):
            f = open(path)
            # store initial metrics type annotation
            initial_position = f.tell()
            first_line = f.readline()
            f.seek(initial_position)
            if (first_line[0:5] == '@type'):
                cur_type_annotation[0] = first_line
            else:
                cur_type_annotation[0] = None
            f.close()
        reader.register_read_listener(read_listener)
        with reader:
            for desc in reader:
                if (num_descriptors % 10000 == 0):
                    print('{0} descriptors processed.'.format(num_descriptors))
                num_descriptors += 1
                if (desc.fingerprint not in descriptors):
                    descriptors[desc.fingerprint] = {}
                    num_relays += 1
                    # stuff type annotation into stem object
                desc.type_annotation = cur_type_annotation[0]
                if desc.published is not None:
                    t_published =  timestamp(desc.published.replace(tzinfo=pytz.UTC))
                    if (desc.published.day > 27): #february ends the 28th and I am lazy
                        if (desc.fingerprint not in end_of_month_desc):
                            end_of_month_desc[desc.fingerprint] = {}
                        end_of_month_desc[desc.fingerprint][t_published] = desc
                    descriptors[desc.fingerprint]\
                            [t_published] = desc
        print('#descriptors: {0}; #relays:{1}'.\
            format(num_descriptors,num_relays)) 

#TODO use system call to untar after download. Python2.7 does not handle .xz compression mode
def consensus_dir_exist_or_download(in_consensus_dir):
    #pdb.set_trace()
    if not os.path.isdir(in_consensus_dir) and not os.path.exists("{0}.tar.xz".format(in_consensus_dir)):
        print "Consensus dir missing. Downloading tar file"
        wget.download("https://collector.torproject.org/archive/relay-descriptors/consensuses/{0}.tar.xz".
                format(in_consensus_dir.split('/')[-1]), "{0}.tar.xz".format(in_consensus_dir))

def process_consensuses(in_dirs, initial_descriptor_dir, initial_extra_descriptor_dir):
    """For every input consensus, finds the descriptors published most recently before the descriptor times listed for the relays in that consensus, records state changes indicated by descriptors published during the consensus fresh period, and writes out pickled consensus and descriptor objects with the relevant information.
        Inputs:
            in_dirs: list of (consensus in dir, descriptor in dir, extra descriptor in dir,
                processed descriptor out dir) triples *in order*
    """
    descriptors, end_of_month_desc = {}, {}
    extra_descriptors, end_of_month_extra_desc = {}, {}

    def skip_listener(path, exception):

        print('ERROR [{0}]: {1}'.format(path, exception))
        #TODO makes several attemps and use mirror if download fails ? 
        if isinstance(exception, FileMissing) and not os.path.exists("{0}.tar.xz".format(path)):
            path_elems = path.split('/')
            filename = path_elems[-1]
            if "extra" in filename:
                url = "https://collector.torproject.org/archive/relay-descriptors/extra-infos/{0}.tar.xz".format(filename)
            elif "server" in filename:
                url = "https://collector.torproject.org/archive/relay-descriptors/server-descriptors/{0}.tar.xz".format(filename)
            elif "consensuses" in filename:
                url = "https://collector.torproject.org/archive/relay-descriptors/consensuses/{0}.tar.xz".format(filename)
            else:
                raise ValueError("filename is not about descriptors")
            print "Downloading descriptors to {0}.tar.xz".format(path)
            wget.download(url, "{0}.tar.xz".format(path))
        
    # initialize descriptors
    if (initial_descriptor_dir is not None and initial_extra_descriptor_dir is not None):
        read_descriptors(descriptors, end_of_month_extra_desc, initial_descriptor_dir, skip_listener)
        read_descriptors(extra_descriptors, end_of_month_extra_desc, initial_extra_descriptor_dir, skip_listener)
    firstLoop = True
    for in_consensuses_dir, in_descriptors, in_extra_descriptors, desc_out_dir in in_dirs:
		# read all descriptors into memory        a
        if not firstLoop:
            descriptors = {} #free memory
            descriptors.update(end_of_month_desc)
            end_of_month_desc  = {}  #free memory
            extra_descriptors = {}
            extra_descriptors.update(end_of_month_extra_desc)
            end_of_month_extra_desc = {}
            firstLoop=False
        read_descriptors(descriptors, end_of_month_desc,  in_descriptors, skip_listener)
        # read all extra descriptors
        read_descriptors(extra_descriptors, end_of_month_extra_desc, in_extra_descriptors, skip_listener)
        # output pickled consensuses, dict of most recent descriptors, and 
        # list of hibernation status changes

        consensus_dir_exist_or_download(in_consensuses_dir)

        num_consensuses = 0
        pathnames = []
        for dirpath, dirnames, fnames in os.walk(in_consensuses_dir):
            for fname in fnames:
                pathnames.append(os.path.join(dirpath,fname))
        pathnames.sort()
        for pathname in pathnames:
            filename = os.path.basename(pathname)
            if (filename[0] == '.'):
                continue
            
            print('Processing consensus file {0}'.format(filename))
            cons_f = open(pathname, 'rb')

            # store metrics type annotation line
            initial_position = cons_f.tell()
            first_line = cons_f.readline()
            if (first_line[0:5] == '@type'):
                type_annotation = first_line
            else:
                type_annotation = None
            cons_f.seek(initial_position)

            descriptors_out = dict()
            extra_descriptors_out = dict()
            hibernating_statuses = [] # (time, fprint, hibernating)
            cons_valid_after = None
            cons_fresh_until = None
            cons_bw_weights = None
            cons_bwweightscale = None
            relays = {}
            num_not_found = 0
            num_found = 0
            # read in consensus document
            i = 0
            for document in stem.descriptor.parse_file(cons_f, validate=False,
                document_handler='DOCUMENT'):
                if (i > 0):
                    raise ValueError('Unexpectedly found more than one consensus in file: {}'.\
                        format(pathname))
                if (cons_valid_after == None):
                    cons_valid_after = document.valid_after.replace(tzinfo=pytz.UTC)
                    # compute timestamp version once here
                    valid_after_ts = timestamp(cons_valid_after)
                if (cons_fresh_until == None):
                    cons_fresh_until = document.fresh_until.replace(tzinfo=pytz.UTC)
                    # compute timestamp version once here
                    fresh_until_ts = timestamp(cons_fresh_until)
                if (cons_bw_weights == None):
                    cons_bw_weights = document.bandwidth_weights
                if (cons_bwweightscale == None) and \
                    ('bwweightscale' in document.params):
                    cons_bwweightscale = document.params[\
                            'bwweightscale']
                for fprint, r_stat in document.routers.iteritems():
                    relays[fprint] = RouterStatusEntry(fprint, r_stat.nickname, \
                        r_stat.flags, r_stat.bandwidth)
                consensus = document
                i += 1
                            

            # find relays' most recent unexpired descriptor published
            # before the publication time in the consensus
            # and status changes in fresh period (i.e. hibernation)
            for fprint, r_stat in consensus.routers.iteritems():
                pub_time = timestamp(r_stat.published.replace(tzinfo=pytz.UTC))
                desc_time = 0
                descs_while_fresh = []
                desc_time_fresh = None
                # get all descriptors with this fingerprint
                if (r_stat.fingerprint in descriptors):
                    for t,d in descriptors[r_stat.fingerprint].items():
                        # update most recent desc seen before cons pubtime
                        # allow pubtime after valid_after but not fresh_until
                        if (valid_after_ts-t <\
                            TorOptions.router_max_age) and\
                            (t <= pub_time) and (t > desc_time) and\
                            (t <= fresh_until_ts):
                            desc_time = t
                        # store fresh-period descs for hibernation tracking
                        if (t >= valid_after_ts) and \
                            (t <= fresh_until_ts):
                            descs_while_fresh.append((t,d))                                
                        # find most recent hibernating stat before fresh period
                        # prefer most-recent descriptor before fresh period
                        # but use oldest after valid_after if necessary
                        if (desc_time_fresh == None):
                            desc_time_fresh = t
                        elif (desc_time_fresh < valid_after_ts):
                            if (t > desc_time_fresh) and\
                                (t <= valid_after_ts):
                                desc_time_fresh = t
                        else:
                            if (t < desc_time_fresh):
                                desc_time_fresh = t

                # output best descriptor if found
                if (desc_time != 0):
                    num_found += 1
                    # store discovered recent descriptor
                    desc = descriptors[r_stat.fingerprint][desc_time]
                    extra_found = False
                    try :
                        extra_descs = extra_descriptors[r_stat.fingerprint]
                        for key, extra_desc_elem in extra_descs.iteritems():
                            if desc.extra_info_digest == extra_desc_elem.digest():
                                extra_desc = extra_desc_elem
                                extra_found = True
                                break
                        if extra_found:
                            write_history = []
                            #convert to timestamp
                            if extra_desc.write_history_end is not None:
                                end_interval = (int) (time.mktime(extra_desc.write_history_end.utctimetuple()))
                                i = len(extra_desc.write_history_values)
                                for value in extra_desc.write_history_values:
                                    write_history.append((end_interval-i*extra_desc.write_history_interval, value))
                                    i -= 1
                            else :
                                print("Relay {0}.{1} has an extra_desc uncomplete: write_history_end, write_history_values: {3}".format(
                                    r_stat.fingerprint, desc.nickname,extra_desc.write_history_end, extra_desc.write_history_values))
                        else:
                            print("Relay {0}.{1} does not have extra_descriptors"\
                                .format(r_stat.fingerprint, desc.nickname))
                    except KeyError: 
                        print("Relay {0}.{1} does not have extra_descriptors"\
                                .format(r_stat.fingerprint, desc.nickname))
                    if extra_found :
                        descriptors_out[r_stat.fingerprint] = \
                            S, r_stat.nicknameerverDescriptor(desc.fingerprint, \
                                desc.hibernating, desc.nickname, \
                                desc.family, desc.address, \
                                desc.average_bandwidth, desc.burst_bandwidth, \
                                desc.observed_bandwidth, \
                                desc.extra_info_digest, \
                                desc.dir_v2_requests,\
                                desc.dir_v3_requests)
                    else:
                        print "No extra-info descriptor found for {0}-{1}".format(r_stat.fingerprint, r_stat.nickname)
                        descriptors_out[r_stat.fingerprint] = \
                            ServerDescriptor(desc.fingerprint, \
                                desc.hibernating, desc.nickname, \
                                desc.family, desc.address, \
                                desc.average_bandwidth, desc.burst_bandwidth, \
                                desc.observed_bandwidth, \
                                desc.extra_info_digest, \
                                None, None)
                     
                    # store hibernating statuses
                    if (desc_time_fresh == None):
                        raise ValueError('Descriptor error for {0}:{1}.\n Found  descriptor before published date {2}: {3}\nDid not find descriptor for initial hibernation status for fresh period starting {4}.'.format(r_stat.nickname, r_stat.fingerprint, pub_time, desc_time, valid_after_ts))
                    desc = descriptors[r_stat.fingerprint][desc_time_fresh]
                    cur_hibernating = desc.hibernating
                    # setting initial status
                    hibernating_statuses.append((0, desc.fingerprint,\
                        cur_hibernating))
                    if (cur_hibernating):
                        print('{0}:{1} was hibernating at consenses period start'.format(desc.nickname, desc.fingerprint))
                    descs_while_fresh.sort(key = lambda x: x[0])
                    for (t,d) in descs_while_fresh:
                        if (d.hibernating != cur_hibernating):
                            cur_hibernating = d.hibernating                                   
                            hibernating_statuses.append(\
                                (t, d.fingerprint, cur_hibernating))
                            if (cur_hibernating):
                                print('{0}:{1} started hibernating at {2}'\
                                    .format(d.nickname, d.fingerprint, t))
                            else:
                                print('{0}:{1} stopped hibernating at {2}'\
                                    .format(d.nickname, d.fingerprint, t))                   
                else:
#                            print(\
#                            'Descriptor not found for {0}:{1}:{2}'.format(\
#                                r_stat.nickname,r_stat.fingerprint, pub_time))
                    num_not_found += 1
                    
            # output pickled consensus, recent descriptors, and
            # hibernating status changes
            if (cons_valid_after != None) and\
                (cons_fresh_until != None):
                consensus_out = NetworkStatusDocument(\
                    cons_valid_after, cons_fresh_until, cons_bw_weights,\
                    cons_bwweightscale, relays)
                hibernating_statuses.sort(key = lambda x: x[0],\
                    reverse=True)
                outpath = os.path.join(desc_out_dir,\
                    cons_valid_after.strftime(\
                        '%Y-%m-%d-%H-%M-%S-network_state'))
                f = open(outpath, 'wb')
                pickle.dump(consensus_out, f, pickle.HIGHEST_PROTOCOL)
                pickle.dump(descriptors_out,f,pickle.HIGHEST_PROTOCOL)
                pickle.dump(hibernating_statuses,f,pickle.HIGHEST_PROTOCOL)
                f.close()

                print('Wrote descriptors for {0} relays.'.\
                    format(num_found))
                print('Did not find descriptors for {0} relays\n'.\
                    format(num_not_found))
            else:
                print('Problem parsing {0}.'.format(filename))             
            num_consensuses += 1
            
            cons_f.close()
                
        print('# consensuses: {0}'.format(num_consensuses))

if __name__ == "__main__":

    args = parser.parse_args()
    in_dirs = []
    month = args.start_month
    for year in range(args.start_year, args.end_year+1):
      while ((year < args.end_year) and (month <= 12)) or \
             (month <= args.end_month):
        if (month <= 9):
          prepend = '0'
        else:
          prepend = ''
        cons_dir = os.path.join(args.in_dir_cons, 'consensuses-{0}-{1}{2}'.\
                format(year, prepend, month))
        desc_dir = os.path.join(args.in_dir_desc,
                'server-descriptors-{0}-{1}{2}'.format(year, prepend, month))
        extra_desc_dir = os.path.join(args.in_dir_extra_desc,
                'extra-infos-{0}-{1}{2}'.format(year, prepend, month))
        desc_out_dir = os.path.join(args.out_dir,
                'network-state-{0}-{1}{2}'.format(year, prepend, month))
        if (not os.path.exists(desc_out_dir)):
          os.mkdir(desc_out_dir)
        in_dirs.append((cons_dir, desc_dir, extra_desc_dir, desc_out_dir))
        month += 1
      month = 1
    process_consensuses(in_dirs, args.initial_descriptor_dir, args.initial_extrainfo_descriptor_dir)
