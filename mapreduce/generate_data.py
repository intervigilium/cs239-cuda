#!/usr/bin/env python
import sys
import getopt
import random

def generate(data_len):
    random.seed()
    for n in range(int(data_len)):
        print random.randint(0, 1000000000);

def main(argv=None):
    if argv is None:
        argv = sys.argv
    optlist, args = getopt.getopt(argv, '')
    if len(args) != 2:
        print "usage: generate_data.py <number of random numbers>"
    else:
        print(args[1])
        generate(args[1])

if __name__ == "__main__":
    sys.exit(main())
