#!/usr/bin/python
import sys

def process_conll_data(data_dir, out_dir, file_name):
    in_file = open(data_dir + file_name, 'r')
    conll = [[line.split('\t') for line in sent.split('\n')] 
                for sent in in_file.read().split('\n\n')[:-1]]
    in_file.close() 
    #print conll
    #word_sentences = [[line[1] for line in sent] for sent in conll]
    #dependencies = [[-1] + [int(line[6]) for line in sent] for sent in conll]
 
    #read in the universal pos-tag conversions
    pos_in = [line.split() for line in open('en-ptb.map', 'r').read().split('\n')[:-1]]
    pos_map = dict()
    for p in pos_in:
        pos_map[p[0]] = p[1]

    #blank out arc labels
    #insert universal pos tags
    for sent in conll:
        for line in sent:
            line[7] = '_'
            line[4] = pos_map[line[3]]

    out_file = open(out_dir + file_name, 'w')
    for sent in conll:
        for line in sent:
            out_file.write('\t'.join(line) + '\n')
        out_file.write('\n')


if __name__=='__main__':
    train_dir = 'depbank/'
    out_dir ='depbank_ul/'
    process_conll_data(train_dir, out_dir, sys.argv[1])
 
