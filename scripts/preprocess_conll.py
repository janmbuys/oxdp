#!/usr/bin/python
import sys
import math

def process_conll_data(data_dir, out_dir, train_file_name, file_name):
    #some options
    no_words = False
    unk_cutoff = 2
    lower_case = True
    part_lex = False
    lex_threshold = 100
    include_coarse_tags = True
    only_coarse_tags = False
    unlabelled = True
    write_sentences = True
    write_dependencies = False
    print_stats = True

    in_file = open(data_dir + file_name, 'r')
    conll = [[line.split('\t') for line in sent.split('\n')] 
                for sent in in_file.read().split('\n\n')[:-1]]
    in_file.close() 
    
    in_file = open(data_dir + train_file_name, 'r')
    conll_train = [[line.split('\t') for line in sent.split('\n')] 
                for sent in in_file.read().split('\n\n')[:-1]]
    in_file.close() 

    #read in the universal pos-tag conversions
    pos_in = [line.split() for line in open('en-ptb.map', 'r').read().split('\n')[:-1]]
    pos_map = dict()
    for p in pos_in:
        pos_map[p[0]] = p[1]

    #counts the words
    word_count = dict()
    for sent in conll_train:
        for line in sent:
            if lower_case:
                line[1]= line[1].lower()
	    if word_count.has_key(line[1]):
                word_count[line[1]] += 1
	    else:
                word_count[line[1]] = 1
	
    num_tokens = sum([len(sent) for sent in conll])
    vocab = filter(lambda x: word_count[x] >= unk_cutoff, word_count.keys())

    #perform modifications
    for sent in conll:
        for line in sent:
            if lower_case:
                line[1] = line[1].lower()
	    if (part_lex and word_count.has_key(line[1]) and (word_count[line[1]] >= lex_threshold)):
                line[3] = line[1] # replace frequent words' pos tag
            if no_words:
	        line[1] = '_' #blank out the words
            elif ((not word_count.has_key(line[1])) or (word_count[line[1]] < unk_cutoff)):
                line[1] = '<unk>'
            if unlabelled:
                line[7] = 'ROOT'
            #move fine-grained pos to standard column 
            line[4] = line[3]
            if include_coarse_tags:
                line[3] = pos_map[line[4]] 
            if only_coarse_tags:
                line[4] = line[3]   #course postags for now
            #line[6] is head dependency
	
    if write_sentences:
        out_file = open(out_dir + file_name + '.txt', 'w')
        for sent in conll: 
            for line in sent:
                out_file.write(line[1] + ' ')
            out_file.write('\n')
        out_file.close()
    if write_dependencies:
        out_file = open(out_dir + file_name + '.dep', 'w')
        for sent in conll: 
            for line in sent:
                out_file.write(line[6] + ' ')
            out_file.write('\n')
        out_file.close()
 
    #write out the modified file
    out_file = open(out_dir + file_name, 'w')
    for i in range(len(conll)): 
        for line in conll[i]:
            out_file.write('\t'.join(line) + '\n')
        out_file.write('\n')
    out_file.close()

    if print_stats:
        print 'number of sentences', len(conll)
        print 'number of tokens', num_tokens
        print 'size of raw vocabulary', len(word_count)
        print 'size of vocabulary', len(vocab)
        print 'square root of size of vocabulary', int(math.sqrt(len(vocab)))

if __name__=='__main__':
    if (len(sys.argv)<>5):
        print "incorrect number of arguments (4 expected)\n"
    else: 
        train_dir = sys.argv[1] + '/' #'depbank_conll07/'
        train_file = sys.argv[2] 
        process_file = sys.argv[3] 
        out_dir = sys.argv[4] + '/'  #'depbank_conll07_nowords_coarse/'
        process_conll_data(train_dir, out_dir, train_file, process_file)
 
