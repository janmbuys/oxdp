class ParserConfiguration:
    def __init__(self):
        self.stack = []
        self.buffer_ = []
        self.arcs = []  #parent of each node
        self.child_count = [] #number of children of each node
        self.action_seq = []
        self.action_names = []
        self.shift_contexts = []
        self.reduce_contexts = []

    def __init__(self, sentence):
        self.stack = []
        self.sentence = sentence
        self.buffer_ = range(len(sentence)-1, -1, -1) 
        self.arcs = [-1]*len(sentence)
        self.child_count = [0]*len(sentence)
        self.action_seq = []
        self.action_names = []

    def get_action_str(self):
        return map(lambda x:self.action_names[x], self.action_seq)

    def get_shift_contexts(self, vocabulary):
        return [[vocabulary[self.sentence[j]], [vocabulary[self.sentence[i]] for i in ctx]] for j, ctx in enumerate(self.shift_contexts)][1:]

    def get_reduce_contexts(self, vocabulary):
        return [[self.action_names[self.action_seq[j]], [vocabulary[self.sentence[i]] for i in ctx]] for j, ctx in enumerate(self.reduce_contexts)][1:]

    def shift(self):
        self.stack.append(self.buffer_.pop())
        self.action_seq.append(0)

    def reduce_(self):
        #if self.arc[self.stack[-1]]>=0:
        self.stack.pop()
        self.action_seq.append(3)
       
    def get_child_count(self, i):
        return self.child_count[i]

    def has_parent(self, i):
        return self.arcs[i]>=0

    def get_stack_depth(self):
        return len(self.stack)

    def get_stack_top(self):
        return self.stack[-1]

    def get_stack_top_pair(self):
        return self.stack[-2], self.stack[-1]

    def get_stack_second_order_pair(self):
        return self.stack[-3], self.stack[-1]

    def get_stack(self):
        return self.stack

    def get_buffer_next(self):
        return self.buffer_[-1]

    def get_arcs(self):
        return self.arcs

    def is_buffer_empty(self):
        return len(self.buffer_)==0

    def is_terminal_configuration(self):
        return len(self.buffer_)==0 and len(self.stack)==1

    def get_actions(self):
        return self.action_seq

class ArcStandardParser(ParserConfiguration):
    def __init__(self, sentence):
        ParserConfiguration.__init__(self, sentence)
        self.action_names = ['sh', 'la', 'ra']

    #next element already on stack
    def left_arc(self):
        j = self.stack.pop()
        i = self.stack.pop()
        self.stack.append(j)
        self.arcs[i] = j
        self.child_count[j] += 1
        self.action_seq.append(1)
       
    #next element already on stack
    def right_arc(self):
        j = self.stack.pop()
        i = self.stack[-1]
        self.arcs[j] = i
        self.child_count[i] += 1
        self.action_seq.append(2)

    def sentence_oracle(self, dependency_heads, verbose=False):
        dependency_child_count = count_children(dependency_heads)
        self.shift_contexts = [] #context (word positions) on stack for shift decision - now for trigrams
        self.reduce_contexts = [] #context (word positions) on stack for shift/reduce actions - now for trigrams
        is_stuck = False
        #has_shifted_all = False
        while not self.is_terminal_configuration() and not is_stuck:
            #if self.is_buffer_empty() and not has_shifted_all:
            #    #record context for end of sentence token
            #    self.shift_contexts.append(self.get_stack_top_pair())
            #    has_shifted_all = True
            if self.get_stack_depth()==0:
                self.shift_contexts.append([0, 0])
                self.reduce_contexts.append([0, 0])
                self.shift()
            elif self.get_stack_depth()==1:
                self.shift_contexts.append([0, 0])
                self.reduce_contexts.append([0, 0])
                self.shift()
            else:
                i, j = self.get_stack_top_pair()
                #parsing order of node children is determined by which appears first
                if dependency_heads[i]==j and self.get_child_count(i)==dependency_child_count[i]:
                    self.left_arc()
                    self.reduce_contexts.append([i, j])
                elif dependency_heads[j]==i and self.get_child_count(j)==dependency_child_count[j]:
                    self.right_arc()
                    self.reduce_contexts.append([i, j])
                elif not self.is_buffer_empty():
                    self.shift_contexts.append([i, j])
                    #TODO need to test this line (missing before):
                    self.reduce_contexts.append([i, j])
                    self.shift()
                else:
                    is_stuck = True

        if not is_stuck and self.get_arcs()<>dependency_heads:
            is_stuck = True

        if verbose:
            if is_stuck:
                print 'stuck:', self.get_stack()
                print 'arcs found:', self.get_arcs()
                #print 'child counts:', dependency_child_count
            elif self.get_arcs()<>dependency_heads:
                print 'wrong parse:'
                print 'arcs found:', self.get_arcs()
        return not is_stuck

class ArcStandardLookAheadParser(ParserConfiguration):
    def __init__(self, sentence):
        ParserConfiguration.__init__(self, sentence)
        self.action_names = ['sh', 'las', 'ras']

    #next element in buffer
    def left_arc(self):
        j = self.buffer_[-1] 
        i = self.stack.pop()
        self.arcs[i] = j
        self.child_count[j] += 1
        self.action_seq.append(1)
     
    #next element in buffer 
    def right_arc(self):
        j = self.buffer_.pop()
        i = self.stack.pop()
        self.buffer_.append(i)
        self.arcs[j] = i
        self.child_count[i] += 1
        self.action_seq.append(2)

    #arc_standard with next element on the buffer, not already on stack
    def sentence_oracle(self, dependency_heads, verbose=True):
        dependency_child_count = count_children(dependency_heads)
        is_stuck = False
        while not self.is_terminal_configuration() and not is_stuck:
            if self.get_stack_depth()==0:
                self.shift()
            elif self.is_buffer_empty():
                is_stuck = True
            else:
                i, j = self.get_stack_top(), self.get_buffer_next()
                #parsing order of node children is determined by which appears first
                if dependency_heads[i]==j and self.get_child_count(i)==dependency_child_count[i]:
                    self.left_arc()
                elif dependency_heads[j]==i and self.get_child_count(j)==dependency_child_count[j]:
                    self.right_arc()
                elif not self.is_buffer_empty():
                    self.shift()
                else:
                    is_stuck = True

        if not is_stuck and self.get_arcs()<>dependency_heads:
            is_stuck = True

        if verbose:
            if is_stuck:
                print 'stuck:', self.get_stack()
                print 'arcs found:', self.get_arcs()
                #print 'child counts:', dependency_child_count
            elif self.get_arcs()<>dependency_heads:
                print 'wrong parse:'
                print 'arcs found:', self.get_arcs()
       
        return not is_stuck

class ArcEagerParser(ParserConfiguration):
    def __init__(self, sentence):
        ParserConfiguration.__init__(self, sentence)
        self.action_names = ['sh', 'lae', 'rae', 're']

    def left_arc(self):
        #if self.arc[self.stack[-1]]==-1:
        i = self.stack.pop()
        j = self.buffer_[-1]
        self.arcs[i] = j
        self.child_count[j] += 1
        self.action_seq.append(1)
        
    def right_arc(self):
        i = self.stack[-1]
        j = self.buffer_.pop()
        self.stack.append(j)
        self.arcs[j] = i
        self.child_count[i] += 1
        self.action_seq.append(2)

    def sentence_oracle(self, dependency_heads, verbose=False):
        dependency_child_count = count_children(dependency_heads)
        while not self.is_terminal_configuration():
            if self.get_stack_depth()==0:
                self.shift()
            elif self.is_buffer_empty():
                self.reduce_()
            else:
                i, j = self.get_stack_top(), self.get_buffer_next()
                #always add an arc if it is valid
                if dependency_heads[i]==j and self.get_child_count(i)==dependency_child_count[i] and not self.has_parent(j):
                    self.left_arc()
                elif dependency_heads[j]==i: 
                    self.right_arc()
                elif self.has_parent(i) and self.get_child_count(i)==dependency_child_count[i]:
                    self.reduce_()
                else:
                    self.shift()

        is_wrong_parse = False
        if -1 in self.get_arcs()[1:]:
            is_wrong_parse = True
            if verbose:
                print 'incomplete parse:'
                print 'arcs found:', self.get_arcs()
        elif self.get_arcs()<>dependency_heads:
            is_wrong_parse = True
            if verbose:
                print 'wrong parse:'
                print 'arcs found:', self.get_arcs()
        return not is_wrong_parse

class ArcHybridParser(ParserConfiguration):
    def __init__(self, sentence):
        ParserConfiguration.__init__(self, sentence)
        self.action_names = ['sh', 'lah', 'ra']

    def left_arc(self):
        #ArcEagerParser.left_arc(self)
        i = self.stack.pop()
        j = self.buffer_[-1]
        self.arcs[i] = j
        self.child_count[j] += 1
        self.action_seq.append(1)

    def right_arc(self):
        #ArcStandardParser.right_arc(self)
        j = self.stack.pop()
        i = self.stack[-1]
        self.arcs[j] = i
        self.child_count[i] += 1
        self.action_seq.append(2)

    def sentence_oracle(self, dependency_heads, verbose=False):
        dependency_child_count = count_children(dependency_heads)
        is_stuck = False
        while not self.is_terminal_configuration() and not is_stuck:
            if self.get_stack_depth()==0:
                self.shift()
            else:
                found_arc = False
                if not self.is_buffer_empty():
                    i, j = self.get_stack_top(), self.get_buffer_next()
                    #always add an arc if it is valid
                    if dependency_heads[i]==j and self.get_child_count(i)==dependency_child_count[i]:
                        self.left_arc()
                        found_arc = True
                if not found_arc and self.get_stack_depth()>=2:
                    i, j = self.get_stack_top_pair()
                    if dependency_heads[j]==i and self.get_child_count(j)==dependency_child_count[j]:
                        self.right_arc()
                        found_arc = True
                if not found_arc:
                    if not self.is_buffer_empty():
                        self.shift()
                    else:
                        is_stuck = True
                    
        if -1 in self.get_arcs()[1:]:
            is_stuck = True
            if verbose:
                print 'incomplete parse:'
                print 'arcs found:', self.get_arcs()
        elif self.get_arcs()<>dependency_heads:
            is_stuck = True
            if verbose:
                print 'wrong parse:'
                print 'arcs found:', self.get_arcs()
       
        return not is_stuck

class ArcStandardSecondOrderParser(ArcStandardParser):
    def __init__(self, sentence):
        ParserConfiguration.__init__(self, sentence)
        self.action_names = ['sh', 'la1', 'ra1', 're', 'la2', 'ra2']

    def left_arc_second_order(self):
        k = self.stack.pop()
        j = self.stack.pop()
        i = self.stack.pop()
        self.stack.append(j)
        self.stack.append(k)
        self.arcs[i] = k
        self.child_count[k] += 1
        self.action_seq.append(4)
        
    def right_arc_second_order(self):
        k = self.stack.pop()
        j = self.stack[-1]
        i = self.stack[-2]
        self.arcs[k] = i
        self.child_count[i] += 1
        self.action_seq.append(5)

    # extend to second order model for non-projective dependencies
    def sentence_oracle(self, dependency_heads, verbose=False):
        dependency_child_count = count_children(dependency_heads)
        is_stuck = False
        while not self.is_terminal_configuration() and not is_stuck:
            if self.get_stack_depth()<2:
                self.shift()
            else:
                i, j = self.get_stack_top_pair()
                #parsing order of node children is determined by which appears first
                if dependency_heads[i]==j and self.get_child_count(i)==dependency_child_count[i]:
                    self.left_arc()
                elif dependency_heads[j]==i and self.get_child_count(j)==dependency_child_count[j]:
                    self.right_arc()
                elif self.get_stack_depth()<3:
                    if not self.is_buffer_empty():
                        self.shift()
                    else:
                        is_stuck = True
                else:
                    #for second order stack dependencies
                    i, k = self.get_stack_second_order_pair()
                    if dependency_heads[i]==k and self.get_child_count(i)==dependency_child_count[i]:
                        self.left_arc_second_order()
                    elif dependency_heads[k]==i and self.get_child_count(k)==dependency_child_count[k]:
                        self.right_arc_second_order()
                    elif not self.is_buffer_empty():
                        self.shift()
                    else:
                        is_stuck = True

        if not is_stuck and self.get_arcs()<>dependency_heads:
            is_stuck = True

        if verbose:
            if is_stuck:
                print 'stuck:', self.get_stack()
                print 'arcs found:', self.get_arcs()
                #print 'child counts:', dependency_child_count
            elif self.get_arcs()<>dependency_heads:
                print 'wrong parse:'
                print 'arcs found:', self.get_arcs()
        return not is_stuck

def count_children(dependency_heads):
    child_count = [0]*len(dependency_heads)
    for j in dependency_heads[1:]:
        child_count[j] +=1
    return child_count

def is_projective_dependency(dependency_heads):
    for i in range(len(dependency_heads)-1):
        for j in range(i+1, len(dependency_heads)):
            if ((dependency_heads[i]<i and 
                  (dependency_heads[j]<i and dependency_heads[j]>dependency_heads[i])) or
                (dependency_heads[i]>i and dependency_heads[i]>j and
                  (dependency_heads[j]<i or dependency_heads[j]>dependency_heads[i])) or
                (dependency_heads[i]>i and dependency_heads[i]<j and
                  (dependency_heads[j]>i and dependency_heads[j]<dependency_heads[i]))):
                return False
    return True

def process_conll_data(data_dir, file_name):
    in_file = open(data_dir + file_name, 'r')
    conll = [[line.split('\t') for line in sent.split('\n')] 
                for sent in in_file.read().split('\n\n')[:-1]]
    in_file.close() 
    #print conll
    word_sentences = [[line[1] for line in sent] for sent in conll]
    dependencies = [[-1] + [int(line[6]) for line in sent] for sent in conll]
   
    #write word sentences to file
    s_file = open(data_dir + file_name + '.sentences', 'w')
    for sent in word_sentences:
        s_file.write(' '.join(sent) + '\n')
    s_file.close()

    #write dependencies to file
    d_file = open(data_dir + file_name + '.dependencies', 'w')
    for sent in dependencies:
        for i in sent:
            d_file.write(str(i) + ' ')
        d_file.write('\n')
    d_file.close()
    
    word_table = dict()
    word_count = 1 #0 is placeholder for sentence init
    #compile word lookup table
    for sent in word_sentences:
        for word in sent:
            if not word_table.has_key(word):
                word_table[word] = word_count
                word_count += 1
    #compile vocabulary list
    word_list = ['']*word_count
    word_list[0] = 'ROOT' #'<s>'
    #don't have explicit '</s>'
    for word in word_table.keys():  
        word_list[word_table[word]] = word
    
    #convert sentences
    sentences = [[0] + [word_table[word] for word in sent] for sent in word_sentences]
    #print word_list
    #print sentences

    return word_table, word_list, sentences, dependencies

    #nice if we only want a set
    #word_set = set()
    #for sent in word_sentences:
    #word_set = word_set.union(set(sent))
    #print word_set

    #word_sentences = []
    #dependencies = []
    #for sent in conll:
    #word_sentence = [line[1] for line in sent]
    #sentence_dependency = [line[6] for line in sent]
    
    #to remove punctuation (not necessary now...)
    #punct = [dep=='_' for i, dep in enumerate(sentence_dependency)]
    #word_sentence = map(lambda x: x[0], filter(lambda x: x[1], 
    #    [(word, sentence_dependency[i]!='_') for i, word in enumerate(word_sentence)]))
    #sentence_dependency = filter(lambda x: x!='_', sentence_dependency)

    #print word_sentence
    #print sentence_dependency
    
    #word_sentences.append(word_sentence)
    #dependencies.append(map(int, sentence_dependency))

if __name__=='__main__':
    train_dir = ''  #'parsing/'
    train_file = 'dutch_alpino_dev.conll' 
    [dictionary, vocabulary, sentences, dependencies] = process_conll_data(train_dir, train_file)
    n_projective = 0
    n_projective_stuck = 0
    n_non_projective_found = 0
    n_non_projective_stuck = 0
    w_ctx_f = open(train_dir + train_file + '.words.contexts', 'w')
    a_ctx_f = open(train_dir + train_file + '.actions.contexts', 'w')

    for i, sentence in enumerate(sentences):
        is_projective = is_projective_dependency(dependencies[i])
        #if is_projective:
        #    print dependencies[i]
        #else:
        #    print dependencies[i], 'Non-projective'

        #else:
        parser = ArcStandardParser(sentence) 
        #parser = ArcStandardSecondOrderParser(sentence)
        has_parse = parser.sentence_oracle(dependencies[i])
        w_contexts = parser.get_shift_contexts(vocabulary)
        for ctx in w_contexts:
            w_ctx_f.write(ctx[0] + ' ' + ' '.join(ctx[1]) + '\n')
        a_contexts = parser.get_reduce_contexts(vocabulary)
        for ctx in a_contexts:
            a_ctx_f.write(ctx[0] + ' ' + ' '.join(ctx[1]) + '\n')

        #print 'actions:', parser.get_action_str(), '\n'
        if has_parse:
            if is_projective:
                n_projective += 1
            else:
                n_non_projective_found += 1
        else:
            if is_projective:
                n_projective_stuck += 1
            else:
                n_non_projective_stuck += 1

    w_ctx_f.close()
    a_ctx_f.close()
    print 'total sentences:', len(sentences)
    print 'projective:', n_projective
    print 'non-projective found:', n_non_projective_found
    print 'non-projective stuck:', n_non_projective_stuck
    print 'projective stuck:', n_projective_stuck

