from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from collections import defaultdict
import copy
import sys

import numpy as np
import keras

from extract_training_data import FeatureExtractor, State

class Parser(object): 

    def __init__(self, extractor, modelfile):
        self.model = keras.models.load_model(modelfile)
        self.extractor = extractor
        
        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):
        state = State(range(1,len(words)))
        state.stack.append(0)    

        while state.buffer: 
            # pass
            # TODO: Write the body of this loop for part 4
            feature = self.extractor.get_input_representation(words, pos, state)
            actions = self.model.predict(np.vstack([feature]))
            lst = list(zip(actions.tolist()[0], range(len(actions.tolist()[0]))))
            lst.sort(reverse=True)
            label = None

            for probability, i in lst:
                label = self.output_labels[i]
                b1 = (label[0] == 'left_arc' or label[0] == 'right_arc') and not state.stack
                b2 = label[0] == 'shift' and len(state.buffer) == 1 and state.stack
                b3 = state.stack and state.stack[-1] == 0 and label[0] == 'left_arc'
                if b1 or b2 or b3:
                    continue
                break

            if label[0] == "shift":
                state.shift()
            elif label[0] == "left_arc":
                state.left_arc(label[1])
            elif label[0] == "right_arc":
                state.right_arc(label[1])

        result = DependencyStructure()
        for p,c,r in state.deps: 
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))
        return result 
        

if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file: 
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
        
