OxDP
====

Generative dependency parsing models. 

Probability models are based on the oxlm PYP and LBL language models.

#### Dependecies

[Cmake](http://www.cmake.org/) is needed to build the toolkit. The external dependencies are [Boost](http://www.boost.org/) and [OpenMP](http://en.wikipedia.org/wiki/OpenMP). Cmake looks for Boost and OpenMP in the locations where the libraries are installed by default using the operating system's package management tool.

#### Installation

Run the following to compile the code for the first time:

    cd oxdp
    mkdir build
    cd build
    cmake ../src
    make

Unit tests for lbl:

    cd build
    make all_tests

### Prepare the training and test data

The training and test data uses the CoNLL dependency parsing format. The script 

oxdp/scripts/preprocess-conll.py 

performs the relevant preprocessing, including replacing words occuring less than a minimum frequency cutoff with the `<unk>` symbol. 

### Training and testing

oxdp/bin/train_pyp_dp

implements supervised parsing with a PYP probability model.



