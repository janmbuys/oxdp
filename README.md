OxDP
====

Generative dependency parsing models. 

Probability models are based on the oxlm PYP and neural (LBL) language models.

#### Dependencies

[Cmake](http://www.cmake.org/) is needed to build the toolkit. The external dependencies are [Boost](http://www.boost.org/) and [OpenMP](http://en.wikipedia.org/wiki/OpenMP). Cmake looks for Boost and OpenMP in the locations where the libraries are installed by default using the operating system's package management tool.

#### Installation

Run the following to compile the code for the first time:

    cd oxdp
    mkdir build
    cd build
    cmake ../src
    make

### Prepare the training and test data

The training and test data uses the CoNLL dependency parsing format. The script 

oxdp/scripts/preprocess-conll.py 

performs the relevant preprocessing, including replacing words occurring less than a minimum frequency cutoff with the `<unk>` symbol. 

### Training and testing

#### Neural parser:

Create an 'oxdp.ini' file with the following contents:

    iterations=20
    minibatch-size=128
    randomise=true
    lambda-lbl=10.0
    representation-size=256
    step-size=0.05
    diagonal-contexts=false
    activation=sigmoid
    threads=4
    labelled-parser=true
    lexicalised=true
    predict-pos=true
    tag-pos=false
    label-features=true
    distance-features=false
    morph-features=false
    parser-type=arcstandard
    context-type=more-extended
    direction-det=false
    sum-over-beam=false
    root-first=true
    bootstrap=false
    bootstrap-iter=0
    complete-parse=true
    max-beam-increment=100
    num-particles=100
    generate-samples=0

Partition the vocabulary using [agglomerative Brown clustering](https://github.com/percyliang/brown-cluster):

    brown-cluster/wcluster --c num-clusters \
                           --threads=10 \
                           --text training.unk.en \
                           --output_dir=clusters

Set `num-clusters` to `sqrt(vocabulary_size)`.

Train the model:

oxdp/bin/train\_sgd -c oxdp.ini -i train.conll --test-set dev.conll --test-set2 test.conll --test-out-file test.out.conll --model-out model.bin --class-file clusters/paths 

#### Neural syntactic language modelling:

Create an 'oxdp.ini' file with the following contents:

    iterations=20
    minibatch-size=128
    randomise=true
    lambda-lbl=10.0
    representation-size=256
    step-size=0.05
    diagonal-contexts=false
    activation=sigmoid
    noise-samples=0
    threads=4
    labelled-parser=false
    lexicalised=true
    parser-type=arcstandard
    context-type=extended-with-ngram
    predict-pos=false
    tag-pos=false
    label-features=false
    distance-features=false
    morph-features=false
    direction-det=true
    sum-over-beam=false
    root-first=true
    bootstrap=false
    bootstrap-iter=5
    complete-parse=false
    max-beam-increment=1
    num-particles=1000
    generate-samples=0

Train the model as before.

#### Bayesian parser:

Create an 'oxdp.ini' file with the following contents:

    iterations=20
    minibatch-size=1
    minibatch-size-unsup=1
    randomise=true
    labelled-parser=true
    parser-type=arcstandard
    lexicalised=true
    sum-over-beam=false
    semi-supervised=false
    direction-det=false
    particle-resample=true
    num-particles=100
    max-beam-size=256

Train the model:

oxdp/bin/train\_gibbs -c oxdp.ini -i train.conll --test-set dev.conll --test-output-file out.conll 

### Citation

If you use this code, please please cite one of the following papers:

1. Generative Incremental Dependency Parsing with Neural Networks - Jan Buys and Phil Blunsom. ACL 2015. [Paper](http://www.aclweb.org/anthology/P15-2142).

2. A Bayesian Model for Generative Transition-based Dependency Parsing - Jan Buys and Phil Blunsom. Depling 2015. [Paper](https://www.cs.ox.ac.uk/files/7413/depling2015-BuysBlunsom-BayesianGenerativeDependencyParsing.pdf).

 

