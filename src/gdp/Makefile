LOCAL_LIBS=${HOME}/packages
CXXFLAGS =  -std=c++0x -g -O3 -fopenmp -Wall -I${LOCAL_LIBS}/include -m64 -I/usr/include/eigen3 -I..
LDFLAGS = -std=c++0x -g -O3 -fopenmp -L${LOCAL_LIBS}/lib64 -L${LOCAL_LIBS}/lib -lboost_program_options -lboost_serialization

all: hpyp_dp hpyp_dp_train hpyp_dp_generate #hpyplm_contexts 

hpyp_dp: hpyp_dp.o
	$(CXX) -o $@ $^ $(LDFLAGS)

hpyplm_contexts: hpyplm_contexts.o
	$(CXX) -o $@ $^ $(LDFLAGS)

hpyp_dp_train: hpyp_dp_train.o
	$(CXX) -o $@ $^ $(LDFLAGS)

hpyp_dp_generate: hpyp_dp_generate.o
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	${CXX} $(CXXFLAGS) -c $< -o $@

%.o: %.c
	${CC} $(CFLAGS) -c $< -o $@


## stuff below here is optional

BOOST_INCLUDE=$(LOCAL_LIBS)/include

#.PHONY: depend echo
#depend:
#	$(CXX) -I.. -MM $(CXXFLAGS) hpyplm.{cc,h} dhpyplm.{cc,h} > makefile.depend

clean:
	rm -f *.o

clobber: clean
	rm makefile.depend hpyplm_contexts hpyp_dp hpyp_dp_train hpyp_dp_generate
