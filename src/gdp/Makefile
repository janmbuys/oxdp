LOCAL_LIBS=${HOME}/packages
CXXFLAGS =  -std=c++11 -g -O3 -Wall -I.. -m64 # -fopenmp -I${LOCAL_LIBS}/include -I/usr/include/eigen3 
LDFLAGS = -std=c++11 -g -O3 # -fopenmp -L${LOCAL_LIBS}/lib64 -L${LOCAL_LIBS}/lib -lboost_program_options -lboost_serialization

all: hpyp_dp

hpyp_dp: transition_parser.o hpyp_dp.o
	$(CXX) ${LDFLAGS} transition_parser.o hpyp_dp.o -o hpyp_dp 
#hpyp_dp: ../corpus/corpus.o transition_parser.o hpyp_dp.o
#	$(CXX) ${LDFLAGS} ../corpus/corpus.o transition_parser.o hpyp_dp.o -o hpyp_dp 

transition_parser: transition_parser.o transition_parser.h

#corpus: ../corpus/corpus.o ../corpus/corpus.h

%.o: %.cc
	${CXX} $(CXXFLAGS) -c $< -o $@

# %.o: %.c
#	${CC} $(CFLAGS) -c $< -o $@

## stuff below here is optional

clean:
	rm -f *.o ../corpus/*.o 

clobber: clean
	rm hpyp_dp 
