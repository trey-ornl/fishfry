SRC := $(wildcard *.cpp)
OBJ := $(SRC:.cpp=.o)
EXE := fishfry

default: $(EXE)

$(EXE): $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $(EXE) $(LIBS)

main.o: main.cpp FishFry.hpp gpu.hpp

PoissonPeriodic3x1DBlockedGPU.o: PoissonPeriodic3x1DBlockedGPU.cpp PoissonPeriodic3x1DBlockedGPU.hpp gpu.hpp

clean:
	rm -f $(OBJ) $(EXE) core*

