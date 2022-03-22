SRC := $(wildcard *.cpp)
OBJ := $(SRC:.cpp=.o)
EXE := fishfry

default: $(EXE)

$(EXE): $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $(EXE) $(LIBS)

main.o: main.cpp FishFry.hpp gpu.hpp

HenryPeriodic.o: HenryPeriodic.cpp HenryPeriodic.hpp gpu.hpp

ParisPeriodic.o: ParisPeriodic.cpp ParisPeriodic.hpp gpu.hpp

clean:
	rm -f $(OBJ) $(EXE) core*

