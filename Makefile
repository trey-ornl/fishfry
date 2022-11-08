SRC := $(wildcard *.cpp)
OBJ := $(SRC:.cpp=.o)
EXE := fishfry

default: $(EXE)

$(EXE): $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $(EXE) $(LIBS)

main.o: main.cpp FishFry.hpp gpu.hpp ParisPencil.hpp

ParisPencil.o: ParisPencil.cpp ParisPencil.hpp gpu.hpp

clean:
	rm -f $(OBJ) $(EXE) core*

