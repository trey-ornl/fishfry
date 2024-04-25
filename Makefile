HEAD := $(wildcard *.hpp)
SRC := $(wildcard *.cpp)
OBJ := $(SRC:.cpp=.o)
EXE := fishfry

default: $(EXE)

$(EXE): $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $(EXE) $(LIBS)

$(OBJ): $(HEAD) Makefile

clean:
	rm -f $(OBJ) $(EXE) core*

