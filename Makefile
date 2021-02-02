CXX       ?= g++
CXX_FLAGS += -std=c++17 -ggdb -fPIC -g -pedantic -Wall -Wextra

BIN     := .
LIB		:= .
TESTS	:= tests
SRC     := src

INCLUDES := -I$(SRC) \
			-I/usr/local/include/opencv4

LIBRARIES   := -L/usr/local/lib -lopencv_core -lopencv_videoio \
			   -lopencv_cudaimgproc -lopencv_cudastereo -lopencv_cudawarping -lopencv_cudafilters \
			   -lopencv_ximgproc -pthread
 
NAME	:= libCSI_Stereo.so
TESTNAME:= test_CSI_Stereo

all: $(LIB)/$(NAME) $(BIN)/$(TESTNAME)

fresh: clean all

$(LIB)/$(NAME): $(SRC)/*.cpp
	$(CXX) $(CXX_FLAGS) -shared $(INCLUDES) $^ -o $@ $(LIBRARIES)

$(BIN)/$(TESTNAME): $(TESTS)/*.cpp
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $^ -o $@ $(LIBRARIES) -L$(LIB) -lCSI_Stereo -lopencv_highgui -lopencv_imgcodecs

clean:
	-rm -f $(LIB)/$(NAME)
	-rm -f $(BIN)/$(TESTNAME)
