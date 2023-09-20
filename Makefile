SRC  := SU3_main.cpp
OUT  := LettuceSU3
INC  := ./Eigen_3.4
CXX  := icpx

# OMP flag differs between compilers
ifeq ($(CXX),icpx)
    OMPFLAG := -fiopenmp
else ifeq ($(CXX),g++)
    OMPFLAG := -fopenmp
else ifeq ($(CXX),gcc)
    OMPFLAG := -fopenmp
else ifeq ($(CXX),clang)
    OMPFLAG := -fopenmp
else
    $(error Unsupported compiler: $(CXX))
endif

# Other compiler flags
COMMON_FLGS := -I $(INC) -std=c++20 -O3 -DNDEBUG -fno-math-errno -flto -static $(OMPFLAG)
ARCH_FLGS   := -march=native -mtune=native

# If using nvcc allow for different host compilers
HOSTCOMPILER := $(shell which $(CXX))
override FLGS +=

.PHONY: all build cluster compass cuda cuda_cluster run clean

all:    build
build:
		$(CXX) $(SRC) $(COMMON_FLGS) $(ARCH_FLGS) -o $(OUT)
cluster:
		$(MAKE) build ARCH_FLGS="-march=broadwell -mtune=broadwell"
compass:
		$(MAKE) build ARCH_FLGS="-march=alderlake -mtune=alderlake"
cuda:
		nvcc --compiler-bindir=$(HOSTCOMPILER) -I $(INC) $(SRC) -std=c++20 -O3 -arch=native -Xcompiler -march=native,-mtune=native,$(OMPFLAG),-DNDEBUG,-fno-math-errno,-flto -o $(OUT)
cuda_cluster:
		nvcc --compiler-bindir=$(HOSTCOMPILER) -I $(INC) $(SRC) -std=c++20 -O3 -arch=native -Xcompiler -march=broadwell,-mtune=broadwell,$(OMPFLAG),-DNDEBUG,-fno-math-errno,-flto -o $(OUT)
run:
		./$(OUT)
clean:
		rm -f $(OUT)
