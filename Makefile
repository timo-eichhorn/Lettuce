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
else ifeq ($(CXX),clang++)
    OMPFLAG := -fopenmp
# else
#     $(error Unsupported compiler: $(CXX))
endif

# Compiling on and for Apple Silicon
ARCH_NAME   := $(shell uname -m)
ifeq ($(ARCH_NAME),arm64)
    CXX     := clang++
    OMPFLAG := -Xclang -fopenmp -lomp
endif

ifeq (,$(shell which $(CXX)))
  $(warning Compiler $(CXX) not found)# in $(PATH)")
  UNSUPPORTED_COMPILER := true
  ifeq ($(CXX),icpx)
    $(warning If $(CXX) is installed, perhaps the oneAPI environment has not been initialized yet (per default by running "source /opt/intel/oneapi.setvars"))
  endif
endif

# Other compiler flags
# Compiling with clang++ and static fails unless explicitly setting -fopenmp=libiomp5
COMMON_FLGS := -I $(INC) -std=c++20 -O3 -DNDEBUG -fno-math-errno -flto -Wall -Wextra -Wpedantic $(OMPFLAG)
ARCH_FLGS   := -march=native -mtune=native
# WARN_FLGS   := -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wzero-as-null-pointer-constant -Wunused -Woverloaded-virtual -Wpedantic -Wconversion -Wsign-conversion -Wfloat-conversion -Wformat=2 -Werror=vla -Wmisleading-indentation -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wnull-dereference

# Allow additional flags (like DFIXED_SEED) from command line
COMMON_FLGS += $(FLGS)

.PHONY: all build cluster compass run clean

all:    build
build:
		$(CXX) $(SRC) $(COMMON_FLGS) $(ARCH_FLGS) -o $(OUT)
cluster:
		$(MAKE) build ARCH_FLGS="-march=broadwell -mtune=broadwell"
compass:
		$(MAKE) build ARCH_FLGS="-march=alderlake -mtune=alderlake"
pleiades:
		$(MAKE) build ARCH_FLGS="-march=znver2 -mtune=znver2"
run:
		./$(OUT)
clean:
		rm -f $(OUT)
