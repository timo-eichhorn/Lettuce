SHELL       := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS   += --warn-undefined-variables --no-builtin-rules

.ONESHELL:
.DELETE_ON_ERROR:

# SRC  := SU3_main.cpp
# OUT  := LettuceSU3
# INC  := ./Eigen_3.4
# CXX  := icpx

#────────────────────────────────────────
# Compiler detection and OpenMP flags
#────────────────────────────────────────
CXX ?= icpx
# Override Make's default if necessary
ifeq ($(origin CXX), default)
    CXX := icpx
endif

# Compiling on and for Apple Silicon requires different flags
ARCH_NAME := $(shell uname -m)
ifeq ($(ARCH_NAME),arm64)
    CXX     := clang++
    OMPFLAG := -Xclang -fopenmp -lomp
else
    # Try fallbacks (clang++ and g++) only if the user did NOT override CXX
    ifeq ($(origin CXX), file)
        ifeq (,$(shell command -v $(CXX) 2>/dev/null))
            $(warning Default compiler $(CXX) not found. Trying clang++...)
            CXX := clang++
            ifeq (,$(shell command -v $(CXX) 2>/dev/null))
                $(warning Compiler clang++ not found. Trying g++...)
                CXX := g++
                ifeq (,$(shell command -v $(CXX) 2>/dev/null))
                    $(error No suitable C++ compiler found.)
                endif
            endif
        endif
    endif
    # Assign correct OMPFLAG flag based on compiler
    ifneq (,$(filter icpx%,$(CXX)))
        OMPFLAG := -fiopenmp
    else ifneq (,$(filter g++%,$(CXX)))
        OMPFLAG := -fopenmp
    else ifneq (,$(filter clang++%,$(CXX)))
        OMPFLAG := -fopenmp
    else
        $(warning Unsupported compiler: $(CXX) - You may encounter errors when trying to compile!)
        OMPFLAG :=
    endif
endif

CXX_BASE := $(notdir $(CXX))

#────────────────────────────────────────
# Source, output, and include paths
#────────────────────────────────────────
SRC       := SU3_main.cpp
OUT       := LettuceSU3
BENCH_SRC := SU3_benchmark.cpp
BENCH_OUT := LettuceSU3_benchmark
INC       := ./Eigen_3.4
FLGS      ?=

#────────────────────────────────────────
# Compiler flags
#────────────────────────────────────────
SIMD_ARCH         ?= native
VALID_SIMD_ARCHES := native generic avx2 avx512 neon

ARCH_FLGS_native  = -march=native -mtune=native
ARCH_FLGS_generic =
ARCH_FLGS_avx2    = -march=core-avx2 -mtune=core-avx2
ARCH_FLGS_avx512  = -march=skylake-avx512 -mtune=skylake-avx512
ARCH_FLGS_neon    = -march=armv8

check-simd-arch = $(if $(filter $(SIMD_ARCH),$(VALID_SIMD_ARCHES)),,$(error Unsupported SIMD_ARCH='$(SIMD_ARCH)' (supported values: $(VALID_SIMD_ARCHES))))
ARCH_FLGS       = $(ARCH_FLGS_$(SIMD_ARCH))

# WARN_FLGS := -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast -Wcast-align -Wzero-as-null-pointer-constant -Wunused -Woverloaded-virtual -Wconversion -Wsign-conversion -Wfloat-conversion -Wformat=2 -Werror=vla -Wmisleading-indentation -Wnull-dereference -Wsitch-enum
WARN_FLGS := -Wall -Wextra -Wpedantic
# The following warnings are only recognized by g++ (and not by icpx and clang++)
# ifneq (,$(filter g++%,$(CXX_BASE)))
# 	WARN_FLGS += -Wduplicated-cond -Wduplicated-branches -Wlogical-op
# endif

COMMON_FLGS = $(check-simd-arch) -I $(INC) -std=c++20 -O3 -DNDEBUG -fno-math-errno -flto $(WARN_FLGS) $(OMPFLAG) $(ARCH_FLGS) $(FLGS)
# Old comment still valid?: Compiling with clang++ and static fails unless explicitly setting -fopenmp=libiomp5

#────────────────────────────────────────
# Targets
#────────────────────────────────────────
.PHONY: all check-compiler build generic avx2 avx512 neon cluster phadd compass pleiades bench run clean help

all: build

check-compiler:
	@$(if $(shell command -v $(CXX) 2>/dev/null),, \
		$(warning Compiler $(CXX) not found.) \
		$(if $(filter icpx%,$(CXX)), \
			$(warning If icpx is installed, initialize the oneAPI environment with 'source /opt/intel/oneapi/setvars'.),) \
		$(error Aborting: no C++ compiler $(CXX) available))
	@$(if $(filter gcc clang,$(CXX_BASE)),$(error Aborting: $(CXX) is a C compiler, not a C++ compiler),)

build: check-compiler
	$(CXX) $(COMMON_FLGS) $(SRC) -o $(OUT)

generic: SIMD_ARCH := generic
generic: build

avx2: SIMD_ARCH := avx2
avx2: build

avx512: SIMD_ARCH := avx512
avx512: build

neon: SIMD_ARCH := neon
neon: build

cluster phadd: SIMD_ARCH := avx2
cluster phadd: ARCH_FLGS := -march=broadwell -mtune=broadwell -static
cluster phadd: build

compass: SIMD_ARCH := avx2
compass: ARCH_FLGS := -march=alderlake -mtune=alderlake -static
compass: build

pleiades: SIMD_ARCH := avx2
pleiades: ARCH_FLGS := -march=znver2 -mtune=znver2 -static
pleiades: build

bench: check-compiler
	$(CXX) $(COMMON_FLGS) $(BENCH_SRC) -o $(BENCH_OUT)

run:
	./$(OUT)

clean:
	rm -f $(OUT) $(BENCH_OUT)

help:
	@echo "Available targets:"
	@echo "  build (default)  Build the main simulation"
	@echo "  generic          Build without host-specific SIMD tuning"
	@echo "  avx2             Build for AVX2 CPUs"
	@echo "  avx512           Build for AVX-512 CPUs"
	@echo "  neon             Build for ARM NEON CPUs"
	@echo "  cluster, phadd   Build for Broadwell cluster nodes"
	@echo "  compass          Build for Alder Lake nodes"
	@echo "  pleiades         Build for Zen 2 nodes"
	@echo "  bench            Build SU3_benchmark.cpp"
	@echo "  run              Run the main simulation"
	@echo "  clean            Remove built executables"
