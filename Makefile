FLG1 := -std=c++20 -O3 -march=native -mtune=native -fiopenmp -DNDEBUG -fno-math-errno -flto -static
FLG2 := -std=c++20 -O3 -march=broadwell -mtune=broadwell -fiopenmp -DNDEBUG -fno-math-errno -flto -static
OUT  := LettuceSU3
INC  := ./Eigen_3.4
CC   := icpx
override FLGS +=


all:
		$(CC) -I $(INC) SU3_main.cpp $(FLG1) $(FLGS) -o $(OUT)
cluster:
		$(CC) -I $(INC) SU3_main.cpp $(FLG2) $(FLGS) -o $(OUT)
cuda:
		nvcc --compiler-bindir=/opt/intel/oneapi/compiler/2023.2.1/linux/bin/icpx -I./Eigen_3.4 SU3_main.cpp -std=c++20 -O3 -arch=native -Xcompiler -march=native,-mtune=native,-fiopenmp,-DNDEBUG,-fno-math-errno,-flto -o LettuceSU3
cuda_cluster:
		nvcc --compiler-bindir=/opt/intel/oneapi/compiler/2023.2.1/linux/bin/icpx -I./Eigen_3.4 SU3_main.cpp -std=c++20 -O3 -arch=native -Xcompiler -march=broadwell,-mtune=broadwell,-fiopenmp,-DNDEBUG,-fno-math-errno,-flto -o LettuceSU3
run:
		./$(OUT)
clean:
		rm $(OUT)
