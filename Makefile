# FLG1 := -std=c++20 -O3 -march=native -mtune=native -fiopenmp -DNDEBUG -fno-math-errno -flto -static
# FLG2 := -std=c++20 -O3 -march=broadwell -mtune=broadwell -fiopenmp -DNDEBUG -fno-math-errno -flto -static
# OUT  := LettuceSU3
# INC  := ./Eigen_3.4
# CC   := icpx


# all:
# 		$(CC) -I $(INC) SU3_main.cpp $(FLG1) $(FLGS) -o $(OUT)
# cluster:
# 		$(CC) -I $(INC) SU3_main.cpp $(FLG2) $(FLGS) -o $(OUT)
# run:
# 		./$(OUT)
# clean:
# 		rm $(OUT)


FLG1 := -std=c++20 -O3 -march=native -mtune=native -fopenmp -DNDEBUG -fno-math-errno -fcx-fortran-rules -flto -static
FLG2 := -std=c++20 -O3 -march=broadwell -mtune=broadwell -fopenmp -DNDEBUG -fno-math-errno -fcx-fortran-rules -flto -static
OUT  := LettuceSU3_noeigen
CC   := g++
override FLGS +=


all:
		$(CC) SU3_main.cpp $(FLG1) $(FLGS) -o $(OUT)
cluster:
		$(CC) SU3_main.cpp $(FLG2) $(FLGS) -o $(OUT)
run:
		./$(OUT)
clean:
		rm $(OUT)
