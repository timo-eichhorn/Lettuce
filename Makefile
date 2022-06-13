FLG1 := -std=c++20 -O3 -march=native -mtune=native -fiopenmp -DNDEBUG -fno-math-errno -flto -static
FLG2 := -std=c++20 -O3 -march=broadwell -mtune=broadwell -fiopenmp -DNDEBUG -fno-math-errno -flto -static
FLGS := 
OUT  := LettuceSU3
INC  := ./Eigen_3.4
CC   := icpx


all:
		$(CC) -I $(INC) SU3_main.cpp $(FLG1) $(FLGS) -o $(OUT)
cluster:
		$(CC) -I $(INC) SU3_main.cpp $(FLG2) $(FLGS) -o $(OUT)
run:
		./$(OUT)
clean:
		rm $(OUT)
