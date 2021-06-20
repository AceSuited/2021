.DEFAULT_GOAL:=all

all: miniShell program1 program2 program3

miniShell: miniShell.cpp
	g++ -std=c++11 -o miniShell miniShell.cpp
program1: program1.cpp
	g++ -std=c++11 -o program1 program1.cpp
program2: program2.cpp
	g++ -std=c++11 -o program2 program2.cpp
program3: program3.cpp
	g++ -std=c++11 -pthread -o program3 program3.cpp
clean:
	rm miniShell program1 program2 program3