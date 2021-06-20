project3: VirtualMemory.o Process.o Scheduler.o PhysicalMemory.o Simulator.o  main.o
	g++ -o project3 Process.o Scheduler.o PhysicalMemory.o Simulator.o VirtualMemory.o main.o

VirtualMemory.o: VirtualMemory.h VirtualMemory.cpp
	g++ -c -o VirtualMemory.o VirtualMemory.cpp

Process.o: Process.h Process.cpp VirtualMemory.h
	g++ -c -o Process.o Process.cpp

Scheduler.o: Scheduler.h Scheduler.cpp Process.h
	g++ -c -o Scheduler.o Scheduler.cpp

PhysicalMemory.o: PhysicalMemory.h PhysicalMemory.cpp Process.h
	g++ -c -o PhysicalMemory.o PhysicalMemory.cpp

Simulator.o: Simulator.h Simulator.cpp Process.h Scheduler.h PhysicalMemory.h
	g++ -c -o Simulator.o Simulator.cpp

main.o: Process.h Simulator.h main.cpp
	g++ -c -o main.o main.cpp

clean:
	rm *.o project3