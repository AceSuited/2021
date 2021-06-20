
#ifndef PROCESS_H
#define PROCESS_H

#include <string>
#include<queue>
#include <iostream>
#include <cstring>
#include "VirtualMemory.h"
using namespace std;

extern string directory ;



struct Event{
    bool IO_flag;
    int time;
    int priority;
    string code_fileName;
    int pid;
    bool isDone = false;
};

struct Instruction{
    int op_code;
    int arg;
};

class Process {

public:
    string code_fileName;
    queue<Instruction> instructionTable;

    int pid;
    int priority;
    int sleepTime;
    int timeQuantum;



    int line;
    Instruction currentInstruction;

    VirtualMemory virtualMemory;

    Process();
    Process(Event event, int pid, int numPage, string pageReplaceOption);

    void initInstructions();

    bool operator <(Process &p)  {
        return (this->pid < p.pid);
        }


};


#endif
