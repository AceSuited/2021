
#ifndef SIMULATOR_H
#define SIMULATOR_H
#include "Process.h"
#include "Scheduler.h"
#include <queue>
#include <iostream>
#include "PhysicalMemory.h"

// simulator class
// scheduler, buddy system and memory management system is implemented
class Simulator {
public:

    string pageReplaceOption;
    int VM_size, PM_size, page_size;
    int numPage, numFrame;
    int process_id_issued;
    int cycle;

    queue<Event> eventLog;
    Event currentEvent;

    Scheduler scheduler;

    PhysicalMemory physicalMemory;

    int numPageFault;

    FILE * sched;
    FILE * mem;

    Simulator(int VM_size, int PM_size, int page_size, queue<Event> eventLog,string pageReplacementOption);

    void step();

    void wakeSleepingProcess();

    void eventHandler();

    void printCycle(bool isThereNewlyScheduled);

    bool execute(Process &process);

    void simulate();

    void closeProcess();

    bool checkTerminateCondition();

    void initReferenceInfo();

    void updateReferenceInfo();
};


#endif
