
#ifndef SCHEDULER_H
#define SCHEDULER_H
#include "Process.h"
#include <queue>
#include <deque>
#include<vector>
#include<algorithm>
#include <set>


class Scheduler {
public:
    vector<deque<Process>> runQueue;
    vector<Process> sleepList;
    vector<Process> IOWaitList;
    Process runningProcess;



    Scheduler();

    void registerProcess(Process& process);

    bool scheduleProcess();

    vector<Process> getAliveProcesses();

    bool comparator(const Process &a, const Process &b);


};


#endif
