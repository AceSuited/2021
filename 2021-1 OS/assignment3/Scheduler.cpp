//
// Created by sodap on 2021-05-25.
//

#include "Scheduler.h"

// constructor
Scheduler::Scheduler() {
    for(int i =0; i <10; i++){
        deque<Process> queue;
        runQueue.push_back(queue);
    }

}

bool Scheduler:: comparator(const Process &a, const Process &b){
    /// comparator method for sorting algorithm
    if(a.pid < b.pid){
        return true;
    }
    else{
        return false;
    }
}

void Scheduler::registerProcess(Process& process) {
    /// register process into running Queue.
    /// multilevel-queue. priority0~4 : FCFS Algorithm
    ///                   priority5~9 : Round Robin Algorithm with time quantum 10
    if(process.priority <=4) {
        runQueue[process.priority].push_back(process);
    }
    else{
        process.timeQuantum = 10;
        runQueue[process.priority].push_back(process);
    }

}

bool Scheduler::scheduleProcess() {
    /// schedule a process to be executed for a given cycle time.
    /// return true if any process is scheduled as a running process
    /// return false if no process is scheduled.

    bool scheduled = false;

    /// if current process` priority is over 4(which means, it follows round robin algorithm),
    /// check the time quantum expiration.
    if(runningProcess.priority > 4 && runningProcess.timeQuantum == 0){
        registerProcess(runningProcess);
        runningProcess = Process();
        cout<< "quantum expired" << endl;
    }

    // if there are no running process, just loop running queue from 0 level to 9 level.
    if(runningProcess.pid==-1){
        for(int i =0; i < 10; i++){
            if(!runQueue[i].empty()){
                runningProcess  = runQueue[i].front();
                runQueue[i].pop_front();
                scheduled = true;
                break;
            }
        }
    }
    // if there are running process, check whether if there is process with higher priority,
    // if higher priority process exists, replace running process.
    else{

        for(int i =0; i < runningProcess.priority; i++){
            if(!runQueue[i].empty()){
                registerProcess(runningProcess);
                runningProcess= runQueue[i].front();
                runQueue[i].pop_front();
                scheduled = true;
                break;
            }
        }
    }

  
    return scheduled;
}



vector<Process> Scheduler :: getAliveProcesses(){
    /// helper method for printing output.
    /// returns vector filled with alive processes in scheduler.
    vector<Process> aliveProcesses;
    for(int i =0; i< runQueue.size(); i++){


            for (int j = 0; j < runQueue[i].size(); j++) {
                Process p = runQueue[i].front();
                aliveProcesses.push_back(runQueue[i].front());
                runQueue[i].pop_front();
                runQueue[i].push_back(p);


        }
    }

    if(!IOWaitList.empty()){
        for(int i = 0; i < IOWaitList.size(); i ++){
            aliveProcesses.push_back(IOWaitList[i]);
        }
    }
    if(!sleepList.empty()){
        for(int i = 0; i < sleepList.size(); i ++){
            aliveProcesses.push_back(sleepList[i]);
        }

    }
    if(runningProcess.pid != -1){
        aliveProcesses.push_back(runningProcess);
    }
    if(aliveProcesses.size() > 1){
        sort(aliveProcesses.begin(), aliveProcesses.end());

        for (int i = 1; i < aliveProcesses.size(); i++) {
            if (aliveProcesses[i - 1].pid == aliveProcesses[i].pid){
                aliveProcesses.erase(aliveProcesses.begin() + i);
            }
        }

    }

    return aliveProcesses;

}


