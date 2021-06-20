

#include "Process.h"

/// constructor for vacant Process
Process::Process() {
    pid = -1;
    priority = -1;
    code_fileName = "NONE";


    sleepTime = -1;
    timeQuantum = -1;




    line = -1;


}

Process::Process(Event event, int pid, int numPage, string pageReplaceOption) {
    /// constructor

    this->pid = pid;
    priority = event.priority;
    code_fileName = event.code_fileName;


    virtualMemory = VirtualMemory(numPage, pageReplaceOption);

    /// check process` priority and initialize time Quantum if it is a subject of round robin scheduling
    if(priority>=5){
        timeQuantum =10;
    }
    else{
        timeQuantum =-1;
    }

    line = 0;
}

void Process::initInstructions() {
    /// reads instruction from codes of corresponding file.
    /// make instruction and save it into instructionTable.
    FILE *inputFile = fopen((directory + "/" + code_fileName).c_str(), "r");
    int num_operation;
    fscanf(inputFile, "%d\n", &num_operation);

    for(int i = 0; i < num_operation; i++){
        Instruction operation;
        fscanf(inputFile, "%d\t%d\n", &operation.op_code, &operation.arg);

        instructionTable.push(operation);
    }


}

