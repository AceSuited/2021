
#include "Simulator.h"

// constructor
Simulator::Simulator( int VM_size, int PM_size, int page_size,queue<Event> eventLog, string pageReplaceOption) {
    this->pageReplaceOption = pageReplaceOption;

    // configure directory option
    string schedFileName = directory + "/scheduler.txt";
    string memFileName = directory + "/memory.txt";

    // make files for outputs
    sched = fopen (schedFileName.c_str(), "w");
    mem = fopen(memFileName.c_str(), "w");


    cycle = 0;

    // variables for events
    this->eventLog = eventLog;
    currentEvent = eventLog.front();
    this->eventLog.pop();

    // process id issued by this simulator system
    process_id_issued = 0;

    this->VM_size = VM_size;
    this->PM_size = PM_size;
    this->page_size =page_size;

    // calculate configured memory size from input file
    numPage = VM_size / page_size;
    numFrame = PM_size / page_size;
    numPageFault = 0;


    // physical memory class for system
    physicalMemory =  PhysicalMemory(numFrame, pageReplaceOption);

}

// represents a cycle of simulator`s operation
void Simulator::step() {
    bool isThereNewlyScheduled;
    bool isRunningProcessBlocked;

    // check whether sleeping processes are waking up in this step
    wakeSleepingProcess();

    // Get a next instruction from input file and execute instructions
    eventHandler();

    // Schedule a process to be execute in this cycle
    isThereNewlyScheduled = scheduler.scheduleProcess();

    // next if statement will be executed when page replacement algorithm is "Sampled LRU Algorithm"
    // renew reference bytes of all the running processes`s allocated memories, every 8 cycle.
    // all reference bit to 0
    if (strcmp(pageReplaceOption.c_str(), "sampled") == 0 && cycle % 8 ==0 ){
        initReferenceInfo();
    }

    // execute Process
    isRunningProcessBlocked = execute(scheduler.runningProcess);


    // print out the results of this cycle
    printCycle(isThereNewlyScheduled);

    // if current scheduled process is terminated(run out of instructions to execute), then close the process.
    if(scheduler.runningProcess.instructionTable.empty() && scheduler.runningProcess.pid != -1){
        isRunningProcessBlocked = true;
        closeProcess();
    }

    // if running process is blocked or closed, let running process be a empty process.
    if(isRunningProcessBlocked){
        scheduler.runningProcess = Process();
    }

    // increment cycle number
    cycle ++;
}

void Simulator::wakeSleepingProcess() {
    //// method for check whether sleeping processes is to be awake

    // examine every processes in sleeping queue of scheduler object
    for(int i= 0; i < scheduler.sleepList.size(); i++){

        // decrease remaining sleeping time of every process.
        scheduler.sleepList[i].sleepTime -= 1;

        // if there are processes or a process which remaining sleeping time is 0, awake it(or them) by adding it(them) to running queue of scheduler object.
        if(scheduler.sleepList[i].sleepTime == 0){
            scheduler.registerProcess(scheduler.sleepList[i]);
            scheduler.sleepList.erase(scheduler.sleepList.begin() + i);
        }
    }

}

void Simulator::eventHandler() {
    ////  Get a next instruction from input file and execute instructions if current cycle is a cycle to be execute the current instruction


    // if an event is already handled and there are more events left in eventLog variable, configure next event to be handled.
    if(currentEvent.isDone && !eventLog.empty()){
        currentEvent = eventLog.front();
        eventLog.pop();
    }

    // if current cycle is a cycle to execute current event, execute it
    if(cycle == currentEvent.time){

        while(true) {

            // IO Event
            if (currentEvent.IO_flag) {
                int calledPid = currentEvent.pid;
                for (int i = 0; i < scheduler.IOWaitList.size(); i++) {

                    // examine matching processes in IO wait list of scheduler object.
                    // then register into running Queue which means end of IO wait for that process.
                    if (scheduler.IOWaitList[i].pid == calledPid) {
                        scheduler.registerProcess(scheduler.IOWaitList[i]);
                        scheduler.IOWaitList.erase(scheduler.IOWaitList.begin() + i);
                    }
                }
            }
            // New Program Event
            else {

                // create new Process object
                Process newProcess(currentEvent, process_id_issued, numPage, pageReplaceOption);
                process_id_issued++;

                // read codes and save info of instructions for new Process
                newProcess.initInstructions();

                // put new process into running queue
                scheduler.registerProcess(newProcess);

            }
            // current event is handled
            currentEvent.isDone = true;

            // if there are more events for current cycle time, then repeat above sequences for remaining events to be handled
            if(!eventLog.empty() && cycle == eventLog.front().time){
                currentEvent = eventLog.front();
                eventLog.pop();
            }
            // if no more events are left for current cycle, end.
            else{
                break;
            }
        }
    }

}

bool Simulator::execute(Process &process) {
    //// execute an instruction of scheduled running process for this cycle.

    // if there is no running process, skip this method by return true
    if(process.pid ==-1){
        return true;
    }


    bool isRunningProcessBlocked = false;

    // get an instruction to be executed for this cycle from instructionTable
    process.currentInstruction = process.instructionTable.front();
    process.instructionTable.pop();
    process.line += 1;


    /// Memory allocation
    if (process.currentInstruction.op_code == 0) {

        process.virtualMemory.memoryAllocate(process.currentInstruction.arg);


    }
    /// memory Access
    else if (process.currentInstruction.op_code == 1) {

        // get a pageBlock to be accessed in physical memory
        int page_id = process.currentInstruction.arg;
        PageBlock &accessingPageBlock = process.virtualMemory.pageBlockTable[page_id];

        // case miss! (page fault)
        if(!accessingPageBlock.isValidated){

            // allocate a new buddy block from physical memory
            FrameBlock allocatedFrameBlock =  physicalMemory.buddyAllocate(process, accessingPageBlock, scheduler);

            // update infos to accessing pageBlock
            accessingPageBlock.allocation_id  = allocatedFrameBlock.allocation_id;
            accessingPageBlock.isValidated = true;

            //  also update page table info to be written in output files
            for(int i = accessingPageBlock.start_idx; i <= accessingPageBlock.end_idx; i++){
                process.virtualMemory.memory[i].allocation_id = allocatedFrameBlock.allocation_id;
                process.virtualMemory.memory[i].valid_bit = 1;
                if (strcmp(pageReplaceOption.c_str(), "clock") == 0) {
                    process.virtualMemory.memory[i].reference_bit = 1;
                }
            }
            numPageFault++;
        // case hit!
        }else{

            // if page replacement algorithm is "lru", put accessed frame block to the top of stack,
            // so when page replacing operation is required, least accessed process can be selected as a victim by picking it up from bottom of the stack.
            if (strcmp(pageReplaceOption.c_str(), "lru") == 0) {
                for (int i = 0; i < physicalMemory.frameBlockTable.size(); i++) {
                    if (physicalMemory.frameBlockTable[i].allocation_id == accessingPageBlock.allocation_id) {
                        FrameBlock temp = physicalMemory.frameBlockTable[i];
                        physicalMemory.frameBlockTable.erase(physicalMemory.frameBlockTable.begin() + i);
                        physicalMemory.frameBlockTable.push_front(temp);
                        break;
                    }
                }
            }
            // if page replacement algorithm is "clock", additional page table info updating is required.(reference bit)
            if (strcmp(pageReplaceOption.c_str(), "clock") == 0) {
                for (int i = 0; i < physicalMemory.frameBlockTable.size(); i++) {
                    if (physicalMemory.frameBlockTable[i].allocation_id == accessingPageBlock.allocation_id) {
                        physicalMemory.frameBlockTable[i].reference_bit = 1;
                    }
                }
                for(int i = accessingPageBlock.start_idx; i <= accessingPageBlock.end_idx; i++){
                    process.virtualMemory.memory[i].reference_bit = 1;
                }
            }
        }

        // in case of "sampled lru" algorithm, update accessed pageBlock` reference bit to 1.
        if (strcmp(pageReplaceOption.c_str(), "sampled") == 0) {
            accessingPageBlock.referenceBit = 1;
            for (int i = accessingPageBlock.start_idx; i <= accessingPageBlock.end_idx; i++) {
                process.virtualMemory.memory[i].reference_bit = 1;
            }
        }


        ///  Memory release
        } else if (process.currentInstruction.op_code == 2) {

            // get a pageBlock to be released
            int page_id = process.currentInstruction.arg;
            PageBlock &releasingPageBlock = process.virtualMemory.pageBlockTable[page_id];

            // if releasing pageBlock is present in physical memory, find corresponding frameBlock in physical memory and deallocate it also.
            if(releasingPageBlock.isValidated){
                for(int i =0; i< physicalMemory.frameBlockTable.size(); i++){
                    if(physicalMemory.frameBlockTable[i].allocation_id == releasingPageBlock.allocation_id){
                        physicalMemory.buddyDeAllocate(process,physicalMemory.frameBlockTable[i]);

                        // this following if statement is required due to my implementation of page replacing algorithm.
                        // For basic lru and sampled lru algorithms, I maintained frameBlocks in frameBlock table only when frameBlock is present on physical memory.
                        // For clock algorithm I maintained all the allocated frameBlocks(whether it is present or not) in frameBlock table.
                        if(strcmp(pageReplaceOption.c_str(), "clock") != 0) {
                            physicalMemory.frameBlockTable.erase(physicalMemory.frameBlockTable.begin() + i);
                        }

                    }

                }
            }
        // deAllocate pageBlock from process` virtual memory
        process.virtualMemory.memoryRelease(page_id);

        }
        /// case Non memory instruction
        else if (process.currentInstruction.op_code == 3) {

        }
        /// Sleep Instruction
        else if (process.currentInstruction.op_code == 4) {

            // if this current sleep instruction is end of code, just end the execution without adding process into Sleep Queue of scheduler object.
            if(process.instructionTable.empty()){ return true; }

            // Set sleeping time and push current process into sleeping queue of scheduler object.
            process.sleepTime = process.currentInstruction.arg;
            scheduler.sleepList.push_back(process);

            // current(running, executing) process is blocked
            isRunningProcessBlocked = true;

        /// IO wait Instruction
        } else if (process.currentInstruction.op_code == 5) {
            // if this current IO wait instruction is end of code, just end the execution without adding process into IO wait Queue of scheduler object.
            if(process.instructionTable.empty()){ return true; }

            // put current process into IO wait queue of scheduler.
            scheduler.IOWaitList.push_back(process);
            isRunningProcessBlocked = true;
        }


        // if current executing process` priority is over 4, (which means it follows round robin scheduling algorithm)
        // then decrease time quantum value.
        if(process.priority > 4){
            process.timeQuantum -=1;
    }



    return isRunningProcessBlocked;
}

void Simulator::initReferenceInfo() {
    /// this method is only used when page replace Algorithm is "sampled LRU"
    /// called every 8 cycle of simulator, and initialize reference bits and renew reference bytes in every processes of scheduler object.

    // initialize running process`s reference info
    for (auto &pageBlock : scheduler.runningProcess.virtualMemory.pageBlockTable) {
        if (pageBlock.page_id != -2) {
            pageBlock.referenceByte.push_front(pageBlock.referenceBit);
            pageBlock.referenceByte.pop_back();
            pageBlock.referenceBit = 0;
            for(int i = pageBlock.start_idx; i <= pageBlock.end_idx; i++){
                scheduler.runningProcess.virtualMemory.memory[i].reference_bit = 0;
            }
        }
    }

    // initialize reference info of processes in running queue
    for (int j = 0; j < scheduler.runQueue.size(); j++) {
        for (int k = 0; k < scheduler.runQueue[j].size(); k++) {
            for (auto &pageBlock : scheduler.runQueue[j][k].virtualMemory.pageBlockTable) {
                if (pageBlock.page_id != -2) {
                    pageBlock.referenceByte.push_front(pageBlock.referenceBit);
                    pageBlock.referenceByte.pop_back();
                    pageBlock.referenceBit = 0;
                    for(int i = pageBlock.start_idx; i <= pageBlock.end_idx; i++){
                        scheduler.runQueue[j][k].virtualMemory.memory[i].reference_bit = 0;
                    }
                }
            }

        }
    }

    // initialize reference info of processes in IO wait queue
    if (!scheduler.IOWaitList.empty()) {
        for (int k = 0; k < scheduler.IOWaitList.size(); k++) {
            for (auto &pageBlock : scheduler.IOWaitList[k].virtualMemory.pageBlockTable) {
                if (pageBlock.page_id != -2) {
                    pageBlock.referenceByte.push_front(pageBlock.referenceBit);
                    pageBlock.referenceByte.pop_back();
                    pageBlock.referenceBit = 0;
                    for(int i = pageBlock.start_idx; i <= pageBlock.end_idx; i++){
                        scheduler.IOWaitList[k].virtualMemory.memory[i].reference_bit = 0;
                    }
                }
            }

        }
    }
    // initialize reference info of processes in sleep wait queue
    if (!scheduler.sleepList.empty()) {
        for (int k = 0; k < scheduler.sleepList.size(); k++) {
            for (auto &pageBlock : scheduler.sleepList[k].virtualMemory.pageBlockTable) {
                if (pageBlock.page_id != -2) {
                    pageBlock.referenceByte.push_front(pageBlock.referenceBit);
                    pageBlock.referenceByte.pop_back();
                    pageBlock.referenceBit = 0;
                    for(int i = pageBlock.start_idx; i <= pageBlock.end_idx; i++){
                        scheduler.sleepList[k].virtualMemory.memory[i].reference_bit = 0;
                    }
                }
            }

        }
    }

}

void Simulator::printCycle(bool isThereNewlyScheduled) {
    /// printing method


    Process running = scheduler.runningProcess;

    fprintf(sched, "[%d Cycle] Scheduled Process: ", cycle);
    if(isThereNewlyScheduled){
        fprintf(sched, "%d %s (priority %d)\n", running.pid, running.code_fileName.c_str(), running.priority);

    } else{
        fprintf(sched, "None\n");
    }
    // Line 2

    fprintf(sched, "Running Process: ");
    if (running.pid != -1) {
        fprintf(sched, "Process#%d(%d) running code %s line %d(op %d, arg %d)\n", running.pid, running.priority,
                running.code_fileName.c_str(), running.line, running.currentInstruction.op_code,
                running.currentInstruction.arg);
    }
    else {
        fprintf(sched, "None\n");
    }
    vector<deque<Process>> runQueue = scheduler.runQueue;
    for(int i =0; i < runQueue.size(); i++){
        fprintf(sched, "RunQueue %d: ", i);
        if(runQueue[i].empty()){
            fprintf(sched, "Empty");
        }
        else{
            while(!runQueue[i].empty()){
                Process p = runQueue[i].front();
                runQueue[i].pop_front();
                fprintf(sched, "%d(%s) ", p.pid, p.code_fileName.c_str());

            }
        }
        fprintf(sched, "\n");
    }
    // Line 4
    vector<Process> sleepList = scheduler.sleepList;
    fprintf(sched, "SleepList: ");
    if (sleepList.empty()) {
        fprintf(sched, "Empty");
    }
    else{
    while(!sleepList.empty()){
        Process p = sleepList.front();
        sleepList.erase(sleepList.begin());
        fprintf(sched, "%d(%s) ", p.pid, p.code_fileName.c_str() );
        }
    }
    fprintf(sched, "\n");

    // Line 5
    vector<Process> IOWaitList = scheduler.IOWaitList;
    fprintf(sched, "IOWait List: ");
    if (IOWaitList.empty()) {
        fprintf(sched, "Empty");
    }
    else{
    while(!IOWaitList.empty()){
        Process p = IOWaitList.front();
        IOWaitList.erase(IOWaitList.begin());
        fprintf(sched, "%d(%s) ", p.pid, p.code_fileName.c_str());
        }
    }
    fprintf(sched, "\n");
    fprintf(sched, "\n");



    if (running.pid != -1){
        if (running.currentInstruction.op_code == 0){
            fprintf(mem, "[%d Cycle] Input : Pid[%d] Function[%s] Page ID[%d] Page Num[%d]\n",
                    cycle, running.pid, "ALLOCATION", (running.virtualMemory.page_id_issued-1), running.currentInstruction.arg);
        }
        if (running.currentInstruction.op_code == 1){
            fprintf(mem, "[%d Cycle] Input : Pid[%d] Function[%s] Page ID[%d] Page Num[%d]\n",
                    cycle, running.pid, "ACCESS", running.currentInstruction.arg, running.virtualMemory.pageBlockTable[running.currentInstruction.arg].size);
        }
        if (running.currentInstruction.op_code == 2){
            fprintf(mem, "[%d Cycle] Input : Pid[%d] Function[%s] Page ID[%d] Page Num[%d]\n",
                    cycle, running.pid, "RELEASE", running.currentInstruction.arg, running.virtualMemory.pageBlockTable[running.currentInstruction.arg].size);
        }
        if (running.currentInstruction.op_code == 3){
            fprintf(mem, "[%d Cycle] Input : Pid[%d] Function[%s]\n", cycle, running.pid, "NON-MEMORY");
        }
        if (running.currentInstruction.op_code == 4){
            fprintf(mem, "[%d Cycle] Input : Pid[%d] Function[%s]\n",cycle,running.pid, "SLEEP" );
        }
        if (running.currentInstruction.op_code == 5){
            fprintf(mem, "[%d Cycle] Input : Pid[%d] Function[%s]\n", cycle,running.pid, "IOWAIT" );
        }

    }else { // NO-OP NEED TO LOOK BACK
        fprintf(mem, "[%d Cycle] Input : Function[NO-OP]\n", cycle);
    }

    // PM state
    fprintf(mem, "%-30s", ">> Physical Memory: ");
    // show PM state
    for (int i=0; i<numFrame; i++){
        if (i % 4 == 0){
            fprintf(mem, "|");
        }

        if (physicalMemory.memory[i].allocation_id == -1){
            fprintf(mem, "-");
        }else{
            fprintf(mem, "%d", physicalMemory.memory[i].allocation_id);
        }
    }
    fprintf(mem, "|\n");

    //line 2
    vector<Process> aliveProcesses = scheduler.getAliveProcesses();

    for(int i= 0; i < aliveProcesses.size(); i ++) {
        fprintf(mem, ">> pid(%d) %-20s", aliveProcesses[i].pid, "Page Table(PID): ");
        for (int j=0; j<numPage; j++){
            if (j % 4 == 0){
                fprintf(mem, "|");
            }
            int page_id = aliveProcesses[i].virtualMemory.memory[j].page_id;
            if (page_id== -1){
                fprintf(mem, "-");
            }else{
                fprintf(mem, "%d", page_id);
            }
        }
        fprintf(mem, "|\n");
        fprintf(mem, ">> pid(%d) %-20s", aliveProcesses[i].pid, "Page Table(AID): ");
        for (int j=0; j<numPage; j++){
            if (j % 4 == 0){
                fprintf(mem, "|");
            }
            int allocation_id = aliveProcesses[i].virtualMemory.memory[j].allocation_id;
            if (allocation_id== -1){
                fprintf(mem, "-");
            }else{
                fprintf(mem, "%d", allocation_id);
            }
        }
        fprintf(mem, "|\n");
        fprintf(mem, ">> pid(%d) %-20s", aliveProcesses[i].pid, "Page Table(VALID): ");
        for (int j=0; j<numPage; j++){
            if (j % 4 == 0){
                fprintf(mem, "|");
            }
            int valid_bit = aliveProcesses[i].virtualMemory.memory[j].valid_bit;
            if (valid_bit== -1){
                fprintf(mem, "-");
            }else{
                fprintf(mem, "%d", valid_bit);
            }
        }
        fprintf(mem, "|\n");
        fprintf(mem, ">> pid(%d) %-20s", aliveProcesses[i].pid, "Page Table(Ref): ");
        for (int j=0; j<numPage; j++){
            if (j % 4 == 0){
                fprintf(mem, "|");
            }
            int reference_bit = aliveProcesses[i].virtualMemory.memory[j].reference_bit;
            if (reference_bit== -1){
                fprintf(mem, "-");
            }else{
                fprintf(mem, "%d", reference_bit);
            }
        }
        fprintf(mem, "|\n");
    }
    fprintf(mem, "\n");


}

void Simulator::simulate() {
    /// trigger method for simulating system

    step();
    while(!checkTerminateCondition()){
        step();
    }
    fprintf(mem,"page fault = %d\n", numPageFault);
}

bool Simulator::checkTerminateCondition() {
    /// check whether the system should be terminated or not by checking scheduler`s queues
    /// if there are any processes left in shceduler`s queue, then return false
    /// if there are any events left in eventLog, then return false;
    /// returning false means keep running
    bool condition = true;

    for(int i =0; i< scheduler.runQueue.size(); i++){
        if(scheduler.runQueue[i].size() !=0){
            return false;
        }
    }

    if(!scheduler.IOWaitList.empty()){
        return false;
    }
    if(!scheduler.sleepList.empty()){
        return false;
    }
    if(scheduler.runningProcess.pid != -1){
        return false;
    }

    if(!eventLog.empty()){
        return false;
    }

    return condition;

}

void Simulator::closeProcess() {
    /// methods for terminating process which does not have any instructions to be executed any more.
    /// check if current process`s pageBlock is present in physical memory. If exists, deallocate them.
    /// I followed deallocating order stated in the spec of project3
    int AID= -11111;
    for(int i = 0; i< numPage; i++){

        // find pageBlock allocated in physical memory.
        if(scheduler.runningProcess.virtualMemory.memory[i].valid_bit == 1){

            AID = scheduler.runningProcess.virtualMemory.memory[i].allocation_id;

            // find corresponding frameBlock in physical memory
            for(int j = 0; j < physicalMemory.frameBlockTable.size(); j++){

                if(physicalMemory.frameBlockTable[j].allocation_id == AID) {

                    // deallocate it!
                    physicalMemory.buddyDeAllocate(scheduler.runningProcess, physicalMemory.frameBlockTable[j]);



                    // this following if statement is required due to my implementation of page replacing algorithm.
                    // For basic lru and sampled lru algorithms, I maintained frameBlocks in frameBlock table only when frameBlock is present on physical memory.
                    // For clock algorithm I maintained all the allocated frameBlocks(whether it is present or not) in frameBlock table.
                    if(strcmp(pageReplaceOption.c_str(), "clock") != 0) {
                        physicalMemory.frameBlockTable.erase(physicalMemory.frameBlockTable.begin() + j);
                    }

                }
            }

        }
    }

}





