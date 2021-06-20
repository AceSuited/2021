


#include "PhysicalMemory.h"


// constructor
PhysicalMemory::PhysicalMemory() {}

PhysicalMemory::PhysicalMemory(int numFrame , string pageReplaceOption) {
    /// constructor
    this->numFrame = numFrame;
    this->pageReplaceOption = pageReplaceOption;
    memory = new Frame[numFrame];
    allocation_id_issued = 0;

    // for "clock" page replacement algorithm
    clockSavePoint =  -1;

    // initialize physical memory
    for(int i = 0; i< numFrame; i++){
        memory[i].allocation_id= -1;

    }

    FrameBlock firstEmptyBlock;
    firstEmptyBlock.start_idx = 0;
    firstEmptyBlock.end_idx = numFrame -1;
    firstEmptyBlock.size = numFrame;
    firstEmptyBlock.allocation_id = -1;
    firstEmptyBlock.mappedProcessId = -1;
    firstEmptyBlock.reference_bit = 0;
    firstEmptyBlock.is_valid = false;


    // initialize free block list for buddy system
    num_freeBlocksList = ceil(log(numFrame) / log(2)) + 1;
    freeBlocks = new vector<FrameBlock>[num_freeBlocksList];
    freeBlocks[num_freeBlocksList - 1].push_back(firstEmptyBlock);

    // frame block table
    frameBlockTable = deque<FrameBlock>();

}

/// comparator for sorting algorithm
bool comparator(FrameBlock a, FrameBlock b) {
    if(a.allocation_id < b.allocation_id){
        return true;
    }
    else{
        return false;
    }

}

FrameBlock PhysicalMemory::buddyAllocate(Process& requestingProcess, PageBlock& accessingPageBlock, Scheduler &scheduler) {
/// buddy allocation method.
/// 1. check free spaces of exact size, if exists allocate it
/// 2. if exact size does not exist, check bigger free spaces, if bigger free space exists, split it until it matches the size
/// 3. if neither exact size and bigger size free space exist, replace existing allocated frameBlock from physical memory.

    // check the size of requesting accessing pageBlock
    int size = accessingPageBlock.size;

    // check corresponding buddy size to allocate.
    int exp = ceil(log(size) / log(2));

    FrameBlock allocated;
    allocated.start_idx = -100;

    /// save mapped process` pid.
    allocated.mappedProcessId = requestingProcess.pid;

    // flag for representing if allocating is done.
    bool done = false;


    while(!done) {
        /// Free Block with exact size Exists.
        if(!freeBlocks[exp].empty()){

            // erase the free block from freeBlocklist
            allocated = freeBlocks[exp].front();
            freeBlocks[exp].erase(freeBlocks[exp].begin());

            // if requested PageBlock is first time to be allocated,
            // assign new allocation id
            if(accessingPageBlock.allocation_id == -1) {
                allocated.allocation_id = allocation_id_issued;
                for (int i = allocated.start_idx; i <= allocated.end_idx; i++) {
                    memory[i].allocation_id = allocation_id_issued;
                }

                allocation_id_issued++;
            }
            // if requested PageBlock has already been allocated before
            // (if it already has allocation id), do not assign new allocation id. just use allocation id issued in past.
            else{

                allocated.allocation_id = accessingPageBlock.allocation_id;
                for (int i = allocated.start_idx; i <= allocated.end_idx; i++) {
                    memory[i].allocation_id = allocated.allocation_id;
                }
                // Only for "clock page replacement algorithm"
                // because FrameBlock table`s each index represents allocation id, there must be no duplicate frameBlocks with same allocation id.
                // so, frameBlock which had been allocated before, must be deleted from frameBlockTable.
                if(strcmp(pageReplaceOption.c_str(), "clock") == 0){
                    sort(frameBlockTable.begin(), frameBlockTable.end(), comparator);
                    frameBlockTable.erase(frameBlockTable.begin() + allocated.allocation_id);
                }
            }


            allocated.mappedProcessId = requestingProcess.pid;

            // add additional values to newly allocated frameBlock, in case of "clock page replacement algorithm"
            if(strcmp(pageReplaceOption.c_str(), "clock") == 0){
                allocated.reference_bit = 1;
                allocated.is_valid = true;

            }
            // assign newly allocated FrameBlock to frameBlockTable.
            frameBlockTable.push_front(allocated);

            // allocation done! escape the while loop by changing it to true
            done = true;

        }
        /// if free block with exact same size does not exists, search for larger Free Block
        else{
            // flag for representing larger free block found
            bool foundLargerFreeBlock = false;
            int i;
            // iterate freeBlock list over until larger free block found
            for (i = exp + 1; i < num_freeBlocksList; i++) {
                if (!freeBlocks[i].empty()) {
                    foundLargerFreeBlock = true;
                    break;
                }
            }

            // if larger free block found,
            if (foundLargerFreeBlock) {

                // erase founded large free block from freeblocklist
                FrameBlock freeLargeBlock;
                freeLargeBlock = freeBlocks[i].front();
                freeBlocks[i].erase(freeBlocks[i].begin());
                i--;

                // split it until it reaches requiring buddy size
                for (; i >= exp; i--) {

                    // split
                    FrameBlock fb1, fb2;
                    fb1.start_idx = freeLargeBlock.start_idx;
                    fb1.end_idx = freeLargeBlock.start_idx + (freeLargeBlock.end_idx - freeLargeBlock.start_idx) / 2;
                    fb1.size = fb1.end_idx - fb1.start_idx + 1;

                    fb2.start_idx = freeLargeBlock.start_idx + (freeLargeBlock.end_idx - freeLargeBlock.start_idx + 1) / 2;
                    fb2.end_idx = freeLargeBlock.end_idx;
                    fb2.size = fb2.end_idx - fb2.start_idx + 1;
                    freeBlocks[i].push_back(fb1);
                    freeBlocks[i].push_back(fb2);

                    freeLargeBlock = freeBlocks[i].front();
                    freeBlocks[i].erase(freeBlocks[i].begin());
                }

                // allocated free block
                allocated = freeLargeBlock;
                allocated.size = allocated.end_idx - allocated.start_idx + 1;

                // if requested PageBlock is first time to be allocated,
                // assign new allocation id
                if(accessingPageBlock.allocation_id == -1) {

                    allocated.allocation_id = allocation_id_issued;
                    for (int j = allocated.start_idx; j <= allocated.end_idx; j++) {
                        memory[j].allocation_id = allocation_id_issued;
                    }
                    allocation_id_issued++;
                }
                // if requested PageBlock has already been allocated before
                // (if it already has allocation id), do not assign new allocation id. just use allocation id issued in past.
                else{

                    allocated.allocation_id = accessingPageBlock.allocation_id;
                    for (int j = allocated.start_idx; j <= allocated.end_idx; j++) {
                        memory[j].allocation_id = allocated.allocation_id;
                    }

                    // Only for "clock page replacement algorithm"
                    // because FrameBlock table`s each index represents allocation id, there must be no duplicate frameBlocks with same allocation id.
                    // so, frameBlock which had been allocated before, must be deleted from frameBlockTable.
                    if(strcmp(pageReplaceOption.c_str(), "clock") == 0){
                        sort(frameBlockTable.begin(), frameBlockTable.end(), comparator);
                        frameBlockTable.erase(frameBlockTable.begin() + allocated.allocation_id);
                    }

                }
                // add additional values to newly allocated frameBlock, in case of "clock page replacement algorithm"
                allocated.mappedProcessId = requestingProcess.pid;
                if(strcmp(pageReplaceOption.c_str(), "clock") == 0){
                    allocated.reference_bit = 1;
                    allocated.is_valid = true;

                }
                // add additional values to newly allocated frameBlock, in case of "clock page replacement algorithm"
                frameBlockTable.push_front(allocated);

                // allocation done! escape the while loop by changing it to true
                done = true;


            }
            /// Page Replacing
            /// If there are no free space in physical memory to allocate, replace frameBlocks already allocated.
            else {

                /// case lru option
                if (strcmp(pageReplaceOption.c_str(), "lru") == 0) {

                    // the frameBlockTable keep its frameBlock table`s order in which least accessed frameBlock is positioned at the end.
                    // so just victimize last frameBlock in frameBlock table
                    FrameBlock victim = frameBlockTable.back();

                    /// find process which has pageBlock mapped to the victim frameBlock in scheduler

                    // if running process is the owner of victim frameBlock,
                    if(victim.mappedProcessId == requestingProcess.pid){
                        buddyDeAllocate(requestingProcess, victim);
                    }
                    // if process in running queue is the owner of victim frameBlock
                    else{
                        for(int j =0; j< scheduler.runQueue.size(); j++){

                            for (int k = 0; k < scheduler.runQueue[j].size(); k++) {
                                if(scheduler.runQueue[j][k].pid == victim.mappedProcessId){
                                    buddyDeAllocate(scheduler.runQueue[j][k], victim);
                                    break;
                                }
                            }
                        }
                        // if process in IO wait List is the owner of victim frameBlock
                        if(!scheduler.IOWaitList.empty()){
                            for(int j = 0; j < scheduler.IOWaitList.size(); j ++){
                                if(scheduler.IOWaitList[j].pid == victim.mappedProcessId){
                                    buddyDeAllocate(scheduler.IOWaitList[j], victim);
                                }
                            }
                        }
                        // if process in sleep List is the owner of victim frameBlock
                        if(!scheduler.sleepList.empty()){
                            for(int j = 0; j < scheduler.sleepList.size(); j ++){
                                if(scheduler.sleepList[j].pid == victim.mappedProcessId){
                                    buddyDeAllocate(scheduler.sleepList[j], victim);
                                }
                            }

                        }

                    }
                    // delete victim frame block from frameBlock table.
                    frameBlockTable.pop_back();

                }
                /// case sampled option
                if (strcmp(pageReplaceOption.c_str(), "sampled") == 0) {

                    // sort the frameBlockTable
                    // I sorted the Table because if there are tie situation among frameBlocks, frame block with smaller allocation id must be victimized
                    sort(frameBlockTable.begin(), frameBlockTable.end(), comparator);

                    // decimal value of each candidate frameBlock`s reference byte will be saved here
                    int referByteValues[frameBlockTable.size()];


                    // find mapped process with candidate frameBlock.
                    // if mapped process found, find mapped pageBlock with candidate frameBlock.
                    // then calculate decimal reference value/

                    // for every frame blocks allocated in physicial memory,
                    for(int x = 0; x < frameBlockTable.size(); x++) {

                        // if mapped with running process,
                        for (auto &pageBlock : requestingProcess.virtualMemory.pageBlockTable) {
                            if (pageBlock.allocation_id == frameBlockTable[x].allocation_id) {
                                referByteValues[x] = calculateReferValue(pageBlock.referenceByte);
                                break;
                            }
                        }

                        // if mapped with process in running queue
                        for (int j = 0; j < scheduler.runQueue.size(); j++) {
                            for (int k = 0; k < scheduler.runQueue[j].size(); k++) {
                                if (scheduler.runQueue[j][k].pid == frameBlockTable[x].mappedProcessId) {
                                    for (auto &pageBlock : scheduler.runQueue[j][k].virtualMemory.pageBlockTable) {
                                        if (pageBlock.allocation_id == frameBlockTable[x].allocation_id) {
                                            referByteValues[x] = calculateReferValue(pageBlock.referenceByte);
                                            break;
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                        // if mapped with process in IO wait List
                        if(!scheduler.IOWaitList.empty()){
                            for(int j = 0; j < scheduler.IOWaitList.size(); j ++){
                                if (scheduler.IOWaitList[j].pid == frameBlockTable[x].mappedProcessId) {
                                    for (auto &pageBlock : scheduler.IOWaitList[j].virtualMemory.pageBlockTable) {
                                        if (pageBlock.allocation_id == frameBlockTable[x].allocation_id) {
                                            referByteValues[x] = calculateReferValue(pageBlock.referenceByte);
                                            break;
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                        // if mapped with process in sleep List
                        if(!scheduler.sleepList.empty()){
                            for(int j = 0; j < scheduler.sleepList.size(); j ++){
                                if (scheduler.sleepList[j].pid == frameBlockTable[x].mappedProcessId) {
                                    for (auto &pageBlock : scheduler.sleepList[j].virtualMemory.pageBlockTable) {
                                        if (pageBlock.allocation_id == frameBlockTable[x].allocation_id) {
                                            referByteValues[x] = calculateReferValue(pageBlock.referenceByte);
                                            break;
                                        }
                                    }
                                    break;
                                }
                            }

                        }

                    }

                    // compare reference byte values and decide victim
                    int min = referByteValues[0];
                    int index = 0;

                    for(int j =0; j <frameBlockTable.size(); j++){
                        if(referByteValues[j] <  min){
                            min = referByteValues[j];
                            index = j;
                        }
                    }

                    // victim elected
                    FrameBlock victim = frameBlockTable[index];

                    // again, find mapped process with the elected victim
                    if(victim.mappedProcessId == requestingProcess.pid){
                        buddyDeAllocate(requestingProcess, victim);
                    }
                    else{
                        for(int j =0; j< scheduler.runQueue.size(); j++){

                            for (int k = 0; k < scheduler.runQueue[j].size(); k++) {
                                if(scheduler.runQueue[j][k].pid == victim.mappedProcessId){
                                    buddyDeAllocate(scheduler.runQueue[j][k], victim);
                                    break;
                                }
                            }
                        }

                        if(!scheduler.IOWaitList.empty()){
                            for(int j = 0; j < scheduler.IOWaitList.size(); j ++){
                                if(scheduler.IOWaitList[j].pid == victim.mappedProcessId){
                                    buddyDeAllocate(scheduler.IOWaitList[j], victim);
                                    break;
                                }
                            }
                        }
                        if(!scheduler.sleepList.empty()){
                            for(int j = 0; j < scheduler.sleepList.size(); j ++){
                                if(scheduler.sleepList[j].pid == victim.mappedProcessId){
                                    buddyDeAllocate(scheduler.sleepList[j], victim);
                                    break;
                                }
                            }

                        }

                    }
                    // erase from frameBlockTable after deAllocation is done
                    frameBlockTable.erase(frameBlockTable.begin() + index);

                }
                /// case Clock Algorithm
                if (strcmp(pageReplaceOption.c_str(), "clock") == 0) {

                    // sort the frameBlockTable for clock pointer
                    sort(frameBlockTable.begin(), frameBlockTable.end(),comparator);

                    // get clock pointer from last cycle
                    clock = clockSavePoint;

                    // configure clock pointer to  start in this cycle
                    // if clock pointer reaches end of the frameBlock list, turn the pointer back to 0, otherwise +1
                    if(clock + 1 <= frameBlockTable.size() - 1){
                        clock = clock + 1;
                    }
                    else{
                        clock = 0;
                    }

                    

                    FrameBlock victim;


                    // iterate frameBlockTable until frameBlock with reference bit 0 is found
                    // if frameBlock with reference bit 1 found, change reference bit to 0.(give second chance)
                    while (true){

                        // if frameBlock is not allocated(present) in physical memory, just continue
                        if(!frameBlockTable[clock].is_valid){
                            // move clock pointer
                            if(clock == frameBlockTable.size() -1){
                                clock = 0;
                            }
                            else{
                                clock++;
                            }
                            continue;
                        }

                        // if reference bit is 0, victim found! save clock pointer and escape the loop
                        if(frameBlockTable[clock].reference_bit == 0){
                            victim = frameBlockTable[clock];
                            clockSavePoint = clock;
                            break;
                        }
                        // if reference bit is 1, turn the bit back to 0, and find mapped process, mapped pageblock and change its reference bit to 0.
                        else{
                            // if frame block is mapped to running process
                            frameBlockTable[clock].reference_bit = 0;
                            if(frameBlockTable[clock].mappedProcessId == scheduler.runningProcess.pid) {
                                for (auto &pageBlock : scheduler.runningProcess.virtualMemory.pageBlockTable) {
                                    if (pageBlock.allocation_id == frameBlockTable[clock].allocation_id) {
                                        for (int k = pageBlock.start_idx; k <= pageBlock.end_idx; k++) {
                                            requestingProcess.virtualMemory.memory[k].reference_bit = 0;
                                        }
                                        break;
                                    }
                                }
                            }

                            // if frame block is mapped to process in running Queue
                            for (int j = 0; j < scheduler.runQueue.size(); j++) {
                                for (int k = 0; k < scheduler.runQueue[j].size(); k++) {
                                    if (scheduler.runQueue[j][k].pid == frameBlockTable[clock].mappedProcessId) {
                                        for (auto &pageBlock : scheduler.runQueue[j][k].virtualMemory.pageBlockTable ) {
                                            if (pageBlock.allocation_id == frameBlockTable[clock].allocation_id ) {
                                                for(int p = pageBlock.start_idx; p <= pageBlock.end_idx; p++){
                                                    scheduler.runQueue[j][k].virtualMemory.memory[p].reference_bit = 0;
                                                }
                                                break;
                                            }
                                        }
                                        break;
                                    }
                                }
                            }
                            // if frame block is mapped to process in IO wait list
                            if(!scheduler.IOWaitList.empty()){
                                for(int j = 0; j < scheduler.IOWaitList.size(); j ++){
                                    if (scheduler.IOWaitList[j].pid == frameBlockTable[clock].mappedProcessId) {
                                        for (auto &pageBlock : scheduler.IOWaitList[j].virtualMemory.pageBlockTable) {
                                            if (pageBlock.allocation_id == frameBlockTable[clock].allocation_id) {
                                                for(int k = pageBlock.start_idx; k <= pageBlock.end_idx; k++){
                                                    scheduler.IOWaitList[j].virtualMemory.memory[k].reference_bit = 0;
                                                }
                                                break;
                                            }
                                        }
                                        break;
                                    }
                                }
                            }

                            // if frame block is mapped to process in sleep list
                            if(!scheduler.sleepList.empty()){
                                for(int j = 0; j < scheduler.sleepList.size(); j ++){
                                    if (scheduler.sleepList[j].pid == frameBlockTable[clock].mappedProcessId) {
                                        for (auto &pageBlock : scheduler.sleepList[j].virtualMemory.pageBlockTable) {
                                            if (pageBlock.allocation_id == frameBlockTable[clock].allocation_id) {
                                                for(int k = pageBlock.start_idx; k <= pageBlock.end_idx; k++){
                                                    scheduler.sleepList[j].virtualMemory.memory[k].reference_bit = 0;
                                                }
                                                break;
                                            }
                                        }
                                        break;
                                    }
                                }

                            }

                        }

                        // move clock pointer
                        if(clock == frameBlockTable.size() -1){
                            clock = 0;
                        }
                        else{
                            clock++;
                        }
                    }


                    // now, since victim found, find again mapped process with the victim and do the de allocation operation
                    if(victim.mappedProcessId == requestingProcess.pid){
                        buddyDeAllocate(requestingProcess, victim);
                    }
                    else{
                        for(int j =0; j< scheduler.runQueue.size(); j++){

                            for (int k = 0; k < scheduler.runQueue[j].size(); k++) {
                                if(scheduler.runQueue[j][k].pid == victim.mappedProcessId){
                                    buddyDeAllocate(scheduler.runQueue[j][k], victim);
                                    break;
                                }
                            }
                        }

                        if(!scheduler.IOWaitList.empty()){
                            for(int j = 0; j < scheduler.IOWaitList.size(); j ++){
                                if(scheduler.IOWaitList[j].pid == victim.mappedProcessId){
                                    buddyDeAllocate(scheduler.IOWaitList[j], victim);
                                    break;
                                }
                            }
                        }
                        if(!scheduler.sleepList.empty()){
                            for(int j = 0; j < scheduler.sleepList.size(); j ++){
                                if(scheduler.sleepList[j].pid == victim.mappedProcessId){
                                    buddyDeAllocate(scheduler.sleepList[j], victim);
                                    break;
                                }
                            }

                        }

                    }



                }


            }
        }
    }
    return allocated;
}

void PhysicalMemory::buddyDeAllocate(Process& requestingProcess, FrameBlock victim) {
    /// DeAllocating method
    /// requires victim frameBlock and a Process which mapped to the victim
    int i;
    int buddyNumber, buddyAddress;

    // victim(will be return its space to freeBlockList)`s buddy size
    int exp = ceil(log(victim.size) / log(2));

    // return the space to freeBlock List
    freeBlocks[exp].push_back(victim);


    // buddy number and buddy address. These values are for finding victim`s buddy in free Block list(if exists)
    // buddy number
    buddyNumber = victim.start_idx / victim.size;

    // buddy address
    if(buddyNumber % 2 !=0){
        buddyAddress = victim.start_idx - pow(2, exp);
    }
    else{
        buddyAddress = victim.start_idx + pow(2, exp);
    }

    // search in free Block list to find its buddy
    for( i = 0; i < freeBlocks[exp].size(); i++){

        // if buddy found and is also free
        if(freeBlocks[exp][i].start_idx == buddyAddress){

            // merge two buddies to one larger free block
            FrameBlock mergedBlock;
            if(buddyNumber % 2 == 0){

                mergedBlock.start_idx = victim.start_idx;
                mergedBlock.end_idx = victim.start_idx + 2 * pow(2, exp) - 1;
                mergedBlock.size = mergedBlock.end_idx - mergedBlock.start_idx + 1;
                mergedBlock.allocation_id = -1;
                buddyDeAllocate(requestingProcess, mergedBlock);
            }else{
                mergedBlock.start_idx = buddyAddress;
                mergedBlock.end_idx = buddyAddress + 2 * pow(2, exp) - 1;
                mergedBlock.size = mergedBlock.end_idx - mergedBlock.start_idx + 1;
                mergedBlock.allocation_id = -1;
                buddyDeAllocate(requestingProcess, mergedBlock);
            }
            freeBlocks[exp].erase(freeBlocks[exp].begin() + i);
            freeBlocks[exp].erase(freeBlocks[exp].begin() + freeBlocks[exp].size() - 1);
            break;
        }
    }


    // find mapped PageBlock to the victim
    // then reflect deAllocation results to the mapped Page block
    for(int j = 0; j <requestingProcess.virtualMemory.pageBlockTable.size(); j++){
        if(victim.allocation_id == requestingProcess.virtualMemory.pageBlockTable[j].allocation_id){

            // mapped pageBlock found,
            PageBlock& inValidatedPageBlock = requestingProcess.virtualMemory.pageBlockTable[j];
            inValidatedPageBlock.isValidated = false;
            // for sampled algorithm, reference bit to 0
            if(strcmp(pageReplaceOption.c_str(), "sampled") == 0) {
                inValidatedPageBlock.referenceBit = 0;
            }
            // valid bit to 0
            for(int k = inValidatedPageBlock.start_idx; k <= inValidatedPageBlock.end_idx; k++){
                requestingProcess.virtualMemory.memory[k].valid_bit = 0;

                // for sampled algorithm, reference bit to 0
                if(strcmp(pageReplaceOption.c_str(), "sampled") == 0){
                    requestingProcess.virtualMemory.memory[k].reference_bit = 0;
                }
            }
        }
    }

    // reflect the deAllocation results to physical memory
    for (int j = victim.start_idx; j <= victim.end_idx; j++) {

        memory[j].allocation_id = -1;

    }

    // if page replace option is "clock", add additional info to victim frameBlock
    if(strcmp(pageReplaceOption.c_str(), "clock") == 0 && victim.allocation_id != -1){
        frameBlockTable[victim.allocation_id].is_valid = false;
    }



}

int PhysicalMemory::calculateReferValue(deque<int> referenceByte) {
    /// helper method for "sampled page replacement algorithm"
    /// calculate decimal value of reference byte
    int i = 7;
    int referValue = 0;
    while (i >= 0){
        referValue += referenceByte[i] * pow(2,abs(i-7));
        i--;
    }
    return referValue;

}


