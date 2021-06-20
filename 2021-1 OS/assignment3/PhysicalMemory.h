#ifndef PHYSICALMEMORY_H
#define PHYSICALMEMORY_H

#include <vector>
#include <cmath>
#include <deque>
#include <cstring>
#include <iostream>
#include <algorithm>
#include "Process.h"
#include "Scheduler.h"

using namespace std;

struct FrameBlock{
    int mappedProcessId;
    int start_idx;
    int end_idx;
    int size;
    int allocation_id;
    int reference_bit;
    bool is_valid;
};


struct Frame{
    int allocation_id;

};

class PhysicalMemory {
public:
    int numFrame;
    string pageReplaceOption;
    int allocation_id_issued;
    int num_freeBlocksList;

    int clock;
    int clockSavePoint;

    vector<FrameBlock> * freeBlocks;
    deque<FrameBlock> frameBlockTable;



    Frame* memory;
    PhysicalMemory();
    PhysicalMemory(int numFrame, string pageReplaceOption);
    FrameBlock buddyAllocate(Process& requestingProcess, PageBlock &accessingPageBlock, Scheduler &scheduler);
    void buddyDeAllocate(Process& requestingProcess, FrameBlock victim);
    int calculateReferValue(deque<int> referenceByte);

};


#endif
