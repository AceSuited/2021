//
// Created by sodap on 2021-05-27.
//


#include "VirtualMemory.h"


// constructor
VirtualMemory::VirtualMemory() {
}


/// constructor
VirtualMemory::VirtualMemory(int numPage, string pageReplaceOption) {
    this->numPage = numPage;
    memory = new Page[numPage];
    page_id_issued = 0;
    this->pageReplaceOption = pageReplaceOption;
    for(int i = 0; i< numPage; i++){
        memory[i].isAllocated= false;
        memory[i].page_id = -1;
        memory[i].allocation_id = -1;
        memory[i].reference_bit = -1;
        memory[i].valid_bit = -1;
    }
}

void VirtualMemory::memoryAllocate(int size) {
    /// memory allocation method.
    /// it is an operation implementation of instruction opcode 0


    int start = -999999;
    // find empty spaces from remaining virtual memory
    for(int  i= 0; i < numPage; i++){
        if(memory[i].page_id == -1){
            start = i;
            break;
        }
    }

    // allocate memory, sign it into pageTable
    for(int i = start; i < start + size; i++){

        memory[i].page_id = page_id_issued;
        memory[i].valid_bit = 0;

        // if page replacement algorithm is "sampled" or "clock", then assign reference bit to the page table also.
        if(strcmp(pageReplaceOption.c_str(), "sampled") == 0 || strcmp(pageReplaceOption.c_str(), "clock") == 0 ){
            memory[i].reference_bit  = 0;
        }
    }

    // Create PageBlock
    PageBlock pageBlock;
    pageBlock.start_idx = start;
    pageBlock.end_idx = start + size - 1;
    pageBlock.size = size;
    pageBlock.page_id = page_id_issued;
    pageBlock.isValidated = false;
    pageBlock.allocation_id = -1;

    // if page replacement algorithm is "sampled" assign referenece byte data structure to the pageBlock
    if(strcmp(pageReplaceOption.c_str(), "sampled") == 0) {
        for (int i = 0; i < 8; i++) { pageBlock.referenceByte.push_back(0); }
        pageBlock.referenceBit = 0;
    }

    // add pageBlock table
    pageBlockTable.push_back(pageBlock);
    page_id_issued ++;
}

void VirtualMemory::memoryRelease(int page_id) {
    /// memory Release method.
    /// it is an operation implementation of instruction opcode 2


    // find requested pageBlocck and DeAlloacate memory space from it
    PageBlock &releasingPageBlock = pageBlockTable[page_id];
    for(int i = releasingPageBlock.start_idx; i <= releasingPageBlock.end_idx; i++){
        memory[i].allocation_id = -1;
        memory[i].valid_bit = -1;
        memory[i].page_id = -1;
        memory[i].isAllocated = false;
        memory[i].reference_bit = -1;
    }
    releasingPageBlock.allocation_id = -2;
    releasingPageBlock.isValidated = false;
    releasingPageBlock.end_idx = -2;
    releasingPageBlock.start_idx = -2;
    releasingPageBlock.page_id = -2;

    if(strcmp(pageReplaceOption.c_str(), "sampled") == 0){
        releasingPageBlock.referenceBit = -1;
    }
    

}




