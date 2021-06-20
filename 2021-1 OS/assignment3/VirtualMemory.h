
#ifndef VIRTUALMEMORY_H
#define VIRTUALMEMORY_H
#include<vector>
#include<deque>
#include<string>
#include <cstring>
using namespace std;

struct PageBlock{
    int page_id;
    int allocation_id;
    bool isValidated;
    int start_idx;
    int end_idx;
    int size;

    int referenceBit;
    deque<int> referenceByte;
};


struct Page{
    int page_id;
    int allocation_id;
    int valid_bit;
    int reference_bit;
    bool isAllocated;
};

class VirtualMemory {
public:
    int numPage;
    int page_id_issued;
    string pageReplaceOption;
    Page* memory;
    vector<PageBlock> pageBlockTable;
    VirtualMemory();
    VirtualMemory(int numPage, string pageReplaceOption);


    void memoryAllocate(int size);

    void memoryRelease(int page_id);
};


#endif
