
#include <string>
#include <cstring>
#include <cstdio>
#include <queue>
#include "Process.h"
#include "Simulator.h"
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
using namespace std;


extern string directory = "";
int main(int argc, char* argv[ ] ){


    // get options input
    string pageReplaceOption = "";
    int eventNum, VM_size, PM_size, page_size;
    if(argc != 1){
        for(int i = 1; i < argc; i++) {
            string type;
            string value;
            type = strtok(argv[i], "=");
            value = strtok(nullptr, "=");
            if(strcmp(type.c_str(), "-dir") == 0){
                directory = value;
            }else if (strcmp(type.c_str(), "-page") == 0){
                pageReplaceOption = value;
            }
        }
    }
    char pathname[1024];
    getcwd(pathname, sizeof(pathname));
    if(strcmp(pageReplaceOption.c_str(), "") == 0){
        pageReplaceOption = "lru";
    }
    if(strcmp(directory.c_str(), "") == 0){
        directory = pathname;
    }

    // read inputFile
    FILE *inputFile = fopen((directory + "/input").c_str(), "r");
    fscanf (inputFile, "%d\t%d\t%d\t%d\n", &eventNum, &VM_size, &PM_size, &page_size);

    // save input file`s contents(events) into a queue
    // this queue with events will passed to simulator object
    queue<Event> eventLog;
    for(int i=0; i<eventNum; i++){
        int time;
        char temp[40];
        int priority;
        fscanf(inputFile, "%d\t%s\t%d\n", &time, temp, &priority );
        Event event;

        if(strcmp(temp, "INPUT") == 0){
            event.time = time;
            event.IO_flag = true;
            event.code_fileName= "";
            event.pid = priority;
        }else{
            event.time = time;
            event.code_fileName  = string(temp);
            event.priority = priority;
            event.IO_flag = false;
        }
        eventLog.push(event);
    }


//     create simulator object and do simulation
    Simulator simulator(VM_size, PM_size, page_size, eventLog, pageReplaceOption);
    simulator.simulate();

//    srand(time(NULL));
//
//    for (int i = 0; i < 1000; i++) {
//        printf("1 %d\n", rand() % 32);
//    }

}