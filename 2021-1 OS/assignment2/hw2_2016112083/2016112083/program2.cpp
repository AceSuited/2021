#include <iostream>
#include <time.h>
#include <sys/types.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include<unistd.h>
#include<vector>

#include <string>
#include <stdio.h>
#include <string.h>
#include <cmath>
using namespace std;

// global variables init

int N;                  // total num elements of array to sort.
int total_process_num;  
int * array;
int max_depth;          
int remain_divide;

long getnanosec(timespec start, timespec stop){ 
    long seconds = stop.tv_sec - start.tv_sec;
    long nanoseconds = stop.tv_nsec - start.tv_nsec;
    long output = seconds*(1000*1000*1000) + nanoseconds ;
    return output;
}


void merge(int *array, int first,  int last)
{
    // merge function
	int* temp = new int[last - first + 1];
	
	int mid = (first + last) / 2;
    int i, j, index;
    i = first;		// First array index
	j = mid + 1;	// Second array index
	index = 0;		// temporarily sorted array idx

	
    // compare each one element of first array and second array, then sort it to temp array
    // untill either one of them run out its all the elements.
    while (i <= mid && j <= last)
	{
		if (array[i] >= array[j]){
         temp[index++] = array[i++];
        }
		else{
            temp[index++] = array[j++];
        }
	}

    // if last array still has elements
	if (i > mid){
        for(int x = j; x <= last; x++){
            temp[index++] = array[x];
        }
    }
    // if first array still has elements.
	else{
		for(int x = i; x<=mid; x++ ){
            temp[index++] = array[x];
        }
    }
    
    // copy the sorted array 'temp', to original array
	for (i = first, index = 0; i <= last; i++, index++) {
        array[i] = temp[index];
    }

	delete[] temp;
}

void mergeSort(int *array, int first, int last, int depth)
{   
    // merge the results pieces of children processes(program1)
    
    int mid = (first + last) / 2;   

    // if splitted size reaches minimal size , and more pieces to split left,
    // do no further splits since this size of piece is already sorted by child process
    if(depth == max_depth && remain_divide > 1){
        remain_divide -=1;
        return;
    }

     // if splitted size reaches minimal size , and more pieces to split left,
     // do no further splits since this size of piece is already sorted by child process
    else if(remain_divide == 1){
        remain_divide -= 1;
        return;
    }



	if (first < last )
	{
        // stop split when all the piceces reauired for total_num_process are already exists.
        // which means, when it reaches the same size as size of program1 has sorted, do not split
        if(remain_divide >0){mergeSort(array, first, mid, depth+1);}
		if(remain_divide >0){mergeSort(array, mid + 1, last,depth+1);}
        
        // sorted pieces by program1 will merge here.
        merge(array, first, last);
		
	}
}
void parseInputs(int first, int last, int depth, vector<string> &input_lines, vector<int>&len_input)
{   
    
    int mid = (first + last) / 2;
    
    // if spliting reaches max_depth, save current splitted part as an input line for a child process.
    if(depth == max_depth && remain_divide > 1){
        string input = "";
        input += to_string(last - first + 1);
        input += " ";
        len_input.push_back(last - first + 1);
        for(int i= first; i <= last; i++){
            input+= to_string(array[i]);
            input += " ";
        }
        input_lines.push_back(input);
        remain_divide -=1;
        return;
        
    }
    // if spliting reaches max_depth, save current splitted part "PLUS" rest of picese part of array as an input line for a child process
    else if(remain_divide == 1){
       
        string input = "";
        input += to_string(N - first);
        input += " ";
        len_input.push_back(N - first);
        for(int i= first; i <= N-1; i++){
            input+= to_string(array[i]);
            input += " ";
        }
        input_lines.push_back(input);
        remain_divide -=1;
        return;
      
    }



	if (first < last )
	{
        // stop split when all the piceces reauired for total_num_process are already exists.
        if(remain_divide >0){parseInputs(first, mid, depth+1,input_lines, len_input);}
		if(remain_divide >0){parseInputs( mid + 1, last, depth+1, input_lines, len_input);}

	}
}


void multiProcessing(vector<string> input_lines, vector<int>len_input){
    
    // init variables
    pid_t pid_table[total_process_num];
    int P2C_pipe[total_process_num][2]; // parent to child pipe
    int C2P_pipe[total_process_num][2]; // child to parent pip[]


    // create pipes
    for (int i = 0; i < total_process_num; ++i){
        pipe(P2C_pipe[i]);
        pipe(C2P_pipe[i]);
    }
    
    // fork childeren proccesses // 
    pid_t pid;
    int num_forked = 0;
    do
    {
        pid = fork();
        // PARENT: count how many children it has
        if (pid > 0) {
            pid_table[num_forked] = pid;
            ++num_forked;
        }
        // CHILD: save its pid to pid_table.
        else {
            pid_table[num_forked] = (int)getpid();
        }
    }
    //PARENT: fork untill num_forked reach total_process_num
    while (pid > 0 && num_forked < total_process_num);


    ////////////////////////////////////////////////////
    // PARENT PROCESS///////////////////////////////////
    if (pid > 0){
        
         ////////////////////////////////////////////////////////////////////////////////////////////////
        // Send input lines to child processes (elements which program1 should sort)/////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////


        string input;
        int len;

        // loop for every childeren and send input lines through P2C pipe
        for (int i = 0; i < total_process_num; ++i)
        {   
            // get input line for this child
            input = input_lines[i];
            
            //close reading channel of P2C pipe. No use
            close(P2C_pipe[i][0]);
            
            //send input to child using P2C pipe
            write(P2C_pipe[i][1], input.c_str(), strlen(input.c_str()));
            
            //close writing channel of P2C pipe after writing.
            close(P2C_pipe[i][1]);
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////
        // wait for the children to finish //
        ////////////////////////////////////////////////////////////////////////////////////////////////
        
        int childrenFinished = 0;

        while (childrenFinished < total_process_num)
        {   
            
            int status;
            int childFinish = wait(&status);
            
            // count number of finished child
            ++childrenFinished;

        ////////////////////////////////////////////////////////////////////////////////////////////////
        // When a child finish Get Result from Pipe/////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////
            
            // find current finished child`s pid in pid_table and assign its index to childIndex
            int childIndex;
            for(int i =0; i < total_process_num; i++){
                if(pid_table[i] == childFinish){
                    childIndex = i;
                }
            }
            
            
            
            // variables for string processing
            input = input_lines[childIndex];
            len = len_input[childIndex];

            //get result lines of program1 execution for this child process
            
            // pipe control: close writing chaneel since we are not gonna write to child
            close(C2P_pipe[childIndex][1]);
            char output[strlen(input.c_str())];

            // read output lines
            read(C2P_pipe[childIndex][0], output, strlen(input.c_str()));
            
            // close after usage done
            close(C2P_pipe[childIndex][0]);

            // string processing  for output result.
            char *ptr = strtok(output, " ");
            int index = 0;
            for(int k=0; k < childIndex; k++){
                index += len_input[k];
            }
            

            // save results of program1 to parent process`s array. 
            for(int k= 0; k< len; k++){
                array[index++] = atoi(ptr);
                ptr = strtok(NULL, " ");
            }                     
        }
    }
    /// End of Parent Process (end of this function) ///
    ////////////////////////////////////////////////////

    ////////////////////////////////////////////////////
    //CHILD PROCESS/////////////////////////////////////
    if (pid == 0)
    {
        //find the current child`s index from pid_table
        int childPid = getpid();
        int childIndex;
        for(int i =0; i < total_process_num; i++){
                if(pid_table[i] == childPid){
                    childIndex = i;
                }
            }
    
        /////////////////////////////////////////////////////////////////////
        //wait until the parent finish writin input lines for processes /////
        /////////////////////////////////////////////////////////////////////

        
        // close pipe for other children processes
        fd_set rfds;
        int ready;
        for(int k=0;k<total_process_num;k++){
            if(k!=childIndex){
                close(P2C_pipe[k][0]);
                close(P2C_pipe[k][1]);
                close(C2P_pipe[k][0]);
                close(C2P_pipe[k][1]);
            }
        }


        while(true)
        {   
            // initialize fd_set
            FD_ZERO(&rfds);
            // put Parent to child pipe`s reading channel to FD_SET
            FD_SET(P2C_pipe[childIndex][0], &rfds);

            // wait untill pipe`s status change
            ready = select(P2C_pipe[childIndex][0] + 1, &rfds, NULL, NULL, NULL);

            // when inputs from parent are ready...
            if (ready > 0){
                // std::cerr<<"ready"<<std::endl;

                //close the writing channel of the parent-child pipe
                close(P2C_pipe[childIndex][1]);
                
                //redirect input to stdin
                dup2(P2C_pipe[childIndex][0], 0);
                close(P2C_pipe[childIndex][0]);

                //cclose the reading channel of the parent-child pipe
                close(C2P_pipe[childIndex][0]);
                //redirect output from stdout
                dup2(C2P_pipe[childIndex][1], 1);
                
                // close after usage
                close(C2P_pipe[childIndex][1]);

                //EXECUTE PROGRAM 1!!! ///
                execlp("./program1","./program1", NULL);

                //hope these under lines should be unreachable
                cerr<<"error opening"<<endl;
                exit(1);
            }
        }
    }
    // END OF CHILD PROCESS //
    //////////////////////////
}


int main(int argc, char **argv) {

    // inits
    cin >> N;
    array = new int[N];
    total_process_num = stoi(argv[1]);

    remain_divide = total_process_num;
        if(total_process_num == 1){
            max_depth = -1;
            remain_divide = 2;
            mergeSort(array,0,N-1,1);
        }
    max_depth = 1;

    // max depth to determine wherer to stop splitting tree.
    while(pow(2,max_depth) < total_process_num){
        max_depth += 1;
    }
    // get input
	for (int i = 0; i < N; i++){
		cin >> array[i];
    }

    //inits again
    int part_size = N / total_process_num;
    int index= part_size;
    int prev =0;

    string input ="";
    vector<string> input_lines;  // input line for each child processes (program1)
    vector<int> len_input;       // input length for each child processes (program1)
   
    
    // parse array into several pieces and save the information of pieces to input_lines and len_input.
    parseInputs(0, N-1, 0, input_lines, len_input);
    

    // measure time  
    timespec start;
    clock_gettime(CLOCK_MONOTONIC,&start);


    // sort by multi processing!
    multiProcessing(input_lines, len_input);

    
    // merge the pieces of sorted arrays from children program1
    remain_divide = total_process_num;
    mergeSort(array, 0,N-1,0);

    // measure time
    timespec stop;
    clock_gettime(CLOCK_MONOTONIC,&stop);
    long total_time_ms = getnanosec(start, stop)/ (1000 *1000);
	

	
    // print result.
    for(int i = 0; i < N; i++){
        cout << array[i] << " ";
    }
    cout << "\n" << total_time_ms;


    return 0;
}
