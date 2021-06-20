#include <iostream>
#include <time.h>
#include <cmath>
#include <vector>
#include <pthread.h>
#include <unistd.h>


using namespace std;
int N;
int total_thread_num;
int remain_divide;
int max_depth;
int* array;
int arg_arr_index;
unsigned int microseconds;



// structure for thread args
struct ARG{
    int first;
    int last;
    int depth;
};

long getnanosec(timespec start, timespec stop){ 
    long seconds = stop.tv_sec - start.tv_sec;
    long nanoseconds = stop.tv_nsec - start.tv_nsec;
    long output = seconds*(1000*1000*1000) + nanoseconds ;
    return output;
}


void merge(int first,  int last)
{
    // merge function
	// int* temp = new int[last - first + 1];
    int temp[1000000];

	
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



}


void mergeSort(int first, int last, int depth)
{   
    // merge the results pieces of threads
    
    int mid = (first + last) / 2;   

    // if splitted size reaches minimal size , and more pieces to split left,
    // do no further splits since this size of piece is already sorted by threads
    if(depth == max_depth && remain_divide > 1){
        remain_divide -=1;
        return;
    }

     // if splitted size reaches minimal size , and more pieces to split left,
     // do no further splits since this size of piece is already sorted by  threads
    else if(remain_divide == 1){
        remain_divide -= 1;
        return;
    }



	if (first < last )
	{
        // stop split when all the piceces reauired for total_num_threads are already exists.
        // which means, when it reaches the same size as size of program1 has sorted, do not split
        if(remain_divide >0){mergeSort( first, mid, depth+1);}
		if(remain_divide >0){mergeSort( mid + 1, last,depth+1);}
        
        // sorted pieces by program1 will merge here.
        merge(first, last);
		
	}
}

void mergeSort_simple( int first, int last)
{   
    // original merge sort

    int mid = (first + last) / 2;
	
	// if a splitted piece is bigger than 1 keep split and merge recursively
	if (first < last)
	{
		mergeSort_simple(first, mid);
		mergeSort_simple( mid + 1, last);
		merge( first, last);
	}
}


void * mergeSort_thread(void * parameter)
{       
    
    // mergeSort for thread
    ARG *arg= (ARG*)parameter;  
    int first = (arg->first);
    int last = (arg->last);
    int mid = (first + last) / 2;

    // if a splitted piece is bigger than 1 keep split and merge recursively
	if (first < last )
	{   
     
		mergeSort_simple(first, mid);
		mergeSort_simple( mid + 1, last);
		merge( first, last);
	}
    
}



void parseInputs(int first, int last, int depth, ARG* arg_array)
{   
    int mid = (first + last) / 2;
    
    // if spliting reaches max_depth, save current splitted part as an input arg for a child process.
    if(depth == max_depth && remain_divide > 1){
       
        arg_array[arg_arr_index].first = first;
        arg_array[arg_arr_index].last = last;
        arg_arr_index +=1;
        remain_divide -=1;
        return;
        
    }
    // if spliting reaches max_depth, save current splitted part "PLUS" rest of picese part of array as an input arg for a child process
    else if(remain_divide == 1){
        arg_array[arg_arr_index].first = first;
        arg_array[arg_arr_index].last = N-1;
        
        arg_arr_index +=1;
        remain_divide -=1;
        return;
      
    }



	if (first < last )
	{
        // stop split when all the piceces reauired for total_num_process are already exists.
        if(remain_divide >0){parseInputs(first, mid, depth+1, arg_array);}
		if(remain_divide >0){parseInputs( mid + 1, last, depth+1, arg_array);}

	}
}


int main(int argc, char **argv) {
	
    // inits
	cin >> N;
	array = new int[N];
    ARG arg;
    
    total_thread_num = stoi(argv[1]);
    remain_divide = total_thread_num;

    max_depth = 1;
    while(pow(2,max_depth) < total_thread_num){
        max_depth += 1;

    }

	for (int i = 0; i < N; i++){
		cin >> array[i];
    }
 
    //parse input
    ARG arg_array[total_thread_num];
    arg_arr_index = 0;
    parseInputs(0,N-1,0, arg_array);

    // measure time
    timespec start;
    clock_gettime(CLOCK_MONOTONIC,&start);

    // thread create
    pthread_t threads[total_thread_num];
    pthread_attr_t attr;


    for(int i=0; i< total_thread_num; i++){
        pthread_create(&threads[i], NULL, mergeSort_thread, (void*)&arg_array[i]);
    
    }
    // wait for threads made
    for(int i=0; i< total_thread_num; i++){
        
        pthread_join(threads[i], NULL);

    }
    remain_divide = total_thread_num;
    
    // merge results of threads
    mergeSort(0,N-1,0);

    // measure time
    timespec stop;
    clock_gettime(CLOCK_MONOTONIC,&stop);
    long total_time_ms = getnanosec(start, stop)/ (1000 *1000);

 
    // prints
	for(int i = 0; i < N; i++){
        cout << array[i] << " ";
    }
    cout << "\n" << total_time_ms;
    return 0;
}