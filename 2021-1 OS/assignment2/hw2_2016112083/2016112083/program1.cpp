#include <iostream>
#include <time.h>

using namespace std;
int N;

// for time measuring
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


void mergeSort(int *array, int first, int last)
{
    int mid = (first + last) / 2;
	
	// if a splitted piece is bigger than 1 keep split and merge recursively
	if (first < last)
	{
		mergeSort(array, first, mid);
		mergeSort(array, mid + 1, last);
		merge(array, first, last);
	}
}

int main(int argc, char **argv) {
	
	// init
	cin >> N;
	int *array = new int[N];
    
	//get input
	for (int i = 0; i < N; i++){
		cin >> array[i];
    }
	

	// clock measure;
	timespec start;
    clock_gettime(CLOCK_MONOTONIC,&start);

	 // merge
	mergeSort(array, 0, N-1);
	
    // clock measure;
	timespec stop;
    clock_gettime(CLOCK_MONOTONIC,&stop);
    long total_time_ms = getnanosec(start, stop)/ (1000 *1000);
	


	// print result
	for(int i = 0; i < N; i++){
        cout << array[i] << " ";
    }
    cout << "\n" << total_time_ms;;
    return 0;
}
