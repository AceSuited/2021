

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <regex>

#include <string.h>
#include <unistd.h>
#include <cstdlib>
#include <stdio.h>


#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <time.h>

using namespace std;

void printPrompt(){
	/* 
	print prompt message
	*/

	// get PWD
	char pwd[250];
	getcwd(pwd,sizeof(pwd));
	
	// get name
	char *name;
	name = getlogin();
	
	// get current time
	time_t current_time = time(NULL);
	struct tm *pLocal = localtime(&current_time);
	
	// print prompt
	printf("[%02d:%02d:%02d]%s@%s$", pLocal->tm_hour,pLocal->tm_min, pLocal->tm_sec, name,pwd);
}

bool input(vector<string> &args){
	/* 
	get user input int args vector

	*/

	// clear the vector before get new inputs
	args.clear();

	// get input
	string input;
	getline(cin,input);


	// tokenizing begins here

	// first add spaces for convenience of further tokenizeing 
	int prev=0;
	int curr =0;

	// if '<' found, add space before and after '<' position
	curr = input.find('<');
	while(curr != string::npos){
		input.insert(curr, " ");
		input.insert(curr + 2, " ");
		prev = curr + 3;
		curr = input.find('<', prev);
		
	}
	prev = 0;
	curr = 0;

	// if '>' found, add space before and after '>' position
	curr = input.find('>');
	while(curr != string::npos){
		input.insert(curr, " ");
		input.insert(curr + 2, " ");
		prev = curr + 3;
		curr = input.find('>', prev);
		
	}

	prev = 0;
	curr = 0;

	// if '&' found, add space before and after '&' position
	curr = input.find('&');
	while(curr != string::npos){
		input.insert(curr, " ");
		input.insert(curr + 2, " ");
		prev = curr + 3;
		curr = input.find('&', prev);
		
	}
	
	
	// string tokenizing using strtok

	// in order to use strtok, convert string type to char* type.
	char c_string[strlen(input.c_str())];
	strcpy(c_string,input.c_str());

	// tokenizing and push each tokens to args vector.
	char *ptr = strtok(c_string, " ");
	while( ptr != NULL){
		args.push_back(ptr);
		ptr = strtok(NULL, " ");
	}


	// if user dosen`t put any inputs, return false
	if(args.size() > 0) return true;
	else return false;
}


void execute(vector<string> &args, bool isBackground){
	/*
	execute arguments by using fork() and execvp()
	This function is for args which dose not contains any redirection commands.

	bool isBackground : when true, background execution performed which means parent process does not wait for a child process to be done.
						when false, parent process wait for child process to be done.
	*/
	int num_args = args.size();
	char* char_args[num_args + 1];
    
    
    
	// convert string type tokens to char* type.
	for(int i =0; i < num_args; i++){
		
        char_args[i] = new char[args[i].length() + 1];
		strcpy(char_args[i], args[i].c_str());
	}

	char_args[num_args] = NULL;


	// fork
	pid_t pid;
	pid = fork();

	// child process
	if(pid==0)
	{	
		// execute arguments, if failed, print error message and finish process
		if(execvp(char_args[0], char_args) < 0){
			cout << char_args[0] << ": command not found\n";
			exit(0);
		}
		
	}
	//parent process
	// if this command is not a background command
	else if(pid > 0 && !isBackground) 
	{	
		// wait untill child process to be done
		int status;
		waitpid(pid,&status, 0);
	}
	// if this command is a background command
	else if(pid > 0 && isBackground){
		// no wait
		cout << "pid: " <<pid << "\n";
	}
	
}



bool isChangeDirectory(vector<string> args){
	/*
	check if this argument is about change directory
	return true if command is 'cd', else, false
	*/
	if(args[0].compare("cd") == 0){
		return true;
	}else{
		return false;
	}
}

void changeDirectory(vector<string> args){
	/*
	command cd 
	*/

	// if number of argument tokens are larger than 3, return error
	if(args.size() >= 3)
	{
		cout << "cd: too many arguments\n";
	}
	
	// for valid input, change working directory.
	else if(args.size() <2 || args[1].compare("~") == 0){
		chdir("/");
	}
	// succeed
	else if(chdir(args[1].c_str()) == 0){
		return;		
	}
	// if failed, print error message
	else{
		cout <<"cd: "<< args[1] << ": No such file or directory\n";
		return;
	}
		
	
}

bool isRedirection(vector<string> &args){
	/*
	check if this argument is redirection command
	return true if argument contains '<' or '>', else, false
	*/
	for(int i =0; i < args.size(); i++){
		if(args[i].compare("<")==0 || args[i].compare(">") ==0 ){
			return true;
		}
	}
	return false;
}

bool parse_filename(vector<string> &args, vector<string> &filenames){
	/*
	Helper function which will be used for executing redirection commands
	parse file names and save file names into a vector.
    if an user inputs multiple tokens of '< [filename]' ' > [filename]', final filename token which is appear at the most last will be saved to the vector
    ex) if user input is ./program1 < inputs.txt <inputs2.txt > output.txt > output2.txt,
        final filenames to be saved are 'inputs2.txt' for '<' redirection and 'output2.txt' for '>' redirection
	*/

	// variables
	string input_file_name;
	string outupt_file_name;
	int arg_size = args.size();


	// parse file name
	int i = 0;
	while (i < args.size()){

		// if it meets '<'
		if(args[i].compare("<") == 0 ){

			// if user does not input anything after '<', continue
			if(args.size() <= i + 1){

				args.erase(args.begin() + i);
				continue;
			}

			// save file name and remove '<' token and remove a token contains file name.
			input_file_name = args[i+1];
			args.erase(args.begin() + i, args.begin() + i + 2);
			
		}

		// if it meets '<'
		else if(args[i].compare(">") == 0){

			// if user does not input anything after '>', continue
			if(args.size() <= i + 1){

				args.erase(args.begin() + i);
				continue;
			}
			// save file name and remove '>' token and remove a token contains file name.
			outupt_file_name = args[i+1];
			args.erase(args.begin() + i, args.begin() + i + 2);
		}
		else{
			i+= 1;
		}
	}
	
    // save file names into the vector
	filenames.push_back(input_file_name);
	filenames.push_back(outupt_file_name);

    if(input_file_name.compare("") == 0 && outupt_file_name.compare("") == 0){
        return false;
    }else{
        return true;
    }
	
}

void execute_redirection(vector<string> &args, vector<string> &filenames, bool isBackground){

	/*
	This function execute commands with redirection
	vector<string> filenames : containes filenames as elements. 0 index is an input file and 1 index is a ouput file
	if file name deos not parsed, filenames are equial to empty string. 
	*/
	int num_args = args.size();
	char* char_args[num_args + 1];


	//convert argumnets into c string
	for(int i =0; i < num_args; i++){
		char_args[i] = new char[args[i].length() + 1];
		strcpy(char_args[i], args[i].c_str());
	}
	// add null to the end
	char_args[num_args] = NULL;


	
	// fork
	pid_t pid;
	pid = fork();


	if(pid==0)
	{
		// if input redirection exists,
		// replace standard inputfile with input file by user.
		if(filenames[0].compare("") != 0){
			int in;

			// open input file
			in = open(filenames[0].c_str(), O_RDONLY);
			if(in < 0){

				// if file open fails,
				cout << filenames[0].c_str() << ": No such file or directory\n";
				exit(0);
			}
			//replace stdinput and close file
			dup2(in,0);
			close(in);
		}
		// if ouput redirection exists
		// replace standard output file with outpuut file by user.
		if(filenames[1].compare("") != 0){
			int out;

			// open outputfile
			out = open(filenames[1].c_str(),  O_WRONLY | O_TRUNC | O_CREAT, S_IRUSR | S_IRGRP | S_IWGRP | S_IWUSR);
			if(out < 0){
				// if file open fails,
				cout << filenames[1].c_str() << ": No such file or directory\n";
				exit(0);
			}
			//replace stdoutput and close file
			dup2(out,1);
			close(out);
		}

		// execute arguments
		if(execvp(char_args[0], char_args) < 0){
			cout << char_args[0] << ": command not found\n";
			exit(0);
		};
	}
    //parent process
	// if this command is not a background command
	else if(pid > 0 && !isBackground) 
	{	
		// wait untill child process to be done
		int status;
		waitpid(pid,&status, 0);
	}
	// if this command is a background command
	else if(pid > 0 && isBackground){
		// no wait
		cout << "pid: " <<pid << "\n";
	}
	



}



bool isBacground(vector<string> &args){
	/*
	check if given arguments contains '&' character at "the end of command"
	*/

		for(int i =0; i < args.size(); i++){
		if(args[i].compare("&")==0){
			return true;
		}
	}
	return false;
}

void execute_background(vector<string> &args){
	/*
	execute background commands
	parent process does not wait for child process to be end.
	*/
	args.pop_back();
	int num_args = args.size();
	char* char_args[num_args + 1];

	// type conversion to char*
	for(int i =0; i < num_args; i++){
		char_args[i] = new char[args[i].length() + 1];
		strcpy(char_args[i], args[i].c_str());
	}
	
	char_args[num_args] = NULL;


	// execute accordingly
	if(isChangeDirectory(args))
	{
		changeDirectory(args);
	}
	else if(isRedirection(args)){
        vector<string> filenames;
        if(parse_filename(args, filenames)){
            execute_redirection(args, filenames, true);
        }
        else{
            cout << "syntax error : wrong redirection command\n";
        }
        
	}
	else {
		execute(args, true);
	}
		
		
	
}

int main()
{


	while(true)
	{	
		// print prompt and get input
		vector<string> args;
		printPrompt();
		
		// if input exist,
		if(input(args)){

			// exit command			
			if (args[0].compare("exit") == 0)  
			{	
				exit(0);
			}
			// built in change directory 
            else if(isChangeDirectory(args))
			{
				changeDirectory(args);
			}
			// background execution
			else if(isBacground(args)){
				execute_background(args);
			}
			// redirection execution
			else if(isRedirection(args)){
                vector<string> filenames;
				
				// parse file and redirect
                if(parse_filename(args, filenames)){
                    execute_redirection(args, filenames, false);
                }
                else{
                    cout << "syntax error : wrong redirection command\n";
                }
			}

			// normal execution			
			else{
				execute(args, false);
			}
		}
	}
	return 0;
}


