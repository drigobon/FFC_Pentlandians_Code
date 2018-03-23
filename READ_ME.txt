The code in this directory is used to replicate the Pentlandians’ submission to the Fragile Families Challenge. The single shell command ‘run_all.sh’ in the code directory runs all the necessary files in the proper order. The runtime for this replication tends to be very long, although some of the files are relatively quick. The output are in the ‘output’ directory, including all figures used for the code. Directories outside of 'code', 'data', and 'output' are created by the file and used by the code.

Runtime for each file, along with a description of purpose, inputs, and outputs can be found in the header. 

The ‘ResultsAnalysis’ file includes data that was collected from the challenge organizers regarding the holdout performance of several models. This data is not included in any .csv file and was manually input into this code file. Package requirements can be found in ‘requirements.txt’ for Python, and ‘requirements_r.txt’ for R, both located inside the ‘code’ directory.



Revision Memo:
The revision of our code uses the same base files, but changes the directories of inputs and outputs to be clearer. Additionally, we include a bash script that runs all necessary files to generate predictions and generate result plots.
As a result of our collaborative approach, the aggregation of this code and its verification took a significant amount of time, between 15 and 20 hours. For one, this process would have been easier with better organization of individual's code. However, this was a useful experience for the development of our manuscript, and hopefully will inform others about our study in conjunction with the paper itself.



