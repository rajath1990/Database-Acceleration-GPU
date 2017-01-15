-----------------------------------------------------------------------------

    EE 5351 -- DBMS Acceleration Group 

      Jinfeng Yang (ID# 4651346), 
      Jacob Long   (ID# 4311081), 
      Rajath Raghavendra Prabhakar (ID# 5299546),
      HsiangYu Wen (ID# 5327082)
-----------------------------------------------------------------------------

   Description: 

      This program demonstrates parallelization of a typical mySQL 
         query on nVIDIA GPU devices, using the Q6 query defined in 
         the TPC-H benchmark specification. 

      Our implementation uses ideas from Flash Scan algorithms to decrease
         the initial size of database transfer from disk to host memory from
         1.5 GB to 352 MB.

      In translating the Q6 query to CUDA C code, we further optimize 
         the computation of the "select" and "where" clauses of the query
         using techniques learned throughout the semester. They include: 
            -Pinned Memory 
            -Memory Mapping 
            -Parallel Thread Mapping / Data Parallelization
            -Reduction Sums
            -CUDA Streams 

-----------------------------------------------------------------------------

   To run: 

     1. First, ensure the nVIDIA CUDA C SDK has been installed on the intended target host. 

     2. Copy the "EE5351ProjectFinal" directory to your in your SDK’s C/src folder. 

     3. Copy the *.txt files defined in "EE5351ProjectFinal" to your 
	/C/bin/linux/release folder. 

     4. From "/C/src/EE5351ProjectFinal," type "$ make". 

     5. This will produce executable "EE5351Final" in your SDK’s 
		/C/bin/linux/release directory. 

     6. Finally, execute the binary by typing "$ ./EE5351Final". 

     7. The program will run for approximately 9 seconds. The program will show
	  the actually computed query result, and breakdown response times 
         for I/O, GPU kernels + DMA Transfers, host CPU, and the overall 
         response time. 

-----------------------------------------------------------------------------
