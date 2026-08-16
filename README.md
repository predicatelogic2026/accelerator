1. PS-only programs:

(1) ps_only_synthetic.c: taking synthetic input instances

Use "gcc -O3 -g -o ps_only_synthetic ps_only_synthetic.c -fopenmp" to compile.

Run with "./ps_only_synthetic X Y", where X is the number of threads (1~4), and Y is the number of constants (25~75).

(2) ps_only_benchmark.c: taking benchmark input instances:

Use "gcc -O3 -g -o ps_only_benchmark ps_only_benchmark.c -fopenmp" to compile.

Run with "./ps_only_benchmark X Y Z", where X is the number of threads (1~4), Y is the name of the predicate file, and Z is the name of the rule file. 

- choose from one of the rule files: rules_lubm.txt or rules_oubm.txt
  
- choose from one of the predicate files: predicates_lubm_1d.txt ~ predicates_lubm_5d.txt or predicates_oubm_1d.txt ~ predicates_oubm_3d.txt

2. PS programs for the PS+PL approach
   
(1) ps_synthetic.c: taking synthetic input instances

Use "gcc -O3 -g -o ps_synthetic ps_synthetic.c" to compile.

Run with "./ps_synthetic X Y", where X is the number of PL workers (1~32), and Y is the number of constants (25~75). 

(2) ps_benchmark.c: taking benchmark input instances

Use "gcc -O3 -g -o ps_benchmark ps_benchmark.c" to compile.

Run with "./ps_benchmark X Y Z", where X is the number of PL workers (1~32), Y is the name of the predicate file, and Z is the name of the rule file. (See available predicate and rule file names in item 1.)

Please note that, all provided rule files and predicate files are created from the original dataset from LUBM benchmark (https://swat.cse.lehigh.edu/projects/lubm/) and UOBM benchmark (https://www.cs.ox.ac.uk/isg/tools/UOBMGenerator/.)

3. Vivado FPGA project for the PS+PL approach: FPGA_project_ver1.zip
