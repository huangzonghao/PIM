#TODO list

This file contains the description for the functions and functionalities 
to be implemented.

For new features, please add to the `New Feature` part.

## Planned Features

- [ ] different modes should be there, normal calculation mode, safe calculation
mode(temp files will be written)
- [x] indicate the input file name (default params.json, if not found, the program will abort)
- [x] select the format of the output (csv, nature language and so on, default is csv)
- [x] indicate the output file name (default output.txt)
- [ ] select the policy (specific policy or all, default all)
- [x] verbose or not (default not)
- [ ] recovery option(if a temp file is indicated, then the project will start from the stopped point)
- [ ] interrupt handler(will check if storing the state is need, note no periodically record will be kept in normal mode)



##New Features
* The default ouput contains only the value function of all states in the last period.
  Thus, for each policy, the output is simply an array of $$k^m$$ elements.
* Compare the relative difference of each policy to the optimal policy, i.e., for any two output files 
$$(a_i: i=1,2...,k^m)$$ 
and 
$$(b_i: i=1,2...,k^m)$$
compute $$c_i= \frac{a_i-b_i}{a_i}\cdot 100\%$$ 
for all i, and draw histogram of 
$$(c_i: i=1,2...,k^m)$$
