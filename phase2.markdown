 Please apply the machine learning algorithm you built to each of the 20 test cases in the testing data set. For more information and instructions on how to build your model see the prediction assignment writeup. For each test case you should submit a text file with a single capital letter (A, B, C, D, or E) corresponding to your prediction for the corresponding problem in the test data set. You get 1 point for each correct answer. You may submit up to 2 times for each problem. I know it is a lot of files to submit. It may be helpful to use the following function to create the files. If you have a character vector with your 20 predictions in order for the 20 problems. So something like (note these are not the right answers!):

```
answers = rep("A", 20)
```

then you can load this function by copying and pasting it into R:

``` 
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
```

then create a folder where you want the files to be written. Set that to be your working directory and run:

``` 
pml_write_files(answers)
```

and it will create one file for each submission. Note: if you use this script, please make sure the files that get written out have one character each with your prediction for the corresponding problem ID. I have noticed the script produces strange results if the answers variable is not a character vector. 
