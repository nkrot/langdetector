## langdetector

A demo of a Language Identification Tool.

The implemented approach builds Maximum Entropy model based on character 1-, 2- and 3-grams.
MaxentModel from OpenNLP library (http://opennlp.apache.org/) is used.

The demo tool is hardcoded to run for the following languages:
  Catalan, Spanish, French, Italian, Portuguese, Romanian

Text data used in training is expected to be in the following files:

- data/train/ca.txt
- data/train/es.txt
- data/train/fr.txt
- data/train/it.txt
- data/train/pt.txt
- data/train/ro.txt  
  
The source code as well as data files can be found here:

- https://github.com/nkrot/langdetector
- git@github.com:nkrot/langdetector.git

Train and test data was extracted from tatoeba datasets (http://tatoeba.org/eng/downloads)
- training set: 4000 entries for each language 
- testing set:  1000 entries for each language (except Catalan - 905)

### Results

| Language | match | mismatch | precision, % |
|:--------:|:-----:|:--------:|:------------:|
|  ALL     | 4929  |   976    |  83          |
|   ca     |  688  |   217    |  76          |
|   es     |  770  |   230    |  77          |
|   fr     |  878  |   122    |  87          |
|   it     |  859  |   141    |  85          |
|   pt     |  834  |   166    |  83          |
|   ro     |  900  |   100    |  90          |

### TODO:

 1. it would be good to know what languages are confused most often
 2. train and test on longer sentences
 3. what is the quality if:
     a. training is accomplished on short sentences and testing on the long ones
     b. viceversa

