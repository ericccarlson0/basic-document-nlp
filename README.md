# Basic Document NLP

The purpose of this project is to benchmark basic NLP techniques when it comes to the task of document classification.
In many cases, these documents will have been generated through OCR, in which case extra work would need to be done to 
correct OCR-specific errors.

**NOTE**: This project is just getting started.

## Possible Basic Methods

* **Bag-of-Words**

Performing experiments...

* **Part of Speech, Chunking**

Performing experiments...

* **TF-IDF**

Performing experiments...

* **NER**

Not sure where Named Entity Recognition would be particularly useful.

* **Embeddings**

The files embeddings.py (in language_prep), write_embedding.py (in notebooks/write), and embedding_classifier (in 
notebooks/predict) are used to assess the use of "average embeddings" for words or bigrams. The process to produce 
these embeddings is as follows:
* Initialize a matrix with N rows and M columns, where N is the number of labels and M is the number of dimensions in 
the pre-prepared space of embeddings.
* Open a csv file whose rows have (1) document labels and (2) whose columns have cleaned-up OCR text from the docs.
* For each row of the csv file...
* Iterate over each word in the doc, getting its embedding and adding it to a temporary vector containing the sum of the 
document's words' embeddings thus far. When the iteration is over, divide the vector by the number of words accounted 
for to produce the "average embedding".
* At the same time, add the vector to the row of the N x M matrix which corresponds to the document's label.
* For each label, divide the corresponding row of the association matrix by the number of times that label was found, 
to produce rows that represent average embeddings for documents containing each label.
* Write the N x M matrix, which has the average embedding for each label, and another matrix containing the average 
embedding for each doc to somewhere it can be accessed easily.
