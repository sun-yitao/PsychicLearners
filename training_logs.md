<!---
Copy the first line for easier adding of rows
| Model | Hyperparameters | Dataset Version | Train Accuracy | Validation Accuracy | Test Accuracy |
-->

# Training Logs

## Title Classification

| Model | Hyperparameters | Dataset Version | Train Accuracy | Validation Accuracy | Test Accuracy |
| ----- | --------------- | --------------- | -------------- | ------------------- | ------------- |
| Logistic Regression with Tfidf ngrams and non ngrams | NA | train_split.csv | NA | 74.06 | NA |
| Naive Bayes Count Vectors | NA | train_split.csv | NA | 66.63 | NA |
| Random Forest Count Vectors and TFIDF | Default | train_split.csv | NA | 63.5 | NA |
| Xgb, Count Vectors | Default | train_split.csv | NA | 0.6877658018496433 | NA |
| Xgb, WordLevel TF-IDF | Default | train_split.csv | NA | 0.689048401251097 | NA |
| BRF, Count Vectors | Default | train_split.csv | NA | 0.5544954733991885 | NA |
| BRF, WordLevel TF-IDF | Default | train_split.csv | NA | 0.2852546072320605 | NA |
| BRF, N-Gram VectorsDefault  | Default | train_split.csv | NA | 0.3824696413972083 | NA |
| EEC, Count Vectors | Default |  train_split.csv | NA | 0.0337601164090217 | NA |
| EEC, WordLevel TF-IDF | Default | train_split.csv | NA | 0.1318977220734607 | NA |
| EEC, N-Gram Vectors | Default | train_split.csv | NA | 0.010613322532496269 | NA |
| RUSB, Count Vectors | Default | train_split.csv | NA | 0.375576607187057 | NA |
| RUSB, WordLevel TF-IDF | Default | train_split.csv | NA | 0.3271303526023267 | NA |
| RUSB, N-Gram Vectors | Default | train_split.csv | NA | 0.3239126032267501 | NA |
NB, Count Vectors:  0.6056494378314319
NB, WordLevel TF-IDF:  0.599108930942148
NB, N-Gram Vectors:  0.6283161945050741
LR, Count Vectors:  0.6410146786375944
LR, WordLevel TF-IDF:  0.6142376034142646
LR, N-Gram Vectors:  0.6142376034142646
SVM, N-Gram Vectors:  0.6038492983206198
RF 580, Count Vectors:  0.5815950736182054
RF 580, WordLevel TF-IDF:  0.5977888286342191
RF 5800, Count Vectors:  0.5823226300038253
RF 5800, WordLevel TF-IDF:  0.5834402166167878
Xgb, Count Vectors:  0.6374669036850356
Xgb, WordLevel TF-IDF:  0.6365443321857444

## Image Classification

| Model | Hyperparameters | Dataset Version | Train Accuracy | Validation Accuracy | Test Accuracy |
| ----- | --------------- | --------------- | -------------- | ------------------- | ------------- |