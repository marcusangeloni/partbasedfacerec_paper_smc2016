#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus@liv.ic.unicamp.br>
# Thu 7 Apr 2016 21:12:03

import os
import numpy
import sys
import bob.measure
import argparse
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Combine the features directory with the current file
def get_feature_path(features_path, f):
    return os.path.join(features_path, f + ".npy")

# Load the impostor files from evaluation and test sets
def read_impostors(features_path, eval_path, test_path):
    f = open(eval_path)
    lines = f.readlines()
    f.close()

    dim = numpy.load(get_feature_path(features_path, lines[0].rstrip())).flatten().shape[0]
    eval_impostors = numpy.zeros((len(lines), dim), dtype = numpy.float64)
    
    idx = 0
    for l in lines:
        feature = numpy.load(get_feature_path(features_path, l.rstrip())).flatten()
        eval_impostors[idx] = feature
        idx += 1

    f = open(test_path)
    lines = f.readlines()
    f.close() 

    test_impostors = numpy.zeros((len(lines), dim), dtype = numpy.float64)

    idx = 0
    for l in lines:
        feature = numpy.load(get_feature_path(features_path, l.rstrip())).flatten()
        test_impostors[idx] = feature
        idx += 1

    return (eval_impostors, test_impostors)

# Load the client files from evaluation and test sets
def read_clients(features_path, eval_path, test_path):
    f = open(eval_path)
    lines = f.readlines()
    f.close()
    
    dim = numpy.load(get_feature_path(features_path, lines[0].split()[1])).flatten().shape[0]

    eval_clients_X = numpy.zeros((len(lines), dim), dtype = numpy.float64)
    eval_clients_y = []
    
    idx = 0
    for l in lines:
        fields = l.split()
        feature = numpy.load(get_feature_path(features_path, fields[1])).flatten()
        eval_clients_X[idx] = feature
        eval_clients_y.append(fields[0])
        idx += 1
    
    f = open(test_path)
    lines = f.readlines()
    f.close()
    
    test_clients_X = numpy.zeros((len(lines), dim), dtype = numpy.float64)
    test_clients_y = []

    idx = 0
    for l in lines:
        fields = l.split()
        feature = numpy.load(get_feature_path(features_path, fields[1])).flatten()
        test_clients_X[idx] = feature
        test_clients_y.append(fields[0])
        idx += 1

    return (eval_clients_X, eval_clients_y, test_clients_X, test_clients_y)

# Compute the Equal Error Rate based on impostor and genuine scores vector. And also returns
# the difference between the highest impostor score and lowest genuine score
def EER(impostor, genuine):
    thres = bob.measure.eer_threshold(impostor, genuine)
    FAR,FRR  = bob.measure.farfrr(impostor, genuine, thres)
    eer = ((FAR + FRR) / 2.)
    diff = min(genuine) - max(impostor)

    return eer, diff

# Train a Principal Component Analysis to dimensionality reduction (retaining at least 99.5% of variance)
def pcaTrain(features):
    pca = PCA()
    pca.fit(features)

    # Check the required dimensionality to retain 99.5% of variance
    cumulativeVariance = numpy.cumsum(pca.explained_variance_ratio_)

    components = 0
    while cumulativeVariance[components] < 0.995:
        components += 1
    
    pca = PCA(n_components = components + 1)
    pca.fit(features)

    return pca

# Train a Support Vector Machine classifier with Radial Basis Function kernel with
# chosen C and gamma arguments
def svmTrain(C, gamma, X, y):
    svm = SVC(C = C, gamma = gamma, kernel = 'rbf')
    svm.fit(X, y)
    return svm

# Train a Random Forest classifier with chosen n_estimators and max_features arguments
def randomForestTrain(n_estimators, max_features, X, y):
    if (max_features is None):
        rf = RandomForestClassifier(n_estimators = n_estimators, max_features = None)
    else:
        rf = RandomForestClassifier(n_estimators = n_estimators, max_features = max_features)
    rf.fit(X, y)
    return rf

# Train a K-Neighbors classifier with selected neighbors, weight and
# algorithm
def knnTrain(n_neighbors, weights, algorithm, X, y):
    knn = KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights, algorithm = algorithm)
    knn.fit(X, y)
    return knn

#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description='Train and apply SVM, Random Forest and K-NN classifiers')
parser.add_argument('experiment', default='', help='Name of the experiment (prefix of output score files)')
parser.add_argument('features_path', default='', help='Features directory root (with facial parts folders and npy files)')
parser.add_argument('protocol_dir', default='', help='Protocol directory')

args = parser.parse_args()

if (not(os.path.exists(args.features_path))):
    print('Features directory root (\"' + args.features_path + '\") not found.')
    exit()

if (not(os.path.exists(args.protocol_dir))):
    print('Protocol directory (\"' + args.protocol_dir + '\") not found.')
    exit()

features_path = args.features_path
protocol_dir = args.protocol_dir
experiment = args.experiment

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Experiment (" + experiment + ") started")
print("Features directory: " + features_path)
print("Protocol directory: " + protocol_dir)

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Reading train files to grid search step")
clients = {} # Clients dictionary

####### Read train_clients_01 file
list_file = open(os.path.join(protocol_dir, "train_clients_01.txt"))
lines = list_file.readlines()
list_file.close()

dim_samples = len(lines) # Files counter
dim_features = numpy.load(get_feature_path(features_path, lines[0].split()[1].rstrip())).flatten().shape[0] # Features dimensionality
idx = 0 # Index to fill the X and y sets
gallery_X = numpy.zeros((dim_samples, dim_features), dtype = numpy.float64) # Features matrix

for l in lines:
    fields = l.rstrip().split()

    # Fill the features matrix
    gallery_X[idx] = numpy.load(get_feature_path(features_path, fields[1].rstrip())).flatten()
    
    # Fill the dictionary
    if not (fields[0] in clients):
        clients[fields[0]] = {"svm" : SVC(),
                              "rf" : RandomForestClassifier(),
                              "knn" : KNeighborsClassifier(),
                              "gallery_y" : numpy.zeros((dim_samples), dtype = numpy.uint8)}

    clients[fields[0]]["gallery_y"][idx] = 1
    idx += 1

####### Read train_clients_02 file
list_file = open(os.path.join(protocol_dir, "train_clients_02.txt"))
lines = list_file.readlines()
list_file.close()

dim_samples = len(lines) # Files counter
idx = 0 # Index to fill the X and y sets
probe_X = numpy.zeros((dim_samples, dim_features), dtype = numpy.float64) # Features matrix

# Create the probe_y field in the dictionary
for key in clients:
    clients[key].update({"probe_y" : numpy.zeros((dim_samples), dtype = numpy.uint8)})

for l in lines:
    fields = l.rstrip().split()
    
    # Fill the features matrix
    probe_X[idx] = numpy.load(get_feature_path(features_path, fields[1].rstrip())).flatten()
    
    # Fill the dictionary
    clients[fields[0]]["probe_y"][idx] = 1
    idx += 1

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Grid search structure was created")
             
# Train the PCA with train_clients_01 data
pca = pcaTrain(gallery_X)

# Apply PCA in the training data used in the grid search step
gallery_X = pca.transform(gallery_X)
probe_X = pca.transform(probe_X)

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - PCA trained and applied (retained components = " + str(pca.n_components) + ")")

####### Grid search to choose the classifier settings with training data

# SVM
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - SVM Grid Search started")
C_set = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
gamma_set = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

C = 0.0
gamma = 0.0
eer = 1.1
diff = 0.0

for curr_C in C_set:
    for curr_gamma in gamma_set:
        # Creates a classifier for each user using the current settings
        genuine = numpy.zeros(shape = (0))
        impostor = numpy.zeros(shape = (0))

        for key in clients:
            clients[key]["svm"] = svmTrain(curr_C, curr_gamma, gallery_X, clients[key]["gallery_y"])
            
            for idx in range(0, len(probe_X)):
                score = clients[key]["svm"].decision_function(probe_X[idx].reshape(1, -1))
                if (clients[key]["probe_y"][idx] == 1):
                    genuine = numpy.concatenate((genuine, score), axis = 0)
                else:
                    impostor = numpy.concatenate((impostor, score), axis = 0)
                    
        curr_eer, curr_diff = EER(impostor, genuine)
        print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - SVM: C = " + str(curr_C) + ", gamma = " + str(curr_gamma) + " | EER = " + str(curr_eer) + " | diff = " + str(curr_diff) + ")")

        if (curr_eer < eer) or ((eer == curr_eer) and (curr_diff > diff)):
            eer = curr_eer
            diff = curr_diff
            C = curr_C
            gamma = curr_gamma

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - chosen SVM settings: C = " + str(C) + " gamma = " + str(gamma))
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - SVM Grid Search finished")

# Random Forest
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - RandomForest Grid Search started")
n_estimators_set = [10, 25, 50, 100, 150, 200, 250, 300, 350, 450, 500]
max_features_set = [None, 'auto', 'log2']

n_estimators = 0
max_features = ''
eer = 1.1
diff = 0.0

for curr_n_estimators in n_estimators_set:
    for curr_max_features in max_features_set:
        # Creates a classifier for each user using the current settings
        genuine = numpy.zeros(shape = (0))
        impostor = numpy.zeros(shape = (0))
		
        for key in clients:
            clients[key]["rf"] = randomForestTrain(curr_n_estimators, curr_max_features, gallery_X, clients[key]["gallery_y"])

            for idx in range(0, len(probe_X)):
                score = clients[key]["rf"].predict_proba(probe_X[idx].reshape(1, -1))[0, 1]
                if (clients[key]["probe_y"][idx] == 1):
                    genuine = numpy.concatenate((genuine, [score]), axis = 0)
                else:
                    impostor = numpy.concatenate((impostor, [score]), axis = 0)
         
        curr_eer, curr_diff = EER(impostor, genuine)
        print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - RandomForest: n_estimators = " + str(curr_n_estimators) + ", max_features = " + str(curr_max_features) + " | EER = " + str(curr_eer)  + " | diff = " + str(curr_diff) + ")")
        
        if (curr_eer < eer) or ((eer == curr_eer) and (curr_diff > diff)):
            eer = curr_eer
            diff = curr_diff
            n_estimators = curr_n_estimators
            max_features = curr_max_features

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - chosen RandomForest settings: n_estimators = " + str(n_estimators) + " max_features = " + str(max_features))
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - RandomForest Grid Search finished")

# KNN
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - KNN Grid Search started")
n_neighbors_set = [1, 2, 3, 4, 5, 6, 7, 8, 9]
weights_set = ['distance', 'uniform']
algorithm_set = ['ball_tree', 'kd_tree', 'brute']

n_neighbors = 0
weights = ''
algorithm = ''
eer = 1.1
diff = 0.0

for curr_n_neighbors in n_neighbors_set:
    for curr_weights in weights_set:
        for curr_algorithm in algorithm_set:
            # Creates a classifier for each user using the current settings
            genuine = numpy.zeros(shape = (0))
            impostor = numpy.zeros(shape = (0))

            for key in clients:
                clients[key]["knn"] = knnTrain(curr_n_neighbors, curr_weights, curr_algorithm, gallery_X, clients[key]["gallery_y"])
           
                for idx in range(0, len(probe_X)):
                    score = clients[key]["knn"].predict_proba(probe_X[idx].reshape(1, -1))[0, 1]
                    if (clients[key]["probe_y"][idx] == 1):
                        genuine = numpy.concatenate((genuine, [score]), axis = 0)
                    else:
                        impostor = numpy.concatenate((impostor, [score]), axis = 0)
    
            curr_eer, curr_diff = EER(impostor, genuine)
            print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - KNN: n_neighbors = " + str(curr_n_neighbors) + ", weights = " + curr_weights + ", algorithm = " + curr_algorithm + " | EER = " + str(curr_eer)  + " | diff = " + str(curr_diff) + ")")
    
            if (curr_eer < eer) or ((eer == curr_eer) and (curr_diff > diff)):
                eer = curr_eer
                diff = curr_diff
                n_neighbors = curr_n_neighbors
                weights = curr_weights
                algorithm = curr_algorithm

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - chosen KNN: n_neighbors = " + str(n_neighbors) + " weights = " + weights + " algorithm = " + algorithm)
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - KNN Grid Search finished")


####### Experiments
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - ###### Experiment started #####")
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Reading training set")
list_file = open(os.path.join(protocol_dir, "train_clients.txt"))
lines = list_file.readlines()
list_file.close()

dim_samples = len(lines) # Files counter
idx = 0 # Index to fill X and y sets
train_clients = numpy.zeros((dim_samples, dim_features), dtype = numpy.float64) # Features matrix

# Cleans the y vector
for key in clients:
    clients[key]["gallery_y"] = numpy.zeros((dim_samples), dtype = numpy.uint8)

for l in lines:
    fields = l.rstrip().split()
    train_clients[idx] = numpy.load(get_feature_path(features_path, fields[1].rstrip())).flatten()
    clients[fields[0]]["gallery_y"][idx] = 1
    idx += 1

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Reading evaluation and test sets")
eval_clients_X, eval_clients_y, test_clients_X, test_clients_y = read_clients(features_path,
                                                                              os.path.join(protocol_dir, "evaluation_clients.txt"),
                                                                              os.path.join(protocol_dir, "test_clients.txt"))
eval_impostors, test_impostors = read_impostors(features_path,
                                                os.path.join(protocol_dir, "evaluation_impostors.txt"),
                                                os.path.join(protocol_dir, "test_impostors.txt"))

# Dimensionality reduction of the experiment data
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Aplying PCA")
pca = pcaTrain(train_clients)
train_clients = pca.transform(train_clients)
eval_clients_X = pca.transform(eval_clients_X)
test_clients_X = pca.transform(test_clients_X)
eval_impostors = pca.transform(eval_impostors)
test_impostors = pca.transform(test_impostors)

# Train classifiers
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Training classifiers")
for key in clients:
    clients[key]["svm"] = svmTrain(C, gamma, train_clients, clients[key]["gallery_y"])
    clients[key]["rf"] = randomForestTrain(n_estimators, max_features, train_clients, clients[key]["gallery_y"])
    clients[key]["knn"] = knnTrain(n_neighbors, weights, algorithm, train_clients, clients[key]["gallery_y"])

# Score vectors of genuine comparisons of evaluation and test sets
svm_eval_client = numpy.zeros(shape = (0))
svm_test_client = numpy.zeros(shape = (0))
rf_eval_client = numpy.zeros(shape = (0))
rf_test_client = numpy.zeros(shape = (0))
knn_eval_client = numpy.zeros(shape = (0))
knn_test_client = numpy.zeros(shape = (0))

# Score vectors of impostor comparisons of evaluation and test sets
svm_eval_impostor = numpy.zeros(shape = (0))
svm_test_impostor = numpy.zeros(shape = (0))
rf_eval_impostor = numpy.zeros(shape = (0))
rf_test_impostor = numpy.zeros(shape = (0))
knn_eval_impostor = numpy.zeros(shape = (0))
knn_test_impostor = numpy.zeros(shape = (0))

# Run impostor trials
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Run impostor trials")
for key in clients:
    for trial in eval_impostors:
        score = clients[key]["svm"].decision_function(trial.reshape(1, -1))
        svm_eval_impostor = numpy.concatenate((svm_eval_impostor, score), axis = 0)
        score = clients[key]["rf"].predict_proba(trial.reshape(1, -1))[0, 1]
        rf_eval_impostor = numpy.concatenate((rf_eval_impostor, [score]), axis = 0)
        score = clients[key]["knn"].predict_proba(trial.reshape(1, -1))[0, 1]
        knn_eval_impostor = numpy.concatenate((knn_eval_impostor, [score]), axis = 0)

    for trial in test_impostors:
        score = clients[key]["svm"].decision_function(trial.reshape(1, -1))
        svm_test_impostor = numpy.concatenate((svm_test_impostor, score), axis = 0)
        score = clients[key]["rf"].predict_proba(trial.reshape(1, -1))[0, 1]
        rf_test_impostor = numpy.concatenate((rf_test_impostor, [score]), axis = 0)
        score = clients[key]["knn"].predict_proba(trial.reshape(1, -1))[0, 1]
        knn_test_impostor = numpy.concatenate((knn_test_impostor, [score]), axis = 0)

# Run genuine trials
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Run genuine trials")
for idx in range(0, len(eval_clients_X)):
    score = clients[eval_clients_y[idx]]["svm"].decision_function(eval_clients_X[idx].reshape(1, -1))
    svm_eval_client = numpy.concatenate((svm_eval_client, score), axis = 0)
    score = clients[eval_clients_y[idx]]["rf"].predict_proba(eval_clients_X[idx].reshape(1, -1))[0, 1]
    rf_eval_client = numpy.concatenate((rf_eval_client, [score]), axis = 0)
    score = clients[eval_clients_y[idx]]["knn"].predict_proba(eval_clients_X[idx].reshape(1, -1))[0, 1]
    knn_eval_client = numpy.concatenate((knn_eval_client, [score]), axis = 0)
for idx in range(0, len(test_clients_X)):
    score = clients[test_clients_y[idx]]["svm"].decision_function(test_clients_X[idx].reshape(1, -1))
    svm_test_client = numpy.concatenate((svm_test_client, score), axis = 0)
    score = clients[test_clients_y[idx]]["rf"].predict_proba(test_clients_X[idx].reshape(1, -1))[0, 1]
    rf_test_client = numpy.concatenate((rf_test_client, [score]), axis = 0)
    score = clients[test_clients_y[idx]]["knn"].predict_proba(test_clients_X[idx].reshape(1, -1))[0, 1]
    knn_test_client = numpy.concatenate((knn_test_client, [score]), axis = 0)

print(">>>>>> Evaluation set results (Equal Error Rates) for each classifier")
print("SVM = " + str(EER(svm_eval_impostor, svm_eval_client)))
print("RandomForest = " + str(EER(rf_eval_impostor, rf_eval_client)))
print("KNN = " + str(EER(knn_eval_impostor, knn_eval_client)))
print(">>>>>> Test set results (Equal Error Rates) for each classifier")
print("SVM = " + str(EER(svm_test_impostor, svm_eval_client)))
print("RandomForest = " + str(EER(rf_test_impostor, rf_test_client)))
print("KNN = " + str(EER(knn_test_impostor, knn_test_client)))

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Saving output score files")
numpy.savetxt(experiment + "_svm_eval_genuine.txt", svm_eval_client, fmt='%.19f')
numpy.savetxt(experiment + "_rf_eval_genuine.txt", rf_eval_client, fmt='%.19f')
numpy.savetxt(experiment + "_knn_eval_genuine.txt", knn_eval_client, fmt='%.19f')
numpy.savetxt(experiment + "_svm_test_genuine.txt", svm_test_client, fmt='%.19f')
numpy.savetxt(experiment + "_rf_test_genuine.txt", rf_test_client, fmt='%.19f')
numpy.savetxt(experiment + "_knn_test_genuine.txt", knn_test_client, fmt='%.19f')
numpy.savetxt(experiment + "_svm_eval_impostor.txt", svm_eval_impostor, fmt='%.19f')
numpy.savetxt(experiment + "_rf_eval_impostor.txt", rf_eval_impostor, fmt='%.19f')
numpy.savetxt(experiment + "_knn_eval_impostor.txt", knn_eval_impostor, fmt='%.19f')
numpy.savetxt(experiment + "_svm_test_impostor.txt", svm_test_impostor, fmt='%.19f')
numpy.savetxt(experiment + "_rf_test_impostor.txt", rf_test_impostor, fmt='%.19f')
numpy.savetxt(experiment + "_knn_test_impostor.txt", knn_test_impostor, fmt='%.19f')
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - ###### Experiment finished #####")