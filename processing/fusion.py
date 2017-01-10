#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus@liv.ic.unicamp.br>
# Thu 14 Apr 2016 18:02:45

import numpy
import os
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import bob
import argparse
import metrics

# standardize scores by removing the mean and scaling to unit variance
def standardization(imp_eval, gen_eval, imp_test, gen_test):
    scaler = StandardScaler()
    scaler.fit(numpy.concatenate((imp_eval,gen_eval)))
    imp_eval = scaler.transform(imp_eval)
    gen_eval = scaler.transform(gen_eval)
    imp_test = scaler.transform(imp_test)
    gen_test = scaler.transform(gen_test)

    return imp_eval, gen_eval, imp_test, gen_test

# compute the fusion scores with sum rule
def sum_fusion(imp_b, gen_b, imp_e, gen_e, imp_n, gen_n, imp_m, gen_m):
    imp = imp_b + imp_e + imp_n + imp_m
    gen = gen_b + gen_e + gen_n + gen_m

    return imp, gen

# compute the fusion scores with linear logistic regression
def llr_fusion(imp_ev_b, gen_ev_b, imp_ev_e, gen_ev_e, imp_ev_n, gen_ev_n, imp_ev_m, gen_ev_m, imp_te_b, gen_te_b, imp_te_e, gen_te_e, imp_te_n, gen_te_n, imp_te_m, gen_te_m):
    imp_ev_b = imp_ev_b.reshape(len(imp_ev_b), 1)
    gen_ev_b = gen_ev_b.reshape(len(gen_ev_b), 1)
    imp_ev_e = imp_ev_e.reshape(len(imp_ev_e), 1)
    gen_ev_e = gen_ev_e.reshape(len(gen_ev_e), 1)
    imp_ev_n = imp_ev_n.reshape(len(imp_ev_n), 1)
    gen_ev_n = gen_ev_n.reshape(len(gen_ev_n), 1)
    imp_ev_m = imp_ev_m.reshape(len(imp_ev_m), 1)
    gen_ev_m = gen_ev_m.reshape(len(gen_ev_m), 1)
   
    imp_eval = numpy.concatenate((imp_ev_b, imp_ev_e, imp_ev_n, imp_ev_m), axis = 1)
    gen_eval = numpy.concatenate((gen_ev_b, gen_ev_e, gen_ev_n, gen_ev_m), axis = 1)
    imp_eval = numpy.array(imp_eval,order='C')
    gen_eval = numpy.array(gen_eval,order='C')

    # train the LLR
    llrTrainer = bob.trainer.CGLogRegTrainer()
    llrMachine = bob.machine.LinearMachine()
                
    llrTrainer.train(llrMachine, gen_eval, imp_eval)
    
    # apply the LLR in the evaluation scores
    imp_eval = llrMachine(imp_eval)
    gen_eval = llrMachine(gen_eval)

    imp_eval = numpy.reshape(imp_eval,(imp_eval.shape[0]))
    gen_eval = numpy.reshape(gen_eval,(gen_eval.shape[0]))
    
    # apply the LLR in the test scores
    imp_te_b = imp_te_b.reshape(len(imp_te_b), 1)
    gen_te_b = gen_te_b.reshape(len(gen_te_b), 1)
    imp_te_e = imp_te_e.reshape(len(imp_te_e), 1)
    gen_te_e = gen_te_e.reshape(len(gen_te_e), 1)
    imp_te_n = imp_te_n.reshape(len(imp_te_n), 1)
    gen_te_n = gen_te_n.reshape(len(gen_te_n), 1)
    imp_te_m = imp_te_m.reshape(len(imp_te_m), 1)
    gen_te_m = gen_te_m.reshape(len(gen_te_m), 1)

    imp_test = numpy.concatenate((imp_te_b, imp_te_e, imp_te_n, imp_te_m), axis = 1)
    gen_test = numpy.concatenate((gen_te_b, gen_te_e, gen_te_n, gen_te_m), axis = 1)
    imp_test = numpy.array(imp_test,order='C')
    gen_test = numpy.array(gen_test,order='C')

    imp_test = llrMachine(imp_test)
    gen_test = llrMachine(gen_test)
    imp_test = numpy.reshape(imp_test,(imp_test.shape[0]))
    gen_test = numpy.reshape(gen_test,(gen_test.shape[0]))

    return imp_eval, gen_eval, imp_test, gen_test

#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description='Fusion of facial parts scores')
parser.add_argument('database', default='', help='Database name (prefix of score files)')
parser.add_argument('scores_path', default='', help='Directory with score files')

args = parser.parse_args()

if (not(os.path.exists(args.scores_path))):
    print('Score directory (\"' + args.scores_path + '\") not found.')
    exit()

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Fusion of facial parts scores started")
print("Database: " + args.database)
print("Score directory: " + args.scores_path)

# Facial parts
parts = numpy.array(["eyebrows", "eyes", "nose", "mouth"])

# Combination of each feature and each classifier
experiments = numpy.array(["dct_svm", "dct_rf", "dct_knn", "gabor_svm", "gabor_rf", "gabor_knn", "glcm_svm", "glcm_rf", "glcm_knn", "hog_svm", "hog_rf", "hog_knn", "mlbp_svm", "mlbp_rf", "mlbp_knn"])

# you can select a subset of combination
experiments_eyebrows = experiments #numpy.array(["dct_svm"]) #numpy.array(["mlbp_svm"]) #numpy.array(["gabor_svm"])
experiments_eyes = experiments #numpy.array(["gabor_svm"]) #numpy.array(["mlbp_svm"]) #numpy.array(["hog_svm"])
experiments_nose = experiments #numpy.array(["mlbp_svm"]) #numpy.array(["gabor_svm"]) #numpy.array(["dct_svm"])
experiments_mouth = experiments #numpy.array(["gabor_svm"]) #numpy.array(["gabor_svm"]) #numpy.array(["gabor_svm"])

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " Creating score structure")

# Dictionary os scores
scores = {}

# Create each facial part
for i in range(0, len(parts)):
    scores[parts[i]] = {}

# Read the scores and Z-normalized them
for j in range(0, len(experiments_eyebrows)):
    imp_eval, gen_eval, imp_test, gen_test = metrics.read_scores(args.scores_path, args.database, "eyebrows", experiments_eyebrows[j])
    imp_eval, gen_eval, imp_test, gen_test = standardization(imp_eval, gen_eval, imp_test, gen_test)
    scores["eyebrows"].update({experiments_eyebrows[j] : {}})
    scores["eyebrows"][experiments_eyebrows[j]].update({"gen_eval" : gen_eval, "imp_eval" : imp_eval, "gen_test" : gen_test, "imp_test" : imp_test})

for j in range(0, len(experiments_eyes)):
    imp_eval, gen_eval, imp_test, gen_test = metrics.read_scores(args.scores_path, args.database, "eyes", experiments_eyes[j])
    imp_eval, gen_eval, imp_test, gen_test = standardization(imp_eval, gen_eval, imp_test, gen_test)
    scores["eyes"].update({experiments_eyes[j] : {}})
    scores["eyes"][experiments_eyes[j]].update({"gen_eval" : gen_eval, "imp_eval" : imp_eval, "gen_test" : gen_test, "imp_test" : imp_test})

for j in range(0, len(experiments_nose)):
    imp_eval, gen_eval, imp_test, gen_test = metrics.read_scores(args.scores_path, args.database, "nose", experiments_nose[j])
    imp_eval, gen_eval, imp_test, gen_test = standardization(imp_eval, gen_eval, imp_test, gen_test)
    scores["nose"].update({experiments_nose[j] : {}})
    scores["nose"][experiments_nose[j]].update({"gen_eval" : gen_eval, "imp_eval" : imp_eval, "gen_test" : gen_test, "imp_test" : imp_test})

for j in range(0, len(experiments_mouth)):
    imp_eval, gen_eval, imp_test, gen_test = metrics.read_scores(args.scores_path, args.database, "mouth", experiments_mouth[j])
    imp_eval, gen_eval, imp_test, gen_test = standardization(imp_eval, gen_eval, imp_test, gen_test)
    scores["mouth"].update({experiments_mouth[j] : {}})
    scores["mouth"][experiments_mouth[j]].update({"gen_eval" : gen_eval, "imp_eval" : imp_eval, "gen_test" : gen_test, "imp_test" : imp_test})

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " Score structure completed")

# Header of output file
text = ""
line = "Hour,Fusion,Eyebrows,Eyes,Nose,Mouth,EER,FAE(FRE=0),FRE(FAE=0),FA_test_EER,FR_test_EER,TER_test_EER,FA_test_FRE0,FR_test_FRE0,TER_test_FRE0,FA_test_FAE0,FR_test_FAE0,TER_test_FAE0"
print(line)
text += line + "\n"


for eb in experiments_eyebrows:
    for e in experiments_eyes:
        for n in experiments_nose:
            for m in experiments_mouth:

                print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Eyebrows: " + eb + ", Eyes: "+ e + ", Nose: " + n + ", Mouth: " + m)
                
                # LLR fusion
                imp_eval, gen_eval, imp_test, gen_test = llr_fusion(scores["eyebrows"][eb]["imp_eval"], scores["eyebrows"][eb]["gen_eval"],
                                                            scores["eyes"][e]["imp_eval"], scores["eyes"][e]["gen_eval"],
                                                            scores["nose"][n]["imp_eval"], scores["nose"][n]["gen_eval"],
                                                            scores["mouth"][m]["imp_eval"], scores["mouth"][m]["gen_eval"],
                                                            scores["eyebrows"][eb]["imp_test"], scores["eyebrows"][eb]["gen_test"],
                                                            scores["eyes"][e]["imp_test"], scores["eyes"][e]["gen_test"],
                                                            scores["nose"][n]["imp_test"], scores["nose"][n]["gen_test"],
                                                            scores["mouth"][m]["imp_test"], scores["mouth"][m]["gen_test"])
                                                            
                eer, fa_fre0, fr_fae0, fa_test_eer, fr_test_eer, ter_test_eer, fa_test_fre0, fr_test_fre0, ter_test_fre0, fa_test_fae0, fr_test_fae0, ter_test_fae0 = metrics.compute(imp_eval, gen_eval, imp_test, gen_test)
                
                line = datetime.now().strftime('%d/%m/%Y %H:%M:%S') + ',LLR,' + eb + ',' + e + ',' + n + ',' + m + ',' + ("%.4f"%(eer)) + '%,' + ("%.4f"%(fa_fre0)) + '%,'+ ("%.4f"%(fr_fae0)) + '%,' + ("%.4f"%(fa_test_eer)) + '%,' + ("%.4f"%(fr_test_eer)) + '%,' + ("%.4f"%(ter_test_eer)) + '%,' + ("%.4f"%(fa_test_fre0)) + '%,' + ("%.4f"%(fr_test_fre0)) + '%,' + ("%.4f"%(ter_test_fre0)) + '%,' + ("%.4f"%(fa_test_fae0)) + '%,' + ("%.4f"%(fr_test_fae0)) + '%,' + ("%.4f"%(ter_test_fae0))
                #print(line)
                text += line + "\n"

                # Sum rule fusion
                imp_eval, gen_eval = sum_fusion(scores["eyebrows"][eb]["imp_eval"], scores["eyebrows"][eb]["gen_eval"],
                                                scores["eyes"][e]["imp_eval"], scores["eyes"][e]["gen_eval"],
                                                scores["nose"][n]["imp_eval"], scores["nose"][n]["gen_eval"],
                                                scores["mouth"][m]["imp_eval"], scores["mouth"][m]["gen_eval"])
                imp_test, gen_test = sum_fusion(scores["eyebrows"][eb]["imp_test"], scores["eyebrows"][eb]["gen_test"],
                                                scores["eyes"][e]["imp_test"], scores["eyes"][e]["gen_test"],
                                                scores["nose"][n]["imp_test"], scores["nose"][n]["gen_test"],
                                                scores["mouth"][m]["imp_test"], scores["mouth"][m]["gen_test"])

                eer, fa_fre0, fr_fae0, fa_test_eer, fr_test_eer, ter_test_eer, fa_test_fre0, fr_test_fre0, ter_test_fre0, fa_test_fae0, fr_test_fae0, ter_test_fae0 = metrics.compute(imp_eval, gen_eval, imp_test, gen_test)
    
                line = datetime.now().strftime('%d/%m/%Y %H:%M:%S') + ',SUM,' + eb + ',' + e + ',' + n + ',' + m + ',' + ("%.4f"%(eer)) + '%,' + ("%.4f"%(fa_fre0)) + '%,'+ ("%.4f"%(fr_fae0)) + '%,' + ("%.4f"%(fa_test_eer)) + '%,' + ("%.4f"%(fr_test_eer)) + '%,' + ("%.4f"%(ter_test_eer)) + '%,' + ("%.4f"%(fa_test_fre0)) + '%,' + ("%.4f"%(fr_test_fre0)) + '%,' + ("%.4f"%(ter_test_fre0)) + '%,' + ("%.4f"%(fa_test_fae0)) + '%,' + ("%.4f"%(fr_test_fae0)) + '%,' + ("%.4f"%(ter_test_fae0))
                #print(line)
                text += line + "\n"

    #numpy.savetxt(args.database + "_eval_genuine.txt", gen_eval, fmt='%.19f')
    #numpy.savetxt(args.database + "_test_genuine.txt", gen_eval, fmt='%.19f')
    #numpy.savetxt(args.database + "_eval_impostor.txt", imp_eval, fmt='%.19f')
    #numpy.savetxt(args.database + "_test_impostor.txt", imp_test, fmt='%.19f')


file = open(args.database + "_fusion.csv", "wt")
file.write(text)
file.close()

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Fusion finished")