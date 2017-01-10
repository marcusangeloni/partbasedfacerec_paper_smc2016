#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus@liv.ic.unicamp.br>
# Thu 14 Apr 2016 18:02:45

import numpy
import os
from datetime import datetime
import bob.measure
import argparse

# read scores of evaluation and test set
def read_scores(scores_path, database, part, experiment):
    path = os.path.join(scores_path, database + "_" + experiment.replace("_","_" + part + "_"))
    impostor_eval_file = path + "_eval_impostor.txt"
    genuine_eval_file = path + "_eval_genuine.txt"
    impostor_test_file = path + "_test_impostor.txt"
    genuine_test_file = path + "_test_genuine.txt"

    imp_eval = numpy.loadtxt(impostor_eval_file)
    gen_eval = numpy.loadtxt(genuine_eval_file)
    imp_test = numpy.loadtxt(impostor_test_file)
    gen_test = numpy.loadtxt(genuine_test_file)

    return imp_eval, gen_eval, imp_test, gen_test

# compute the metrics of evaluation and test set
def compute(imp_eval, gen_eval, imp_test, gen_test):

    # Calculate the thresholds (evaluation set)
    T_faefre = bob.measure.eer_threshold(imp_eval, gen_eval)
    T_fae0 = bob.measure.far_threshold(imp_eval, gen_eval, 0.00)
    T_fre0 = bob.measure.frr_threshold(imp_eval, gen_eval, 0.00)

    # Calculate error rates in evaluation set
    far, frr = bob.measure.farfrr(imp_eval, gen_eval, T_faefre)
    eer = (far + frr) / 2 * 100
    far, frr = bob.measure.farfrr(imp_eval, gen_eval, T_fre0)
    fa_fre0 = far * 100
    far, frr = bob.measure.farfrr(imp_eval, gen_eval, T_fae0)
    fr_fae0 = frr * 100
    
    # Calculate error rates in test set
    far, frr = bob.measure.farfrr(imp_test, gen_test, T_faefre)
    fa_test_eer = far * 100
    fr_test_eer = frr * 100
    ter_test_eer = (far + frr) * 100
    far, frr = bob.measure.farfrr(imp_test, gen_test, T_fre0)
    fa_test_fre0 = far * 100
    fr_test_fre0 = frr * 100
    ter_test_fre0 = (far + frr) * 100
    far, frr = bob.measure.farfrr(imp_test, gen_test, T_fae0)
    fa_test_fae0 = far * 100
    fr_test_fae0 = frr * 100
    ter_test_fae0 = (far + frr) * 100

    return eer, fa_fre0, fr_fae0, fa_test_eer, fr_test_eer, ter_test_eer, fa_test_fre0, fr_test_fre0, ter_test_fre0, fa_test_fae0, fr_test_fae0, ter_test_fae0

#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description='Calculate metrics using the score files')
parser.add_argument('experiment', default='', help='Experiment name (prefix of score files, e.g., database_facialpart_feature_classifier_)')
parser.add_argument('scores_path', default='', help='Directory with score files')

args = parser.parse_args()

if (not(os.path.exists(args.scores_path))):
    print('Score directory (\"' + args.scores_path + '\") not found.')
    exit()

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Metrics calculation started")
print("Experiment: " + args.experiment)
print("Score directory: " + args.scores_path)

# Get the path of the score files
eval_impostor_path = os.path.join(args.scores_path, args.experiment + '_eval_impostor.txt')
eval_genuine_path = os.path.join(args.scores_path, args.experiment + '_eval_genuine.txt')
test_impostor_path = os.path.join(args.scores_path, args.experiment + '_test_impostor.txt')
test_genuine_path = os.path.join(args.scores_path, args.experiment + '_test_genuine.txt')

if (not(os.path.exists(eval_genuine_path))):
    print('Score file of impostor in the evaluation set (\"' + args.eval_impostor_path + '\") not found.')
    exit()

if (not(os.path.exists(eval_genuine_path))):
    print('Score file of genuine in the evaluation set (\"' + args.eval_genuine_path + '\") not found.')
    exit()

if (not(os.path.exists(test_impostor_path))):
    print('Score file of impostor in the test set (\"' + args.test_impostor_path + '\") not found.')
    exit()

if (not(os.path.exists(test_genuine_path))):
    print('Score file of genuine in the test set (\"' + args.test_genuine_path + '\") not found.')
    exit()

# Load scores
eval_imp = numpy.loadtxt(eval_impostor_path)
eval_gen = numpy.loadtxt(eval_genuine_path)
test_imp = numpy.loadtxt(test_impostor_path)
test_gen = numpy.loadtxt(test_genuine_path)

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " Scores loaded")

# Calculate thresholds
T_faefre = bob.measure.eer_threshold(eval_imp, eval_gen)
T_fae0 = bob.measure.far_threshold(eval_imp,eval_gen, 0.00)
T_fre0 = bob.measure.frr_threshold(eval_imp,eval_gen, 0.00)

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " Thresholds calculated")

# Log the results
text = 'Experiment: ' + args.experiment
text += '\n\n>>Evaluation'
far, frr = bob.measure.farfrr(eval_imp, eval_gen, T_faefre)
text += '\n    FAE = FRE = ' + ("%.4f"%((far+frr)/2*100)) + '%'
far, frr = bob.measure.farfrr(eval_imp, eval_gen, T_fre0)
text += '\n    FAE (FRE = 0) = ' + ("%.4f"%(far*100)) + '%'
far, frr = bob.measure.farfrr(eval_imp, eval_gen, T_fae0)
text += '\n    FRE (FAE = 0) = ' + ("%.4f"%(frr*100)) + '%'
text += '\n\n>>Test'
far, frr = bob.measure.farfrr(test_imp, test_gen, T_faefre)
text += '\n    (FAE = FRE) - FA: ' + ("%.4f"%(far*100)) + '% | FR: ' + ("%.4f"%(frr*100)) + '% | TER = ' + ("%.4f"%((far+frr)*100)) + '%'
far, frr = bob.measure.farfrr(test_imp, test_gen, T_fre0)
text += '\n    (FRE = 0) - FA: ' + ("%.4f"%(far*100)) + '% | FR: ' + ("%.4f"%(frr*100)) + '% | TER = ' + ("%.4f"%((far+frr)*100)) + '%'
far, frr = bob.measure.farfrr(test_imp, test_gen, T_fae0)
text += '\n    (FAE = 0) - FA: ' + ("%.4f"%(far*100)) + '% | FR: ' + ("%.4f"%(frr*100)) + '% | TER = ' + ("%.4f"%((far+frr)*100)) + '%'

print(text)

file = open(args.experiment + "_results.txt", "wt")
file.write(text)
file.close()

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Metrics calculation finished")