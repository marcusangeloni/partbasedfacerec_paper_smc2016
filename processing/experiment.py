#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus@liv.ic.unicamp.br>
# Thu 14 Apr 2016 18:02:45

import numpy
import os
from datetime import datetime
import argparse
import metrics

#################
# main block
#################

# Get arguments
parser = argparse.ArgumentParser(description='Calculate experiment metrics using the score files (database_part_experiment)')
parser.add_argument('database', default='', help='Database name (prefix of score files)')
parser.add_argument('part', default='', help='Facial part')
parser.add_argument('experiment', default='', help='Experiment, i.e., feature_classifier')
parser.add_argument('scores_path', default='', help='Directory with score files')

args = parser.parse_args()

if (not(os.path.exists(args.scores_path))):
    print('Score directory (\"' + args.scores_path + '\") not found.')
    exit()

experiment_name = args.database + "_" + args.part + "_" + args.experiment

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Metrics calculation started")
print("Experiment: " + experiment_name)
print("Score directory: " + args.scores_path)

# Load scores
imp_eval, gen_eval, imp_test, gen_test = metrics.read_scores(args.scores_path, args.database, args.part, args.experiment)
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " Scores loaded")

# Compute metrics
eer, fa_fre0, fr_fae0, fa_test_eer, fr_test_eer, ter_test_eer, fa_test_fre0, fr_test_fre0, ter_test_fre0, fa_test_fae0, fr_test_fae0, ter_test_fae0 = metrics.compute(imp_eval, gen_eval, imp_test, gen_test)
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " Metrics calculated")

# Log the results
text = "Hour,EER,FAE(FRE=0),FRE(FAE=0),FA_test_EER,FR_test_EER,TER_test_EER,FA_test_FRE0,FR_test_FRE0,TER_test_FRE0,FA_test_FAE0,FR_test_FAE0,TER_test_FAE0\n"
text += datetime.now().strftime('%d/%m/%Y %H:%M:%S') + ',' + ("%.4f"%(eer)) + '%,' + ("%.4f"%(fa_fre0)) + '%,'+ ("%.4f"%(fr_fae0)) + '%,' + ("%.4f"%(fa_test_eer)) + '%,' + ("%.4f"%(fr_test_eer)) + '%,' + ("%.4f"%(ter_test_eer)) + '%,' + ("%.4f"%(fa_test_fre0)) + '%,' + ("%.4f"%(fr_test_fre0)) + '%,' + ("%.4f"%(ter_test_fre0)) + '%,' + ("%.4f"%(fa_test_fae0)) + '%,' + ("%.4f"%(fr_test_fae0)) + '%,' + ("%.4f"%(ter_test_fae0)) + "\n"
print(text)

file = open(experiment_name + "_results.csv", "wt")
file.write(text)
file.close()

print(datetime.now().strftime('%d/%m/%Y %H:%M:%S') + " - Metrics calculation finished")