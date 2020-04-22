#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Marcus de Assis Angeloni <marcus@liv.ic.unicamp.br>
# Thu 14 Apr 2016 18:02:45

import numpy
import os
import bob.measure

# read scores of evaluation and test set
def read_scores(scores_path, database, part, experiment):
    path = os.path.join(scores_path, database + "_" + part + "_" + experiment)
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

    print(max(imp_eval))
    print(min(gen_eval))

    print("marcus")
    print(T_faefre)
    print(T_fae0)
    print(T_fre0)

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