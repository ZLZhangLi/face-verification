import numpy as np
import os
import skimage
import sys
import sklearn.metrics.pairwise as pw
import math
if __name__ == '__main__':
    thershold = 0
    accuracy = 0
    thld = 0
    TEST_SUM = 6000
    POSITIVE_SIMILARIY = "positive_similarity_vgg16.txt"
    NEGATIVE_SIMILARIY = "negative_similarity_vgg16.txt"
    for thershold in np.arange(0.15, 0.90, 0.01):
        True_Positive = 0
        True_Negative = 0
        False_Positive = 0
        False_Negative = 0
        fp = open(POSITIVE_SIMILARIY, 'r')
        lines1 = fp.readlines()
        for line1 in lines1:
            #print type(float(line1))
            if float(line1) >= thershold:
                #  print 'Same Guy\n\n'
                True_Positive += 1
            else:
                #  wrong
                False_Positive += 1
        fn = open(NEGATIVE_SIMILARIY, 'r')
        lines2 = fn.readlines()
        for line2 in lines2:
            if float(line2) >= thershold:
                #  print 'Wrong Guy\n\n'
                #  wrong
                False_Negative += 1
            else:
                #  correct
                True_Negative += 1

        print "thershold: " + str(thershold)
        print "Accuracy: " + str(float(True_Positive + True_Negative) / TEST_SUM) + " %"
        print "True_Positive: " + str(float(True_Positive) / TEST_SUM) + " %"
        print "True_Negative: " + str(float(True_Negative) / TEST_SUM) + " %"
        print "False_Positive: " + str(float(False_Positive) / TEST_SUM) + " %"
        print "False_Negative: " + str(float(False_Negative) / TEST_SUM) + " %"

    	if accuracy < float(True_Positive + True_Negative) / TEST_SUM:
    	    accuracy = float(True_Positive + True_Negative) / TEST_SUM
    	    thld = thershold
    print 'Best performance: %f, with threshold %f ' % (accuracy, thld)
