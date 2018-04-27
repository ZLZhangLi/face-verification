import numpy as np
import os
import skimage
import sys
import caffe
import sklearn.metrics.pairwise as pw
import math

#  sys.path.insert(0, '/Downloads/caffe-master/python');
#  load Caffe model

caffe.set_mode_cpu()

global net
net = caffe.Classifier('deploy2.prototxt', 'deepid1_55_47_iter_1000000.caffemodel')


def compare_pic(feature1, feature2):
    predicts = pw.cosine_similarity(feature1, feature2)
    return predicts


def get_feature(path):
    global net
    X = read_image(path)
    # test_num = np.shape(X)[0];
    # print test_num;
    out = net.forward_all(data=X)
    feature = np.float64(out['deepid'])
    feature = np.reshape(feature, (1, 160))
    return feature


def read_image(filepath):
    averageImg = [129.1863, 104.7624, 93.5940]
    X = np.empty((1, 3, 55, 47))
    filename = filepath.split('\n')
    filename = filename[0]
    im = skimage.io.imread(filename, as_grey=False)
    image = skimage.transform.resize(im, (55, 47)) * 255
    #mean_blob.shape = (-1, 1);
    #mean = np.sum(mean_blob) / len(mean_blob);
    X[0, 0, :, :] = image[:, :, 0] - averageImg[0]
    X[0, 1, :, :] = image[:, :, 1] - averageImg[1]
    X[0, 2, :, :] = image[:, :, 2] - averageImg[2]
    return X

# Iterate file system.
def saveFileInfo(file_list, output_path):
    #print "Writing file info to", output_path
    with open(output_path, mode = 'a') as f:
        for filenames in file_list:
            for item in filenames:
                line = ''.join(str(item)) + '\n'
                f.write(line)
        f.close()

if __name__ == '__main__':
    thershold = 0
    accuracy = 0
    thld = 0
    TEST_SUM = 6000
    #DATA_BASE = "F:/zhangli_code/lfw-cropped/"
    DATA_BASE = "../dataset/lfw-aligned/"
    #MEAN_FILE = 'mean.proto'
    POSITIVE_TEST_FILE = "positive_pairs_path.txt"
    NEGATIVE_TEST_FILE = "negative_pairs_path.txt"
    POSITIVE_SIMILARIY = "positive_similarity_1000000_dropout.txt"
    NEGATIVE_SIMILARIY = "negative_similarity_1000000_dropout.txt"
    #mean_blob = caffe.proto.caffe_pb2.BlobProto()
    #mean_blob.ParseFromString(open(MEAN_FILE, 'rb').read())
    #mean_npy = caffe.io.blobproto_to_array(mean_blob)

    #print mean_npy.shape


    # Positive Test
    f_positive = open(POSITIVE_TEST_FILE, "r")
    PositiveDataList = f_positive.readlines()
    #print PositiveDataList       
    #print '@@@@@@@@@'         
    f_positive.close()
    f_negative = open(NEGATIVE_TEST_FILE, "r")
    NegativeDataList = f_negative.readlines()
    f_negative.close()
    for index in range(len(PositiveDataList)):
        filepath_1 = PositiveDataList[index].split(' ')[0]
        filepath_2 = PositiveDataList[index].split(' ')[1][:-2]
        #print filepath_1
        #print '**********'
        #print filepath_2
        #print '^^^^^^^^^^'
        feature_1 = get_feature(DATA_BASE + filepath_1)
        feature_2 = get_feature(DATA_BASE + filepath_2)
        #print feature_1
        result_p = compare_pic(feature_1, feature_2)
        print result_p
        #print '+++++++++'
        saveFileInfo(result_p.tolist(),POSITIVE_SIMILARIY)


    for index in range(len(NegativeDataList)):
        filepath_1 = NegativeDataList[index].split(' ')[0]
        filepath_2 = NegativeDataList[index].split(' ')[1][:-2]
        feature_1 = get_feature(DATA_BASE + filepath_1)
        feature_2 = get_feature(DATA_BASE + filepath_2)
        result_n = compare_pic(feature_1, feature_2)
        print result_n
        #print '---------'
        saveFileInfo(result_n.tolist(),NEGATIVE_SIMILARIY)

    for thershold in np.arange(0.15, 0.95, 0.02):
        True_Positive = 0
        True_Negative = 0
        False_Positive = 0
        False_Negative = 0
        fp = open(POSITIVE_SIMILARIY, 'r')
        lines1 = fp.readlines()
        for line1 in lines1:
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

