#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <locale>
#include <string>
#include <chrono>
#include <functional>
#include <limits>

using namespace std;
//g++ -std=c++11 ImageTesting.cpp -o ImageTesting;./ImageTesting


#include "db.h"
#include "db_features.h"


#ifdef QT_BUILD
#include <QtCore>
#include <QDebug>
#undef cout
#define cout qDebug()
#define print_endl
#else
#define print_endl cout<<endl;
#endif

static int num_of_unreliable = 0;

class Classifier{
public:
    Classifier(string n):name(n){}
    virtual ~Classifier(){}

    virtual void train(std::vector<ImageInfo>* pDb){ pDbImages=pDb; }
    virtual int recognize(ImageInfo& testImageInfo)=0;
    string get_name(){ return name; }
private:
    string name;

protected:
    std::vector<ImageInfo>* pDbImages;
    static string build_name(string prefix, int param);
};
#include <sstream>
string Classifier::build_name(string prefix, int param){
    ostringstream os;
    os << prefix << ", " << param;
    return os.str();
}

//=================================
class BruteForceClassifier :public Classifier{
public:
    BruteForceClassifier(int max_feats = FEATURES_COUNT) :Classifier(Classifier::build_name("BF", max_feats)), max_features(max_feats){}
    int recognize(ImageInfo& testImageInfo){
        int bestInd= recognize_image_bf(*pDbImages, testImageInfo, max_features);
        int bestClassInd = -1;
        if (bestInd != -1)
            bestClassInd = (*pDbImages)[bestInd].classNo;
        return bestClassInd;
    }

private:
    int max_features;
};

//=================================
class ConventionalTWDClassifier :public Classifier{
public:
    enum class TWD_Type{ Posteriors, DistDiff, DistRatio };
    ConventionalTWDClassifier(int cls_num, TWD_Type t, double th, int feat_count = 64) :
        Classifier(build_name(t,th)), num_of_classes(cls_num), reduced_features_count(feat_count), threshold(th), type(t)
    {}
    int recognize(ImageInfo& testImageInfo);

private:
    int num_of_classes;
    int reduced_features_count;
    double threshold;
    TWD_Type type;
    static string build_name(TWD_Type type, double threshold);
};

string ConventionalTWDClassifier::build_name(TWD_Type type, double threshold){
    string prefix = "";
    switch (type){
    case TWD_Type::Posteriors:
        prefix = "TWD posteriors";
        break;
    case TWD_Type::DistDiff:
        prefix = "TWD diff";
        break;
    case TWD_Type::DistRatio:
        prefix = "TWD ratio";
        break;
    }
    ostringstream os;
    os << prefix << ", " << threshold;
    return os.str();
}

int ConventionalTWDClassifier::recognize(ImageInfo& testImageInfo){
    const vector<ImageInfo>& dbImages = *pDbImages;
    int bestInd = -1;
    double bestDist = 100000, secondBestDist = 100000;
    vector<double> distances(dbImages.size());
    const int DIST_WEIGHT = 100;
    vector<double> probabs(num_of_classes);
    double max_probab, probab;
    for (int j = 0; j < dbImages.size(); ++j){
        distances[j] = testImageInfo.distance(dbImages[j], 0, reduced_features_count);
        if (type == TWD_Type::Posteriors){
            probab = exp(-distances[j] * DIST_WEIGHT);
            if (probab>probabs[dbImages[j].classNo])
                probabs[dbImages[j].classNo] = probab;
        }
        if (distances[j] < bestDist){
            if (bestInd != -1 && dbImages[bestInd].classNo != dbImages[j].classNo)
                secondBestDist = bestDist;
            bestDist = distances[j];
            bestInd = j;
            if (type == TWD_Type::Posteriors){
                max_probab = probab;
            }
        }
    }

    bool is_reliable =
        //true;
        false;

    switch (type){
    case TWD_Type::Posteriors:
    {
        int MAX_PROBABS_COUNT = 5;//probabs.size();//10;
        nth_element(probabs.begin(), probabs.begin() + MAX_PROBABS_COUNT, probabs.end(), std::greater<double>());
        //std::sort(probabs.begin(),probabs.end(), std::greater<double>());
        double sum = 0;
        for (int i = 0; i < MAX_PROBABS_COUNT; ++i){
            sum += probabs[i];
        }
        max_probab /= sum;
        is_reliable = max_probab > threshold;
            //max_probab > 0.0008;//0.4 - 1000 weight
            //max_probab > 0.09;
            //max_probab > 0.0025;
            //max_probab > 0.24;
        //max_probab > 0.21;
    }
    break;
    case TWD_Type::DistDiff:
        //cout << (secondBestDist - bestDist) << endl;
        is_reliable = (secondBestDist - bestDist) > threshold;
        break;
    case TWD_Type::DistRatio:
        is_reliable = (bestDist / secondBestDist) < threshold;
        break;
    }
    if (!is_reliable){
        ++num_of_unreliable;
        bestInd = -1;
        bestDist = 100000;
        int last_feature =
            //FEATURES_COUNT;
            256;
        for (int j = 0; j < dbImages.size(); ++j){
            distances[j] = (distances[j] * reduced_features_count +
                testImageInfo.distance(dbImages[j], reduced_features_count, last_feature)*(last_feature - reduced_features_count)) / last_feature;
            if (distances[j] < bestDist){
                bestDist = distances[j];
                bestInd = j;
            }
        }
    }

    int bestClassInd = -1;
    if (bestInd != -1)
        bestClassInd = dbImages[bestInd].classNo;
    return bestClassInd;
}
//=================================
class ProposedTWDClassifier :public Classifier{
public:
    ProposedTWDClassifier(int cls_num, int feat_count, double th) :
        Classifier(build_name(feat_count,th)), num_of_classes(cls_num), reduced_features_count(feat_count), threshold(1.0/th)
    {}
    int recognize(ImageInfo& testImageInfo);
private:
    int num_of_classes,reduced_features_count;
    double threshold;

    static string build_name(int feat_count, double threshold);
};

string ProposedTWDClassifier::build_name(int feat_count, double threshold){
    ostringstream os;
    os << "Proposed TWD, " << feat_count<<", " << threshold;
    return os.str();
}
#define CHECK_ALL_INSTANCES
int ProposedTWDClassifier::recognize(ImageInfo& testImageInfo){
    const vector<ImageInfo>& dbImages = *pDbImages;
    int bestInd = -1;
    vector<double> distances(dbImages.size());
#ifdef CHECK_ALL_INSTANCES
    vector<int> instances_to_check(dbImages.size());
#else
    vector<int> instances_to_check(num_of_classes);
    vector<double> class_min_distances(num_of_classes);
#endif
    fill(instances_to_check.begin(), instances_to_check.end(), 1);

    //const FeaturesVector& lhs=testImageInfo.features;
    int last_feature =
        //FEATURES_COUNT;
        256;

    for (int cur_features = 0; cur_features<last_feature; cur_features += reduced_features_count){
        double bestDist = 100000;//, secondBestDist=100000;
#ifndef CHECK_ALL_INSTANCES
        fill(class_min_distances.begin(), class_min_distances.end(), numeric_limits<float>::max());
#endif
        for (int j = 0; j < dbImages.size(); ++j){
#ifdef CHECK_ALL_INSTANCES
            if (!instances_to_check[j])
#else
            if (!instances_to_check[dbImages[j].classNo])
#endif
                continue;
            /*const FeaturesVector& rhs=dbImages[j].features;
            float d=0;
            for(int fi=cur_features;fi<cur_features+reduced_features_count;++fi)
            d+=(lhs[fi]-rhs[fi])*(lhs[fi]-rhs[fi]);
            distances[j] += d;*/


            distances[j] += testImageInfo.distance(dbImages[j], cur_features, cur_features + reduced_features_count);
#ifndef CHECK_ALL_INSTANCES
            if (distances[j]<class_min_distances[dbImages[j].classNo])
                class_min_distances[dbImages[j].classNo] = distances[j];
#endif
            if (distances[j] < bestDist){
                bestDist = distances[j];
                bestInd = j;
            }
        }

        int num_of_variants = 0;
        //double threshold = 0.01*(cur_features + reduced_features_count);
        double dist_threshold = bestDist*threshold;// 1.43; // bestDist / 0.7;
#ifdef CHECK_ALL_INSTANCES
        ++num_of_variants;
        int bestClass = dbImages[bestInd].classNo;
        for (int j = 0; j<instances_to_check.size(); ++j){
            if (instances_to_check[j]){
                if (distances[j]>dist_threshold)
                    instances_to_check[j] = 0;
                else if (dbImages[j].classNo != bestClass)
                    ++num_of_variants;
            }
        }
#else
        for (int c = 0; c<instances_to_check.size(); ++c){
            if (instances_to_check[c]){
                if (class_min_distances[c]>dist_threshold)
                    instances_to_check[c] = 0;
                else
                    ++num_of_variants;
            }
        }
#endif
        if (num_of_variants == 1)
            break;
        if (cur_features == 0)
            ++num_of_unreliable;
    }

    int bestClassInd = -1;
    if (bestInd != -1)
        bestClassInd = dbImages[bestInd].classNo;
    return bestClassInd;
}
//=================================
#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;

static Mat get_training_mat(std::vector<ImageInfo>& db, int num_of_features=FEATURES_COUNT){
    Mat trainingDataMat(db.size(), num_of_features, CV_32FC1);

    for (int ind = 0; ind < db.size();++ind){
        for (int fi = 0; fi < num_of_features; ++fi){
            trainingDataMat.at<float>(ind, fi) = db[ind].features[fi];
        }
    }
    return trainingDataMat;
}
static Mat get_query_mat(ImageInfo& test, int num_of_features = FEATURES_COUNT){
    Mat queryDataMat(1, num_of_features, CV_32FC1);

    for (int fi = 0; fi < num_of_features; ++fi){
        queryDataMat.at<float>(0, fi) = test.features[fi];
    }
    return queryDataMat;
}
static Mat get_labels_mat(std::vector<ImageInfo>& db){
    Mat labelsMat(db.size(), 1, CV_32S);

    for (int ind = 0; ind < db.size(); ++ind){
        labelsMat.at<int>(ind, 0) = db[ind].classNo;
    }
    return labelsMat;
}

const int opencv_num_of_features = 256;// FEATURES_COUNT;
//256;
class SVMClassifier :public Classifier{
public:
    SVMClassifier() : Classifier("SVM")
    {
        opencvClassifier = SVM::create();
        opencvClassifier->setType(SVM::C_SVC);
        //opencvClassifier->setC(0.001);
        //opencvClassifier->setKernel(SVM::LINEAR);
        opencvClassifier->setKernel(SVM::RBF);
        //opencvClassifier->setGamma(1.0 / num_of_cont_features);
        opencvClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 100, 1e-6));

    }
    void train(std::vector<ImageInfo>* pDb){
        opencvClassifier->train(get_training_mat(*pDb, opencv_num_of_features), ROW_SAMPLE, get_labels_mat(*pDb));
    }
    int recognize(ImageInfo& testImageInfo){
        float response = opencvClassifier->predict(get_query_mat(testImageInfo, opencv_num_of_features));
        return (int)response;
    }
private:
    Ptr<SVM> opencvClassifier;
};

class RFClassifier :public Classifier{
public:
    RFClassifier(int num_of_classes) : Classifier("RF")
    {
        opencvClassifier = RTrees::create();
        opencvClassifier->setMaxDepth(opencv_num_of_features);
        opencvClassifier->setMaxCategories(num_of_classes);
        /*opencvClassifier->setMinSampleCount(2);
        opencvClassifier->setRegressionAccuracy(0);
        opencvClassifier->setUseSurrogates(false);
        opencvClassifier->setPriors(Mat());
        opencvClassifier->setCalculateVarImportance(false);
        opencvClassifier->setActiveVarCount(0);*/
        opencvClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 512, 1e-6));

    }
    void train(std::vector<ImageInfo>* pDb){
        opencvClassifier->train(get_training_mat(*pDb, opencv_num_of_features), ROW_SAMPLE, get_labels_mat(*pDb));
    }
    int recognize(ImageInfo& testImageInfo){
        float response = opencvClassifier->predict(get_query_mat(testImageInfo, opencv_num_of_features));
        return (int)response;
    }
private:
    Ptr<RTrees> opencvClassifier;
};


class MLPClassifier :public Classifier{
public:
    MLPClassifier(int num_of_cls, int num_of_feats = FEATURES_COUNT) : Classifier("MLP"), num_of_features(num_of_feats), num_of_classes(num_of_cls)
    {
        Mat layer_sizes = Mat(3, 1, CV_16U);
        layer_sizes.row(0) = Scalar(num_of_features);
        layer_sizes.row(1) = Scalar(256);
        layer_sizes.row(2) = Scalar(num_of_classes);

#if 0
        int method = ANN_MLP::BACKPROP;
        double method_param = 0.0001;
        int max_iter = 100;
#else
        int method = ANN_MLP::RPROP;
        double method_param = 0.1;
        int max_iter = 100;
#endif
        opencvClassifier = ANN_MLP::create();
        opencvClassifier->setLayerSizes(layer_sizes);
        opencvClassifier->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
        opencvClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, max_iter, 1e-6));
        opencvClassifier->setTrainMethod(method, method_param);
    }
    void train(std::vector<ImageInfo>* pDb){
        train_responses = Mat::zeros(pDb->size(), num_of_classes, CV_32F);

        // 1. unroll the responses
        for (int i = 0; i < pDb->size(); i++)
        {
            int cls_label = (*pDb)[i].classNo;
            train_responses.at<float>(i, cls_label) = 1.f;
        }
        training_mat = get_training_mat(*pDb, num_of_features);
        bool res = opencvClassifier->train(training_mat, ROW_SAMPLE, train_responses);
        /*cout << " train res=" << res << ' ' << num_of_classes << ' ' << train_responses.rows << ' ' << train_responses.cols <<
            ' ' << training_mat.rows << ' ' << training_mat.cols << ' ' << endl;*/
    }
    int recognize(ImageInfo& testImageInfo){
        //cv::Mat predictions;
        Mat query_mat = get_query_mat(testImageInfo, num_of_features);
        float response = opencvClassifier->predict(query_mat);
        //cout << response << endl;
        /*float maxPrediction = -1;
        response = -1;
        for (int i = 0; i < predictions.cols; i++)
        {
            float prediction = predictions.at<float>(i);
            //cout << i << ' ' << prediction << endl;
            if (prediction > maxPrediction)
            {
                maxPrediction = prediction;
                response = i;
            }
        }*/
        //cout << "final:" << response << ' ' << testImageInfo.classNo<<endl;
        return (int)response;
    }
private:
    Ptr<ANN_MLP> opencvClassifier;
    int num_of_features, num_of_classes;
    Mat train_responses, training_mat;
};
void testRecognitionMethod(ImagesDatabase& totalImages, Classifier* classifier){
    srand(13);
    cout << classifier->get_name().c_str();
    print_endl;
    int num_of_classes = totalImages.size();

    const int TESTS = 2;
    std::vector<ImageInfo> dbImages, testImages;
    double total_time = 0;
    double totalTestsErrorRate = 0, errorRateVar = 0, totalRecall = 0;
    for (int testCount = 0; testCount < TESTS; ++testCount){
        int errorsCount = 0;
        getTrainingAndTestImages(totalImages, dbImages, testImages);
        classifier->train(&dbImages);
        vector<int> testClassCount(num_of_classes), testErrorRates(num_of_classes);
        for (ImageInfo testImageInfo : testImages){
            ++testClassCount[testImageInfo.classNo];
        }
        num_of_unreliable = 0;

        auto t1 = chrono::high_resolution_clock::now();
        for (ImageInfo testImageInfo : testImages){
            int bestClassInd = classifier->recognize(testImageInfo);
            if (bestClassInd == -1 || testImageInfo.classNo != bestClassInd){
                ++errorsCount;
                ++testErrorRates[testImageInfo.classNo];
            }
        }
        auto t2 = chrono::high_resolution_clock::now();
        total_time += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / (1.0*testImages.size());

        double unreliable_ratio = 100.0*num_of_unreliable / testImages.size();

        double errorRate = 100.*errorsCount / testImages.size();
        double recall = 0;
        int num_of_test_classes = 0;
        for (int i = 0; i<num_of_classes; ++i){
            if (testClassCount[i]>0){
                recall += 100.*testErrorRates[i] / testClassCount[i];
                ++num_of_test_classes;
            }
        }

        recall /= num_of_test_classes;
        recall = 100 - recall;
        totalRecall += recall;
        //if (testCount == TESTS - 1)
        {
            cout << "test=" << testCount << " error=" << errorRate << " recall=" << recall << " unrel=" << unreliable_ratio << "% db=" << dbImages.size() <<
                " test=" << testImages.size();
            print_endl;
        }
        totalTestsErrorRate += errorRate;
        errorRateVar += errorRate * errorRate;

    }
    totalTestsErrorRate /= TESTS;
    total_time /= TESTS;
    totalRecall /= TESTS;
    errorRateVar = sqrt((errorRateVar - totalTestsErrorRate * totalTestsErrorRate * TESTS) / (TESTS - 1));
    cout << "Avg error=" << totalTestsErrorRate << " Sigma=" << errorRateVar << " recall=" << totalRecall << " time(ms)=" << total_time;
    print_endl;
}

void testRecognition(){
        ImagesDatabase totalImages;
        unordered_map<string, int> person2indexMap;
#if defined(USE_PCA) || 0
        ImagesDatabase orig_database;
        loadImages(orig_database,FEATURES_FILE_NAME,person2indexMap);
        extractPCA(orig_database, totalImages);
#elif 1
        ImagesDatabase orig_database;
        loadImages(totalImages,FEATURES_FILE_NAME,person2indexMap);

#else
        //loadImages(totalImages,FEATURES_FILE_NAME,person2indexMap);
        ImagesDatabase orig_database;
        loadImages(orig_database, FEATURES_FILE_NAME,person2indexMap);
        for (auto& features:orig_database){
            if (features.size() > 1)
                totalImages.push_back(features);
        }
        cout<<"total size="<<totalImages.size();
#endif
        int num_of_classes = totalImages.size();
        vector<Classifier*> classifiers;
        classifiers.push_back(new BruteForceClassifier());
        classifiers.push_back(new BruteForceClassifier(64));
    #ifndef USE_LCNN
        classifiers.push_back(new BruteForceClassifier(256));
    #endif
        classifiers.push_back(new ConventionalTWDClassifier(num_of_classes, ConventionalTWDClassifier::TWD_Type::Posteriors,0.24));
        classifiers.push_back(new ConventionalTWDClassifier(num_of_classes, ConventionalTWDClassifier::TWD_Type::DistDiff, 0.003));
        classifiers.push_back(new ConventionalTWDClassifier(num_of_classes, ConventionalTWDClassifier::TWD_Type::DistRatio, 0.7));
        classifiers.push_back(new ProposedTWDClassifier(num_of_classes,32,0.7));
        classifiers.push_back(new ProposedTWDClassifier(num_of_classes, 64, 0.7));
        classifiers.push_back(new RFClassifier(num_of_classes));
        classifiers.push_back(new SVMClassifier());
        classifiers.push_back(new MLPClassifier(num_of_classes, 256));// opencv_num_of_features));

        /*for (int feat_count = 1; feat_count < 256;feat_count*=2)
            classifiers.push_back(new ProposedTWDClassifier(num_of_classes, feat_count, 0.7));
        for (double threshold = 0.1; threshold < 1; threshold += 0.05)
            classifiers.push_back(new ProposedTWDClassifier(num_of_classes, 32, threshold));*/

        for (Classifier* classifier : classifiers){
            testRecognitionMethod(totalImages, classifier);
        }
}


#if 0
//#define USE_OUTER
void testVerification(){
    ImagesDatabase trainImages;
    unordered_map<string, int> person2indexMap;
    loadImages(trainImages,PCA_CASIA_FEATURES_FILE_NAME, person2indexMap,true);

    int withinCount=0, outerCount=0;
    for (auto& identity : trainImages){
        if(identity.size()>1)
            withinCount+=identity.size();
        outerCount+=identity.size();
    }
    Mat mat_within_features(withinCount, FEATURES_COUNT, CV_32F);
    int ind = 0;

    //bayesian faces
    for (auto& identity : trainImages){
        int identity_size=identity.size();
        if(identity_size>1){
            for (int i=0;i<identity_size;++i){
                int other_i=i;
                while(i==other_i)
                    other_i=rand()%identity_size;
                for (int j = 0; j < FEATURES_COUNT; ++j){
                    mat_within_features.at<float>(ind, j) =identity[i][j]-identity[other_i][j];
                }
                ++ind;
            }
        }
    }
    const int num_of_inout_features=96;
    PCA pca(mat_within_features, Mat(), CV_PCA_DATA_AS_ROW, num_of_inout_features);
    Mat mat_projection_result=
            pca.project(mat_within_features);
            //mat_within_features;
    cout << "rows="<<mat_projection_result.rows << " cols=" << mat_projection_result.cols;
    cout<<"pca smallest EV="<<pca.eigenvalues.at<float>(0,pca.eigenvalues.rows-1)<<' '<<pca.eigenvalues.at<float>(0,0);
    Mat covar;
    mulTransposed(mat_projection_result,covar,true);
    //covar=mat_projection_result.t()*mat_projection_result;
    covar/=withinCount;
    covar+=Mat::eye(covar.cols,covar.rows,CV_32F)*0.9;
    cout<<"det="<<determinant(covar);
    Mat inv_covar=covar.inv();
#ifdef USE_OUTER
    inv_covar/=sqrt(determinant(covar));
#endif
    cout << "inv_covar rows="<<inv_covar.rows << " cols=" << inv_covar.cols;

#ifdef USE_OUTER
    ind=0;
    Mat mat_outer_features(outerCount, FEATURES_COUNT, CV_32F);
    for (int train_ind=0;train_ind< trainImages.size();++train_ind){
        auto& identity = trainImages[train_ind];
        int identity_size=identity.size();
        for (int i=0;i<identity_size;++i){
            int other_ind=train_ind;
            while(train_ind==other_ind)
                other_ind=rand()%trainImages.size();
            int other_i=rand()%trainImages[other_ind].size();
            auto& other_identity = trainImages[other_ind];

            for (int j = 0; j < FEATURES_COUNT; ++j){
                mat_outer_features.at<float>(ind, j) =identity[i][j]-other_identity[other_i][j];
            }
            ++ind;
        }
    }
    PCA pca_outer(mat_outer_features, Mat(), CV_PCA_DATA_AS_ROW, num_of_inout_features);
    mat_projection_result=pca_outer.project(mat_outer_features);
    mulTransposed(mat_projection_result,covar,true);
    covar/=outerCount;
    covar+=Mat::eye(covar.cols,covar.rows,CV_32F)*0.9;
    cout<<"det1="<<determinant(covar);
    Mat outer_inv_covar=covar.inv()/sqrt(determinant(covar));
    cout << "outer_inv_covar rows="<<outer_inv_covar.rows << " cols=" << outer_inv_covar.cols;
#endif

    ImagesDatabase origImages, totalImages;
    loadImages(origImages,FEATURES_FILE_NAME,person2indexMap);
    totalImages.resize(origImages.size());

    ind=0;
    Mat image_features(1, FEATURES_COUNT, CV_32F);
    for (auto& identity : origImages){
        totalImages[ind].resize(identity.size());
        for (int i=0;i<identity.size();++i){
            for (int j = 0; j < FEATURES_COUNT; ++j){
                image_features.at<float>(0, j) =identity[i][j];
            }
#ifdef USE_OUTER
            totalImages[ind][i].resize(2*num_of_inout_features);
#else
            totalImages[ind][i].resize(num_of_inout_features);
#endif
            Mat projection_image_features=
                    pca.project(image_features);
                    //image_features;
            for (int j = 0; j < num_of_inout_features; ++j){
                totalImages[ind][i][j]=projection_image_features.at<float>(0, j);
            }
#ifdef USE_OUTER
            projection_image_features=pca_outer.project(image_features);
            for (int j = 0; j < num_of_inout_features; ++j){
                totalImages[ind][i][num_of_inout_features+j]=projection_image_features.at<float>(0, j);
            }
#endif
        }
        ++ind;
    }
    const int TESTS = 1;
    std::vector<ImageInfo> dbImages, testImages;
    double total_time = 0;
    double totalTestsErrorRate = 0, errorRateVar = 0;
    for (int testCount = 0; testCount < TESTS; ++testCount)
    {
            int errorsCount = 0;
            getTrainingAndTestImages(totalImages, dbImages, testImages);

            auto t1 = chrono::high_resolution_clock::now();
            for (ImageInfo testImageInfo: testImages){
                int bestInd = -1;
                double bestDist = 100000;
                for (int j = 0; j < dbImages.size(); ++j){
                    double dist = 0;
                    for(int f1=0;f1<num_of_inout_features;++f1){
                        double diff_1=testImageInfo.features[f1]-dbImages[j].features[f1];
#ifdef USE_OUTER
                        double out_diff_1=testImageInfo.features[num_of_inout_features+f1]-
                                dbImages[j].features[num_of_inout_features+f1];
#endif
                        for(int f2=0;f2<num_of_inout_features;++f2){
                            double diff_2=testImageInfo.features[f2]-dbImages[j].features[f2];
                            double d1=inv_covar.at<float>(f1,f2)*diff_1*diff_2;
                            dist+=d1;
#ifdef USE_OUTER
                            double out_diff_2=testImageInfo.features[num_of_inout_features+f2]-
                                    dbImages[j].features[num_of_inout_features+f2];
                            double d2=outer_inv_covar.at<float>(f1,f2)*out_diff_1*out_diff_2;
                            //cout<<diff_1<<' '<<diff_2<<' '<<d1<<' '<<d2;
                            dist-=d2;
#endif
                        }
                    }
                    if (dist < bestDist){
                            bestDist = dist;
                            bestInd = j;
                    }
                }
                if (bestInd == -1 || testImageInfo.classNo != dbImages[bestInd].classNo){
                    ++errorsCount;
                }
            }
            auto t2 = chrono::high_resolution_clock::now();
            double rec_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/(1.0*testImages.size());
            total_time += rec_time;
            double errorRate = 100.*errorsCount / testImages.size();
            cout << "test=" << testCount << " error=" << errorRate << " rec_time(ms)=" << rec_time<<" dbSize=" << dbImages.size() <<" testSize=" << testImages.size();
            print_endl;
    }
}
#else //joint bayesian
void testVerification(){
    const int num_of_inout_features=256;//FEATURES_COUNT;
    ImagesDatabase totalImages;
    unordered_map<string, int> person2indexMap;
    int total_images_size=loadImages(totalImages,FEATURES_FILE_NAME,person2indexMap);
#if 0
    ImagesDatabase trainImages;
    loadImages(trainImages,PCA_CASIA_FEATURES_FILE_NAME, person2indexMap,true);

    int n = num_of_inout_features;
    Mat u(0,n, CV_64F);
    Mat SW(n, n, CV_64F);
    int within_count = 0;
    for (auto& identity : trainImages){
        int cur_size = identity.size();
        Mat cur(cur_size, n, CV_64F);
        for (int i = 0; i < cur_size; ++i)
            for (int j = 0; j < n; ++j)
                cur.at<double>(i, j) = identity[i][j];

        Mat cov, mu;
        calcCovarMatrix(cur, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
        //cout << identity.first<<' '<<u.rows<<' '<<u.cols<<' '<<cov.rows << ' ' << cov.cols << ' ' << mu.rows << ' ' << mu.cols << '\n';
        u.push_back(mu);

        if (cur_size > 1){
            //cov = cov / (cur.rows - 1);
            within_count+=cur_size;
            cov/= cur_size - 1;
            cov+=Mat::eye(cov.cols,cov.rows,CV_64F)*0.5;
            SW += cov*cur_size/within_count;
        }
    }
    cout<<"u size="<<u.rows<<' '<<u.cols;
    Mat SU, mean_img;
    calcCovarMatrix(u, SU, mean_img, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    SU /= (u.rows - 1);
    SU+=Mat::eye(SU.cols,SU.rows,CV_64F)*0.5;

    Mat F = SW.inv();
    cout << "after all:" << within_count << ' ' <<  SU.at<double>(0, 0) << ' ' << SU.at<double>(1, 0) << ' ' <<  SW.at<double>(0, 0) << ' ' <<  SW.at<double>(1, 0);
    cout << "f:" << F.at<double>(0, 0) << ' ' << F.at<double>(1, 0) << ' ' << F.at<double>(0, 1) << ' ' << F.at<double>(1, 1) << '\n';
    Mat G = -(2 * SU + SW).inv() * SU*F;
    cout << "g:" << G.at<double>(0, 0) << ' ' << G.at<double>(1, 0) << ' ' << G.at<double>(0, 1) << ' ' << G.at<double>(1, 1);
    Mat A = (SU + SW).inv() - (F + G);


    cout<<"A="<<A.at<double>(0,0)<<' '<<A.at<double>(A.rows-1,A.cols-1);
    cout<<"G="<<G.at<double>(0,0)<<' '<<G.at<double>(G.rows-1,G.cols-1);
    cout<<"F="<<F.at<double>(0,0)<<' '<<F.at<double>(F.rows-1,F.cols-1);

    vector<double> xax(total_images_size);
    int ind=0;
    for(auto& identity : totalImages){
        for(auto& features:identity){
            xax[ind]=0;
            for (int i = 0; i < num_of_inout_features; ++i)
                for (int j = 0; j < num_of_inout_features; ++j)
                    xax[ind] += A.at<double>(i,j) * features[i] * features[j];
            ++ind;
        }
    }

#endif
    const int TESTS = 10;
    std::vector<ImageInfo> dbImages, testImages;
    double total_time = 0;
    double totalTestsErrorRate = 0, errorRateVar = 0;
    unordered_map<int,double> image_distances;
    cout<<"start testing";
    print_endl;
    for (int testCount = 0; testCount < TESTS; ++testCount)
    {
            getTrainingAndTestImages(totalImages, dbImages, testImages);

            int errorsCount = 0;
            auto t1 = chrono::high_resolution_clock::now();
            for (ImageInfo testImageInfo: testImages){
                int bestInd = -1;
                double bestDist = 100000;
                for (int j = 0; j < dbImages.size(); ++j){
                    double dist = 0;
                    int pair_ind;
                    if(testImageInfo.indexInDatabase<dbImages[j].indexInDatabase)
                        pair_ind=testImageInfo.indexInDatabase*total_images_size+dbImages[j].indexInDatabase;
                    else
                        pair_ind=testImageInfo.indexInDatabase+total_images_size*dbImages[j].indexInDatabase;

                    if(image_distances.find(pair_ind)==image_distances.end())
                    {
#if 0
                        dist = -xax[testImageInfo.indexInDatabase]-xax[dbImages[j].indexInDatabase];
                        for(int f1=0;f1<num_of_inout_features;++f1){
                            for(int f2=0;f2<num_of_inout_features;++f2){
                                dist+=2*G.at<double>(f1,f2)*testImageInfo.features[f1]*dbImages[j].features[f2];
                            }
                        }
#else
                        for(int f1=0;f1<num_of_inout_features;++f1){
                            dist+=(testImageInfo.features[f1]-dbImages[j].features[f1])*(testImageInfo.features[f1]-dbImages[j].features[f1]);
                        }
#endif
                        image_distances.insert(make_pair(pair_ind,dist));
                    }
                    dist=image_distances[pair_ind];
                    if (dist < bestDist){
                            bestDist = dist;
                            bestInd = j;
                    }
                }
                if (bestInd == -1 || testImageInfo.classNo != dbImages[bestInd].classNo){
                    ++errorsCount;
                }
            }
            auto t2 = chrono::high_resolution_clock::now();
            double rec_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()/(1.0*testImages.size());
            total_time += rec_time;
            double errorRate = 100.*errorsCount / testImages.size();
            cout << "test=" << testCount << " error=" << errorRate << " rec_time(ms)=" << rec_time<<" dbSize=" << dbImages.size() <<" testSize=" << testImages.size();
            print_endl;

            totalTestsErrorRate += errorRate;
            errorRateVar += errorRate * errorRate;
    }
    totalTestsErrorRate /= TESTS;
    total_time /= TESTS;
    errorRateVar = sqrt((errorRateVar - totalTestsErrorRate * totalTestsErrorRate * TESTS) / (TESTS - 1));
    cout << "Avg error=" << totalTestsErrorRate << " Sigma=" << errorRateVar;
    print_endl;
}
#endif

