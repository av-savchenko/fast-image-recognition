#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <set>
#include <chrono>
using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;

#include "db.h"
#ifdef QT_BUILD
#include <QtCore>
#include <QDebug>
#undef cout
#define cout qDebug()
#define print_endl
#else
#define print_endl cout<<endl;
#endif


typedef double FEATURE_TYPE;

#define NO_PCA_FEATURES 256 //1536//FEATURES_COUNT //0//256, 0, 128

class Feature_vector{
public:
    Feature_vector(const vector<FEATURE_TYPE>& fv, double out):
      features(fv),output(out)
    {
    }
    vector<FEATURE_TYPE> features;
    double output;
};
ostream& operator<<(ostream& os, const Feature_vector& feature){
    os<<"cn="<<feature.output<<": ";
    for(size_t i=0;i<feature.features.size();++i){
        os<<feature.features[i]<<' ';
    }
    return os;
}


namespace{
    size_t num_of_classes, num_of_cont_features;
    size_t num_of_cont_features_orig;

    vector<Feature_vector> dataset, tmp_dataset;
    vector <vector<size_t> > indices;
    vector<vector<size_t> > training_set;
    vector<size_t> test_set;
    vector<FEATURE_TYPE> minValues, maxValues, avgValues, stdValues;
}

inline float fasterlog2 (float x)
{
    union { float f; uint32_t i; } vx = { x };
      union { uint32_t i; float f; } mx = { (vx.i & 0x007FFFFF) | (0x7e << 23) };
      float y = vx.i;
      y *= 1.0 / (1 << 23);

      return
        y - 124.22544637f - 1.498030302f * mx.f - 1.72587999f / (0.3520887068f + mx.f);
}

#if 1
#define fastlog fasterlog2
#else
#define fastlog log10
#endif

//=========================================
class Classifier{
public:
    Classifier(std::string name):method_name(name)
    {
    }
    virtual ~Classifier(){}
    virtual void train()=0;
    virtual int predict(const Feature_vector& inputFeatures)=0;
    std::string get_name(){return method_name;}
protected:
    std::string method_name;

    virtual FEATURE_TYPE normalize(const Feature_vector& feature_vector, size_t fi);
};

#include <sstream>
template<typename T> string to_string(string prefix, T param){
    ostringstream os;
    os << prefix << ", " << param;
    return os.str();
}
FEATURE_TYPE Classifier::normalize(const Feature_vector& feature_vector, size_t fi){
    return (feature_vector.features[fi]-avgValues[fi]);
}

//=====================================================
class KNNClassifier: public Classifier{
public:
    KNNClassifier(int k):Classifier(to_string("k-NN",k)),K(k) {}
    void train(){}
    int predict(const Feature_vector& inputFeatures);
private:
    int K;
};
int KNNClassifier::predict(const Feature_vector& inputFeatures){
    vector<float> outputs(num_of_classes,0);
    size_t total_training_size=dataset.size()-test_set.size();
    vector<FEATURE_TYPE> distances(total_training_size);
    vector<size_t> idx(total_training_size), class_label(total_training_size);
    size_t cur_ind = 0;

    for(size_t i = 0; i < num_of_classes; ++i){
        outputs[i] = 0;
        for(size_t t = 0; t < training_set[i].size(); ++t){
            FEATURE_TYPE dist = 0;
            for(size_t fi = 0; fi < num_of_cont_features; ++fi){
                size_t training_ind = training_set[i][t];
                //FEATURE_TYPE dist=(dataset[training_ind].features[fi]-dataset[test_ind].features[fi])*(dataset[training_ind].features[fi]-dataset[test_ind].features[fi]);
                FEATURE_TYPE diff = 0,sum=0;
                {
                    diff=normalize(tmp_dataset[training_ind],fi);
                    //sum+=val;
                }
                {
                    FEATURE_TYPE val=normalize(inputFeatures,fi);
                    diff -= val;
                    //sum+=val;
                }
                //if(sum>0) dist += diff*diff/sum;
                dist += diff*diff;
            }
            dist/=num_of_cont_features;
            distances[cur_ind] = dist;
            idx[cur_ind] = cur_ind;
            class_label[cur_ind] = i;
            ++cur_ind;
        }
    }

    std::sort(idx.begin(), idx.end(),
        [&distances](size_t i1, size_t i2) {return distances[i1] < distances[i2]; });
    //cout << distances[idx[0]] << '\n';
    for(size_t i = 0; i < total_training_size; ++i){
        size_t c = class_label[idx[i]];
        ++outputs[c];
        if (outputs[c] >= K){
            break;
        }
    }
    float max_output=-DBL_MAX;
    int bestClass=-1;
    for(size_t i=0;i<num_of_classes;++i){
        if(max_output<outputs[i]){
            max_output=outputs[i];
            bestClass=i;
        }
    }
    return bestClass;
}

//=====================================================
class PNNClassifier: public Classifier{
public:
    PNNClassifier(bool bf=true,std::string name="PNN"):Classifier(name+(bf?"":" (seq)")),bruteforce(bf){}
    void train(){}
    int predict(const Feature_vector& inputFeatures);
protected:
    virtual int predict_bf(const Feature_vector& inputFeatures);
    virtual int predict_sequentional(const Feature_vector& inputFeatures);

    static const size_t delta_features_count=32;
private:
    bool bruteforce;
    static const int output_dividor=1E9;
};

int PNNClassifier::predict_bf(const Feature_vector& inputFeatures){
    size_t total_training_size=dataset.size()-test_set.size();
    double var=0.00002;
    //var*=10;
    if(num_of_cont_features>2000)
        var/=10;
    vector<double> outputs(num_of_classes,0);
    for(size_t i=0;i<num_of_classes;++i){
        double den=total_training_size;

        for(size_t t=0;t<training_set[i].size();++t){
            size_t training_ind=training_set[i][t];
            FEATURE_TYPE dist=0;
            for(size_t fi=0;fi<num_of_cont_features;++fi){
                //dist+=(dataset[training_ind].features[fi]-dataset[test_ind].features[fi])*(dataset[training_ind].features[fi]-dataset[test_ind].features[fi]);
                FEATURE_TYPE diff=0;
                {
                diff=normalize(tmp_dataset[training_ind],fi);
                }
                {
                FEATURE_TYPE val=normalize(inputFeatures,fi);
                diff-=val;
                }
                dist+=diff*diff;
            }
            outputs[i]+=exp(-dist/(2*num_of_cont_features*var));
        }
        outputs[i] /= den;
    }
    double max_output=-DBL_MAX;
    int bestClass=-1;
    for(size_t i=0;i<num_of_classes;++i){
        if(max_output<outputs[i]){
            max_output=outputs[i];
            bestClass=i;
        }
    }
    return bestClass;
}

int PNNClassifier::predict_sequentional(const Feature_vector& inputFeatures){
    size_t total_training_size=dataset.size()-test_set.size();
    double var=0.00002;
    //var*=10;
    if(num_of_cont_features>2000)
        var/=10;
    int bestClass=-1;

    vector<size_t> classes_to_check(num_of_classes,1);
    vector<vector<FEATURE_TYPE>> distances(num_of_classes);
    for(size_t i=0;i<num_of_classes;++i){
        distances[i].resize(training_set[i].size());
    }
    vector<double> outputs(num_of_classes,0);
    size_t max_features=num_of_cont_features;
    double den=total_training_size;
    //size_t max_features=256;
    for(size_t cur_features = 0; cur_features<max_features; cur_features += delta_features_count){
        size_t max_fi=cur_features+delta_features_count;
        if(max_fi>num_of_cont_features)
            max_fi=num_of_cont_features;
        for(size_t i=0;i<num_of_classes;++i){
            if(classes_to_check[i]){
                outputs[i]=0;
                for(size_t t=0;t<training_set[i].size();++t){
                    size_t training_ind=training_set[i][t];
                    for(size_t fi=cur_features;fi<max_fi;++fi){
                        //dist+=(dataset[training_ind].features[fi]-dataset[test_ind].features[fi])*(dataset[training_ind].features[fi]-dataset[test_ind].features[fi]);
                        FEATURE_TYPE diff=0;
                        {
                        diff=normalize(tmp_dataset[training_ind],fi);
                        }
                        {
                        FEATURE_TYPE val=normalize(inputFeatures,fi);
                        diff-=val;
                        }
                        distances[i][t]+=diff*diff;
                    }
                    outputs[i]+=exp(-distances[i][t]/(2*var*max_fi));
                }
                outputs[i]=outputs[i]/den;
            }
        }
        double max_output=-DBL_MAX;
        for(size_t i=0;i<num_of_classes;++i){
            if(classes_to_check[i]){
                if(max_output<outputs[i]){
                    max_output=outputs[i];
                    bestClass=i;
                }
            }
        }

        int num_of_variants = 0;
        float output_threshold = max_output/output_dividor;
        for(size_t i=0;i<num_of_classes;++i){
            if(classes_to_check[i]){
                if (outputs[i]<output_threshold)
                    classes_to_check[i] = 0;
                else
                    ++num_of_variants;
            }
        }
        if (num_of_variants == 1)
            break;
    }
    return bestClass;
}

int PNNClassifier::predict(const Feature_vector& inputFeatures)
{
    int bestClass;
    if(bruteforce){
        bestClass=predict_bf(inputFeatures);
    }
    else{
        bestClass=predict_sequentional(inputFeatures);
    }
    return bestClass;
}


//=====================================================
class PNNwithClusteringClassifier: public Classifier{
public:
    PNNwithClusteringClassifier(int no_clusters):Classifier(to_string("PNN with clustering",no_clusters)),num_clusters(no_clusters) {}
    void train();
    int predict(const Feature_vector& inputFeatures);
private:
    const int num_clusters;
    vector<vector<size_t> > clustered_training_set;
};
void PNNwithClusteringClassifier::train(){
    clustered_training_set.clear();
    clustered_training_set.resize(num_of_classes);
    for(size_t i=0;i<num_of_classes;++i){
        size_t cur_size=training_set[i].size();
        if(cur_size>num_clusters){
            //qDebug()<<cur_size<<' '<<num_clusters;
            vector<size_t> bestClustIndices(training_set[i].size());
            vector<size_t> centroid_indices(num_clusters);
            for(size_t c=0;c<num_clusters;++c)
                centroid_indices[c]=c;
            for(size_t step = 0; step < 100; ++step){
                for(size_t t=0;t<training_set[i].size();++t){
                    bestClustIndices[t]=-1;
                    double bestDist=DBL_MAX;
                    for(size_t c=0;c<num_clusters;++c)
                        if(centroid_indices[c]>=0){
                            double dist=0;
                            size_t d1=training_set[i][centroid_indices[c]];
                            size_t d2=training_set[i][t];
                            for(size_t fi_12=0;fi_12<num_of_cont_features;++fi_12){
                                dist+=(dataset[d1].features[fi_12]-dataset[d2].features[fi_12])*(dataset[d1].features[fi_12]-dataset[d2].features[fi_12]);
                            }
                            dist/=num_of_cont_features;
                            if(dist<bestDist){
                                bestDist=dist;
                                bestClustIndices[t]=c;
                            }
                        }
                }
                for(size_t c=0;c<num_clusters;++c){
                    double bestClustDist=DBL_MAX;
                    centroid_indices[c]=-1;
                    for(size_t t=0;t<training_set[i].size();++t){
                        if(bestClustIndices[t]==c){
                            double clustDist = 0;
                            for(size_t t1=0;t1<training_set[i].size();++t1)
                                if(bestClustIndices[t1]==c){
                                    double dist=0;
                                    size_t d1=training_set[i][t];
                                    size_t d2=training_set[i][t1];
                                    for(size_t fi_12=0;fi_12<num_of_cont_features;++fi_12){
                                        dist+=(dataset[d1].features[fi_12]-dataset[d2].features[fi_12])*(dataset[d1].features[fi_12]-dataset[d2].features[fi_12]);
                                    }
                                    dist/=num_of_cont_features;
                                    clustDist+=dist;
                                }
                            if(clustDist<bestClustDist){
                                bestClustDist=clustDist;
                                centroid_indices[c]=t;
                            }
                        }
                    }
                }
            }
            for(size_t c=0;c<num_clusters;++c){
                if(centroid_indices[c]>=0){
                    clustered_training_set[i].push_back(training_set[i][centroid_indices[c]]);
                }
            }
        }
        else{
            for(size_t j=0;j<cur_size;++j){
                clustered_training_set[i].push_back(training_set[i][j]);
            }

        }
    }
}
int PNNwithClusteringClassifier::predict(const Feature_vector& inputFeatures){
    size_t total_training_size=dataset.size()-test_set.size();
    double var=0.00002;
    //var*=10;
    if(num_of_cont_features>2000)
        var/=10;
    vector<double> outputs(num_of_classes,0);
    for(size_t i=0;i<num_of_classes;++i){
        double den=total_training_size;

        for(size_t t=0;t<clustered_training_set[i].size();++t){
            size_t training_ind=clustered_training_set[i][t];
            FEATURE_TYPE dist=0;
            for(size_t fi=0;fi<num_of_cont_features;++fi){
                //dist+=(dataset[training_ind].features[fi]-dataset[test_ind].features[fi])*(dataset[training_ind].features[fi]-dataset[test_ind].features[fi]);
                FEATURE_TYPE diff=0;
                {
                diff=normalize(tmp_dataset[training_ind],fi);
                }
                {
                FEATURE_TYPE val=normalize(inputFeatures,fi);
                diff-=val;
                }
                dist+=diff*diff;
            }
            outputs[i]+=exp(-dist/(2*num_of_cont_features*var));
        }
        outputs[i] /= den;
        //cout<<i<<' '<<outputs[i];
    }
    double max_output=-DBL_MAX;
    int bestClass=-1;
    for(size_t i=0;i<num_of_classes;++i){
        if(max_output<outputs[i]){
            max_output=outputs[i];
            bestClass=i;
        }
    }
    return bestClass;
}

//=====================================================
inline TermCriteria TC(int iters, double eps)
{
    return TermCriteria(TermCriteria::MAX_ITER + (eps > 0 ? TermCriteria::EPS : 0), iters, eps);
}
class OpencvClassifier: public Classifier{
public:
    OpencvClassifier(std::string method):Classifier(method) {}
    void train();
    int predict(const Feature_vector& inputFeatures);
private:
    virtual StatModel* getClassifier()=0;
};
void OpencvClassifier::train(){
    size_t num_of_training_data=dataset.size()-test_set.size();
    Mat labelsMat(num_of_training_data, 1, CV_32S);
    Mat trainingDataMat(num_of_training_data, num_of_cont_features, CV_32FC1);
    int ind=0;
    for(size_t i=0;i<num_of_classes;++i){
        for(size_t t=0;t<training_set[i].size();++t){
            for(size_t fi=0;fi<num_of_cont_features;++fi){
                FEATURE_TYPE val=normalize(tmp_dataset[training_set[i][t]],fi);
                trainingDataMat.at<float>(ind,fi)=
                    val;
                    //dataset[training_set[i][t]].features[fi];
                    //(dataset[training_set[i][t]].features[fi]-avgValues[fi])/stdValues[fi];
                    //(dataset[training_set[i][t]].features[fi]-minValues[fi])/(maxValues[fi]-minValues[fi])*2-1;
            }
            labelsMat.at<int>(ind,0)=i;
            ++ind;
        }
    }

    getClassifier()->train(trainingDataMat, ROW_SAMPLE, labelsMat);
}
int OpencvClassifier::predict(const Feature_vector& inputFeatures){
    Mat queryMat(1, num_of_cont_features, CV_32FC1);
    for(size_t fi=0;fi<num_of_cont_features;++fi){
        FEATURE_TYPE val=normalize(inputFeatures,fi);
        queryMat.at<float>(0,fi)=
            val;
            //dataset[test_ind].features[fi];
            //(dataset[test_ind].features[fi]-avgValues[fi])/stdValues[fi];
            //(dataset[test_ind].features[fi]-minValues[fi])/(maxValues[fi]-minValues[fi])*2-1;
    }
    cv::Mat predictions;
    getClassifier()->predict(queryMat, predictions);
    float response = predictions.at<float>(0);
    return (int)response;
}

class OpencvSVMClassifier: public OpencvClassifier{
public:
    OpencvSVMClassifier(bool is_linear);
private:
    Ptr<SVM> opencvClassifier;
    StatModel* getClassifier(){return opencvClassifier;}
};
OpencvSVMClassifier::OpencvSVMClassifier(bool is_linear):OpencvClassifier(is_linear?"Linear SVM":"RBF SVM")
{
    // Set up SVM's parameters
    opencvClassifier = SVM::create();
    opencvClassifier->setType(SVM::C_SVC);
    //opencvClassifier->setC(0.001);
    opencvClassifier->setKernel(is_linear?SVM::LINEAR:SVM::RBF);
    opencvClassifier->setGamma(1.0 / num_of_cont_features);
    //opencvClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
}

class OpencvRTreeClassifier: public OpencvClassifier{
public:
    OpencvRTreeClassifier();
private:
    Ptr<RTrees> opencvClassifier;
    StatModel* getClassifier(){return opencvClassifier;}
};
OpencvRTreeClassifier::OpencvRTreeClassifier():OpencvClassifier("Rtree")
{
    opencvClassifier = RTrees::create();
    opencvClassifier->setMaxDepth(num_of_cont_features);
    opencvClassifier->setMaxCategories(num_of_classes);
    /*opencvClassifier->setMinSampleCount(2);
    opencvClassifier->setRegressionAccuracy(0);
    opencvClassifier->setUseSurrogates(false);
    opencvClassifier->setPriors(Mat());
    opencvClassifier->setCalculateVarImportance(false);
    opencvClassifier->setActiveVarCount(0);*/
    opencvClassifier->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 512, 1e-6));
}

//=====================================================
class OpencvMLPClassifier: public Classifier{
public:
    OpencvMLPClassifier();
    void train();
    int predict(const Feature_vector& inputFeatures);
private:
    Ptr<ANN_MLP> opencvClassifier;
};
OpencvMLPClassifier::OpencvMLPClassifier():Classifier("MLP")
{
#if 0
    int method = ANN_MLP::BACKPROP;
    double method_param = 0.0001;
    int max_iter = 1000;
#else
    int method = ANN_MLP::RPROP;
    double method_param = 0.1;
    int max_iter = 1000;
#endif
    opencvClassifier = ANN_MLP::create();
    opencvClassifier->setTermCriteria(TC(max_iter,0));
    opencvClassifier->setTrainMethod(method, method_param);
}
void OpencvMLPClassifier::train(){
    size_t num_of_training_data=dataset.size()-test_set.size();
    Mat labelsMat(num_of_training_data, 1, CV_32S);
    Mat trainingDataMat(num_of_training_data, num_of_cont_features, CV_32FC1);
    int ind=0;
    for(size_t i=0;i<num_of_classes;++i){
        for(size_t t=0;t<training_set[i].size();++t){
            for(size_t fi=0;fi<num_of_cont_features;++fi){
                FEATURE_TYPE val=normalize(tmp_dataset[training_set[i][t]],fi);
                trainingDataMat.at<float>(ind,fi)=
                    val;
                    //dataset[training_set[i][t]].features[fi];
                    //(dataset[training_set[i][t]].features[fi]-avgValues[fi])/stdValues[fi];
                    //(dataset[training_set[i][t]].features[fi]-minValues[fi])/(maxValues[fi]-minValues[fi])*2-1;
            }
            labelsMat.at<int>(ind,0)=i;
            ++ind;
        }
    }
    Mat train_responses = Mat::zeros(num_of_training_data, num_of_classes, CV_32F );

    // 1. unroll the responses
    for( size_t i = 0; i < num_of_training_data; i++ )
    {
        int cls_label = labelsMat.at<int>(i,0);
        train_responses.at<float>(i, cls_label) = 1.f;
    }
    //train_responses.copyTo(labelsMat);
    labelsMat=train_responses;//.clone();
    // 2. train classifier
    /*int layer_sz[] = { num_of_cont_features, 128, num_of_classes };
    int nlayers = (int)(sizeof(layer_sz)/sizeof(layer_sz[0]));
    Mat layer_sizes( 1, nlayers, CV_32S, layer_sz );*/
    Mat layer_sizes = Mat(3, 1, CV_16U);
    layer_sizes.row(0) = Scalar(trainingDataMat.cols);
#ifdef USE_CALTECH
    layer_sizes.row(1) = Scalar(256);
#else
    layer_sizes.row(1) = Scalar(128);
#endif
    layer_sizes.row(2) = Scalar(train_responses.cols);

    opencvClassifier->setLayerSizes(layer_sizes);
    opencvClassifier->setActivationFunction(ANN_MLP::SIGMOID_SYM, 1, 1);
    opencvClassifier->train(trainingDataMat, ROW_SAMPLE, labelsMat);
}
int OpencvMLPClassifier::predict(const Feature_vector& inputFeatures){
    Mat queryMat(1, num_of_cont_features, CV_32FC1);
    for(size_t fi=0;fi<num_of_cont_features;++fi){
        FEATURE_TYPE val=normalize(inputFeatures,fi);
        queryMat.at<float>(0,fi)=
            val;
            //dataset[test_ind].features[fi];
            //(dataset[test_ind].features[fi]-avgValues[fi])/stdValues[fi];
            //(dataset[test_ind].features[fi]-minValues[fi])/(maxValues[fi]-minValues[fi])*2-1;
    }
    cv::Mat predictions;
    opencvClassifier->predict(queryMat, predictions);
    float maxPrediction = predictions.at<float>(0);
    int response = 0;
    //const float* ptrPredictions = predictions.ptr<float>(0);
    for(size_t i = 0; i < predictions.cols; i++)
    {
        float prediction = predictions.at<float>(i);
        if (prediction > maxPrediction)
        {
            maxPrediction = prediction;
            response = i;
        }
    }
    return response;
}

//=====================================================
class FPNNClassifier: public PNNClassifier{
public:
    FPNNClassifier(double scale=1.0, bool bf=true, float output_ratio=0.9f):
        PNNClassifier(bf,to_string("FPNN",scale)),features_scale(scale),output_delta(fastlog(output_ratio)){}
    void train();
protected:
    FEATURE_TYPE normalize(const Feature_vector& feature_vector, size_t fi);
    int predict_bf(const Feature_vector& inputFeatures);
    int predict_sequentional(const Feature_vector& inputFeatures);

private:
    size_t J;
    vector<FEATURE_TYPE> prior_probabs;
    vector<double> a;
    vector<double> feature_weights;
    double features_scale;
    float output_delta;
};

FEATURE_TYPE FPNNClassifier::normalize(const Feature_vector& feature_vector, size_t fi){
#if 0
    FEATURE_TYPE val=(stdValues[fi]!=0)?0.9*(feature_vector.features[fi]-avgValues[fi])/stdValues[fi]:0;
    val=(1-exp(-val))/(1+exp(-val));
#elif 0
    FEATURE_TYPE val=(maxValues[fi]==minValues[fi])?0.0:0.1*(2*(feature_vector.features[fi]-minValues[fi])/(maxValues[fi]-minValues[fi])-1);
    //FEATURE_TYPE val=(maxValues[fi]==minValues[fi])?0.0:(2*(feature_vector.features[fi]-minValues[fi])/(maxValues[fi]-minValues[fi])-1);
#elif 0
    FEATURE_TYPE val=(maxValues[fi]==minValues[fi])?0.0:(feature_vector.features[fi]-avgValues[fi])/(maxValues[fi]-minValues[fi]);
#elif 1
    FEATURE_TYPE val=(stdValues[fi]!=0)?features_scale*(feature_vector.features[fi]-avgValues[fi])/stdValues[fi]:0;
#else
    FEATURE_TYPE val=(feature_vector.features[fi]-avgValues[fi]); //feature_weights[fi]
#endif
    const FEATURE_TYPE max_val=0.5;
    if(val<-max_val)
        val=-max_val;
    else if (val>max_val)
        val=max_val;
    return val;
}

const double PI=atan(1.0)*4;

void FPNNClassifier::train()
{
    size_t num_of_training_data=dataset.size()-test_set.size();
    feature_weights.assign(num_of_cont_features,1.0);
    prior_probabs.resize(num_of_classes);
    for(size_t i=0;i<num_of_classes;++i)
        prior_probabs[i]=1.0*training_set[i].size()/num_of_training_data;

    J=(int)ceil(pow(1.0*num_of_training_data/num_of_classes,1.0/3));//*2);
    //J=(int)ceil(pow(1.0*num_of_training_data/num_of_classes,1.0/3)*2);
    //J=(int)ceil(pow(1.0*num_of_training_data/num_of_classes,1.0/2)*2);
    //J=num_of_training_data/(num_of_classes);
    const int min_J=3;//1;
    if(J<=min_J)
       J = min_J;

    a.resize(num_of_classes*num_of_cont_features*(2*J+1));
    fill(a.begin(),a.end(),0);
    for(size_t fi=0;fi<num_of_cont_features;++fi) {

        for(size_t i=0;i<num_of_classes;++i){
            size_t model_ind=(fi*num_of_classes+i)*(2*J+1);
            a[model_ind]=0.5;
            double cur_mult=1.0/training_set[i].size();
            for(size_t t=0;t<training_set[i].size();++t){
                //cout<<model_ind<<' '<<a[model_ind]<<' '<<tmp_dataset[training_set[i][t]].features[fi]<<' '<<avgValues[fi];
                FEATURE_TYPE val=normalize(tmp_dataset[training_set[i][t]],fi);
                for(size_t j=0;j<J;++j){
                    a[model_ind+2*j+1]+=cos(PI*(j+1)*val)*cur_mult*(J-j)/(J*(J+1));
                    a[model_ind+2*j+2]+=sin(PI*(j+1)*val)*cur_mult*(J-j)/(J*(J+1));
                    //cout<<"train coefs="<<a[model_ind+2*j+1]<<' '<<a[model_ind+2*j+2]<<' '<<j<<' '<<val<<' '<<training_set[i][t];
                }
            }
        }
    }
}

int FPNNClassifier::predict_bf(const Feature_vector& inputFeatures){
    vector<float> outputs(num_of_classes,0);
    vector<double> cos_vals(J),sin_vals(J);
    size_t max_fi=num_of_cont_features;
    for(size_t fi=0;fi<max_fi;++fi){
        FEATURE_TYPE val=normalize(inputFeatures,fi);
        //if(fi==0 || fi==(max_fi-1))
        //    qDebug()<<fi<<' '<<tmp_dataset[test_ind].features[fi]<<' '<<val<<' '<<avgValues[fi]<<' '<<maxValues[fi]<<' '<<minValues[fi]<<' '<<stdValues[fi];
        cos_vals[0]=cos(PI*val);
        sin_vals[0]=sin(PI*val);
        for(size_t j=1;j<J;++j){
            cos_vals[j]=cos_vals[j-1]*cos_vals[0]-sin_vals[j-1]*sin_vals[0];
            sin_vals[j]=cos_vals[j-1]*sin_vals[0]+sin_vals[j-1]*cos_vals[0];
        }
        for(size_t i=0;i<num_of_classes;++i){
            size_t num_of_a_ind=fi*num_of_classes+i;
            size_t model_ind=num_of_a_ind*(2*J+1);
            double probab=a[model_ind];
            //qDebug()<<model_ind<<' '<<probab;
            for(size_t j=0;j<J;++j){
                probab+=(a[model_ind+2*j+1]*cos_vals[j]+a[model_ind+2*j+2]*sin_vals[j]);
                //qDebug()<<i<<' '<<fi<<' '<<model_ind<<' '<<j<<' '<<probab<<' '<<a[model_ind+2*j+1]<<' '<<cos_vals[j]<<' '<<a[model_ind+2*j+2]<<' '<<sin_vals[j];
            }
            outputs[i] += fastlog(probab);
        }
    }

    float max_output=-FLT_MAX;
    int bestClass=-1;
    for(size_t i=0;i<num_of_classes;++i){
        if(max_output<outputs[i]){
            max_output=outputs[i];
            bestClass=i;
        }
    }

    return bestClass;
}
int FPNNClassifier::predict_sequentional(const Feature_vector& inputFeatures){
    int bestClass=-1;

    vector<float> outputs(num_of_classes,0);
    vector<int> classes_to_check(num_of_classes,1);
    vector<double> cos_vals(J),sin_vals(J);

    size_t max_features=num_of_cont_features;
    //size_t max_features=256;
    for(size_t cur_features = 0; cur_features<max_features; cur_features += delta_features_count){
        size_t max_fi=cur_features+delta_features_count;
        if(max_fi>num_of_cont_features)
            max_fi=num_of_cont_features;
        for(size_t fi=cur_features;fi<max_fi;++fi){
            FEATURE_TYPE val=normalize(inputFeatures,fi);
            cos_vals[0]=cos(PI*val);
            sin_vals[0]=sin(PI*val);
            for(size_t j=1;j<J;++j){
                cos_vals[j]=cos_vals[j-1]*cos_vals[0]-sin_vals[j-1]*sin_vals[0];
                sin_vals[j]=cos_vals[j-1]*sin_vals[0]+sin_vals[j-1]*cos_vals[0];
            }
            for(size_t i=0;i<num_of_classes;++i){
                if(classes_to_check[i]){
                    size_t num_of_a_ind=fi*num_of_classes+i;
                    size_t model_ind=num_of_a_ind*(2*J+1);
                    double probab=a[model_ind];
                    for(size_t j=0;j<J;++j){
                        probab+=(a[model_ind+2*j+1]*cos_vals[j]+a[model_ind+2*j+2]*sin_vals[j]);
                    }
                    outputs[i] += fastlog(probab);//num_of_cont_features;
                }
            }
        }

        float max_output=-FLT_MAX;
        for(size_t i=0;i<num_of_classes;++i){
            if(classes_to_check[i] && max_output<outputs[i]){
                max_output=outputs[i];
                bestClass=i;
            }
        }
        int num_of_variants = 0;
        float output_threshold = max_output+output_delta*max_fi;
        //qDebug()<<max_output<<' '<<output_delta<<' '<<max_fi<<' '<<output_threshold<<' '<<outputs[0]<<' '<<outputs[1];
        for(size_t i=0;i<num_of_classes;++i){
            if (outputs[i]<output_threshold)
                classes_to_check[i] = 0;
            else
                ++num_of_variants;
        }
        if (num_of_variants == 1)
            break;
    }

    return bestClass;
}

//=======================================================================

void load_image_dataset(){
    num_of_cont_features=FEATURES_COUNT;///2;

    cout<<FEATURES_FILE_NAME;
    print_endl;
    ifstream ifs(FEATURES_FILE_NAME);
    if (ifs){
        int total_images = 0;
        map<string, int> person2indexMap;
        while (ifs){
                std::string fileName, personName, feat_str;
                if (!getline(ifs, fileName))
                        break;
                if (!getline(ifs, personName))
                        break;
                if (!getline(ifs, feat_str))
                        break;
                //cout << fileName.c_str() << ' ' << personName.c_str() << '\n';
                personName.erase(0, personName.find_first_not_of(" \t\n\r\f\v"));
#ifdef USE_CALTECH
                if(personName.find("BACKGROUND_Google")!=string::npos ||
                        personName.find("257.clutter")!=string::npos)
                        continue;
#endif
                if (person2indexMap.find(personName) == person2indexMap.end()){
#if defined(USE_CASIA) // && defined (USE_LFW)
                    if(person2indexMap.size()>=1000)
                        ;//break;
#endif
                    person2indexMap.insert(std::make_pair(personName, person2indexMap.size()));
                }
                vector<FEATURE_TYPE> features(num_of_cont_features);
                istringstream iss(feat_str);
                //of << fileName << endl << personName << endl;
                FEATURE_TYPE sum=0;
                for(size_t i = 0; i < FEATURES_COUNT; ++i){
                    FEATURE_TYPE feature;
                    iss >> feature;
#if 1
                    sum+=feature*feature;
#else
                    sum+=abs(feature);
#endif
                    if(i<num_of_cont_features)
                        features[i]=feature;
                }
                //cout<<features[0]<<' '<<features[FEATURES_COUNT-1];
#if 1
                sum=sqrt(sum);
#endif
                for(size_t i = 0; i < num_of_cont_features; ++i){
                    features[i]/=sum;
                }
                dataset.push_back(Feature_vector(features, person2indexMap[personName]));

                ++total_images;
        }
        ifs.close();
        num_of_classes=person2indexMap.size();
        indices.resize(num_of_classes);
        for(size_t i=0;i<dataset.size();++i){
            int class_ind=(int)(dataset[i].output);
            indices[class_ind].push_back(i);
        }
        cout<<num_of_classes<<' '<<num_of_cont_features<<' '<<total_images;
        print_endl;
    }
}

void extract_pca_features(){
    Mat training_mat(dataset.size()-test_set.size(), num_of_cont_features, CV_64F);
    int mat_ind=0;
    for(size_t i=0;i<num_of_classes;++i){
        for(size_t t=0;t<training_set[i].size();++t){
            for(size_t fi=0;fi<num_of_cont_features;++fi){
                training_mat.at<double>(mat_ind,fi)=
                    //dataset[training_set[i][t]].features[fi];
                    //(dataset[training_set[i][t]].features[fi]-avgValues[fi])/stdValues[fi];
                    //(dataset[training_set[i][t]].features[fi]-minValues[fi])/(maxValues[fi]-minValues[fi])*2-1;
                        dataset[training_set[i][t]].features[fi]-avgValues[fi];
            }
            ++mat_ind;
        }
    }

    auto start_time = chrono::high_resolution_clock::now();
    PCA pca(training_mat,Mat(),PCA::DATA_AS_ROW,
        NO_PCA_FEATURES);
        //training_mat.cols);
    Mat test_mat(dataset.size(), num_of_cont_features, CV_64F);
    /*cout<<"PCA "<<pca.eigenvalues.rows<<' '<<pca.eigenvalues.cols<<' '<<pca.eigenvalues.at<double>(0,0);
    for(size_t i=0;i<NO_PCA_FEATURES;++i)
        cout<<pca.eigenvalues.at<double>(0,i);*/
    for(size_t i=0;i<dataset.size();++i){
        for(size_t fi=0;fi<num_of_cont_features;++fi){
            test_mat.at<double>(i,fi)=
                //dataset[i].features[fi];
                //(dataset[i].features[fi]-avgValues[fi])/stdValues[fi];
                //(dataset[i].features[fi]-minValues[fi])/(maxValues[fi]-minValues[fi])*2-1;
                    dataset[i].features[fi]-avgValues[fi];
        }
    }
    Mat point=pca.project(test_mat);
    //cout<<point.rows<<' '<<point.cols<<'\n';
    for(size_t i=0;i<dataset.size();++i){
        /*for(size_t fi=0;fi<num_of_cont_features;++fi){
            tmp_dataset[i].features[fi]=0;
        }*/
        tmp_dataset[i].features.resize(point.cols);
        for(size_t fi=0;fi<point.cols;++fi){
            tmp_dataset[i].features[fi]=point.at<double>(i,fi);
            if(tmp_dataset[i].features[fi]!=tmp_dataset[i].features[fi]){
                cout<<"error "<<tmp_dataset[i].features[fi]<<' '<<i<<' '<<fi;
                print_endl;
            }
            //cout<<tmp_dataset[i].features[fi]<<'\n';
        }
        //cout<<tmp_dataset[i].features[0]<<' '<<tmp_dataset[i].features[FEATURES_COUNT-1];
    }
    num_of_cont_features=NO_PCA_FEATURES;

    for(size_t fi=0;fi<num_of_cont_features;++fi){
        minValues[fi]=FLT_MAX;
        maxValues[fi]=-FLT_MAX;
        avgValues[fi]=stdValues[fi]=0;
        int count=0;
        for(size_t i=0;i<num_of_classes;++i){
            for(size_t t=0;t<training_set[i].size();++t){
                ++count;
                FEATURE_TYPE feature=tmp_dataset[training_set[i][t]].features[fi];
                if(feature<minValues[fi])
                    minValues[fi]=feature;
                if(maxValues[fi]<feature)
                    maxValues[fi]=feature;
                avgValues[fi]+=feature;
                stdValues[fi]+=feature*feature;
                //cout<<i<<' '<<t<<' '<<feature<<'\n';
            }
        }

        //cout<<"after PCA:"<<fi<<' '<<avgValues[fi]<<' '<<count;
        avgValues[fi]/=count;
        stdValues[fi]=sqrt((stdValues[fi] - avgValues[fi] * avgValues[fi] * count) / (count - 1));
    }
    //qDebug()<<stdValues[0]<<' '<<stdValues[1]<<' '<<stdValues[2]<<' '<<stdValues[FEATURES_COUNT-2]<<' '<<stdValues[FEATURES_COUNT-1];
}

void split_train_test(double fraction){
    training_set.resize(num_of_classes);

    minValues.resize(num_of_cont_features);
    maxValues.resize(num_of_cont_features);
    avgValues.resize(num_of_cont_features);
    stdValues.resize(num_of_cont_features);
    test_set.clear();
    test_set.reserve(dataset.size());
    for(size_t i=0;i<num_of_classes;++i){
        std::random_shuffle ( indices[i].begin(), indices[i].end() );
        int end=fraction>=1?(int)fraction:ceil(fraction*indices[i].size());
        if(end==0 && !indices[i].empty())
            end=1;
        else if (end>=indices[i].size())
            end=indices[i].size();
        //++end;
        training_set[i].clear();
        training_set[i].assign(indices[i].begin(), indices[i].begin()+end);

        test_set.insert(test_set.end(),indices[i].begin()+end, indices[i].end());
        //test_set.insert(test_set.end(),indices[i].begin(), indices[i].begin()+end);
    }

    tmp_dataset=dataset;
    num_of_cont_features=num_of_cont_features_orig;

    for(size_t fi=0;fi<num_of_cont_features;++fi){
        minValues[fi]=FLT_MAX;
        maxValues[fi]=-FLT_MAX;
        avgValues[fi]=stdValues[fi]=0;
        int count=0;
        for(size_t i=0;i<num_of_classes;++i){
            for(size_t t=0;t<training_set[i].size();++t){
                ++count;
                FEATURE_TYPE feature=dataset[training_set[i][t]].features[fi];
                if(feature<minValues[fi])
                    minValues[fi]=feature;
                if(maxValues[fi]<feature)
                    maxValues[fi]=feature;
                avgValues[fi]+=feature;
                stdValues[fi]+=feature*feature;
            }
        }

        avgValues[fi]/=count;
        stdValues[fi]=sqrt((stdValues[fi] - avgValues[fi] * avgValues[fi] * count) / (count - 1));
    }
}
void testClassification1(){
    load_image_dataset();
    cout<<"pca_features="<<NO_PCA_FEATURES;
    print_endl;
    num_of_cont_features_orig=num_of_cont_features;

    vector<Classifier*> classifiers;
    classifiers.push_back(new KNNClassifier(1));
    classifiers.push_back(new KNNClassifier(3));
    classifiers.push_back(new PNNClassifier(true));
    classifiers.push_back(new PNNwithClusteringClassifier(5));
    classifiers.push_back(new FPNNClassifier(1.0,true));
    classifiers.push_back(new FPNNClassifier(0.33,true));
#if NO_PCA_FEATURES>32
    classifiers.push_back(new PNNClassifier(false));
    classifiers.push_back(new FPNNClassifier(1.0,false));
    classifiers.push_back(new FPNNClassifier(0.33,false));
#endif
    classifiers.push_back(new OpencvSVMClassifier(true));
    classifiers.push_back(new OpencvSVMClassifier(false));
    classifiers.push_back(new OpencvRTreeClassifier());
    //classifiers.push_back(new OpencvMLPClassifier());


    const int TEST_COUNT = 2;//3;//10;
    const int max_fraction = 10;
    int num_of_tests=TEST_COUNT;

    ofstream fres("classification_res.txt");
    for(double fraction=5;fraction<=30;fraction+=5){
    //for(double fraction=1;fraction<=15;fraction+=2){
    //for(double fraction=10;fraction<=60;fraction+=50){
    //for(double fraction=0.5;fraction<=0.5;fraction+=0.1){
        vector<double> totalErrorRates(classifiers.size(),0);
        vector<double> totalErrorSquares(classifiers.size(),0);
        vector<double> totalRecalls(classifiers.size(),0);
        vector<double> totalTimes(classifiers.size(),0);

        for(size_t shuffle_num=0;shuffle_num<1;++shuffle_num){
            for(size_t test_num=0;test_num<num_of_tests;++test_num){
                split_train_test(fraction);
#if NO_PCA_FEATURES>0
                extract_pca_features();
#endif
                for(size_t i=0;i<classifiers.size();++i)
                {
                    //classifiers[i]->print();
                    classifiers[i]->train();
                    //cout<<training_set[0].size()<<' '<<test_set.size()<<'\n';
                    double errorRate=0,recall=0;
                    vector<int> testClassCount(num_of_classes),testErrorRates(num_of_classes);
                    for(size_t j=0;j<test_set.size();++j){
                        ++testClassCount[(int)dataset[test_set[j]].output];
                    }

                    auto t1 = chrono::high_resolution_clock::now();
                    for(size_t j=0;j<test_set.size();++j){
                        int bestClass=classifiers[i]->predict(tmp_dataset[test_set[j]]);
                        if(bestClass!=dataset[test_set[j]].output){
                            errorRate+=1;
                            testErrorRates[(int)dataset[test_set[j]].output]+=1;
                        }
                    }
                    auto t2 = chrono::high_resolution_clock::now();
                    auto diff=std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/(1.0*test_set.size());
                    totalTimes[i] += diff;

                    errorRate/=test_set.size();
                    for(size_t i=0;i<num_of_classes;++i){
                        if(testClassCount[i]>0)
                            recall+=1-1.*testErrorRates[i] / testClassCount[i];
                    }
                    recall/=num_of_classes;
                    totalErrorRates[i]+=errorRate;
                    totalErrorSquares[i] += errorRate*errorRate;
                    totalRecalls[i]+=recall;
                    //cout<<errorRate;print_endl;
                }
            }
            for(size_t i=0;i<num_of_classes;++i){
                std::random_shuffle ( indices[i].begin(), indices[i].end() );
            }
        }
        for(size_t i=0;i<classifiers.size();++i){
            totalErrorRates[i]/=num_of_tests;
            totalErrorSquares[i] = sqrt((totalErrorSquares[i] - totalErrorRates[i] * totalErrorRates[i] * num_of_tests) / (num_of_tests - 1));
            totalRecalls[i]/=num_of_tests;
            cout<<classifiers[i]->get_name().c_str();print_endl;
            cout<<"fraction="<<fraction<<" db_size="<<(dataset.size()-test_set.size())<<" error="<< (100 * totalErrorRates[i]) << " sigma=" << (100 * totalErrorSquares[i])<<" avg time(us)="<<(totalTimes[i]/num_of_tests)<<" recall="<< (100 * totalRecalls[i]);
            fres<<fraction<<"\t"<< (100 * totalErrorRates[i]) << " \t" << (100 * totalErrorSquares[i])<<"\t"<<(totalTimes[i]/num_of_tests);
            print_endl;
        }
        cout<<' ';print_endl;
    }

    for(size_t i=0;i<classifiers.size();++i)
        delete classifiers[i];
}


void testClassification(){
    load_image_dataset();
    cout<<"pca_features="<<NO_PCA_FEATURES;
    print_endl;
    num_of_cont_features_orig=num_of_cont_features;

    double fraction=30;
    const int TEST_COUNT = 3;//2;//10;
    int num_of_tests=TEST_COUNT;

    map<string,double> totalErrorRates;
    map<string,double> totalErrorSquares;
    map<string,double> totalRecalls;
    map<string,double> totalTimes;

    for(size_t test_num=0;test_num<num_of_tests;++test_num){
        split_train_test(fraction);
#if NO_PCA_FEATURES>0
        extract_pca_features();
#endif
        for(float output_ratio=0.6f;output_ratio<1.0f;output_ratio+=0.01f){
            vector<Classifier*> classifiers;
            classifiers.push_back(new FPNNClassifier(1.0,false,output_ratio));
            classifiers.push_back(new FPNNClassifier(0.33,false,output_ratio));

            for(size_t i=0;i<classifiers.size();++i)
            {
                string name=to_string(classifiers[i]->get_name(),output_ratio);
                //classifiers[i]->print();
                classifiers[i]->train();
                //cout<<training_set[0].size()<<' '<<test_set.size()<<'\n';
                double errorRate=0,recall=0;
                vector<int> testClassCount(num_of_classes),testErrorRates(num_of_classes);
                for(size_t j=0;j<test_set.size();++j){
                    ++testClassCount[(int)dataset[test_set[j]].output];
                }

                auto t1 = chrono::high_resolution_clock::now();
                for(size_t j=0;j<test_set.size();++j){
                    int bestClass=classifiers[i]->predict(tmp_dataset[test_set[j]]);
                    if(bestClass!=dataset[test_set[j]].output){
                        errorRate+=1;
                        testErrorRates[(int)dataset[test_set[j]].output]+=1;
                    }
                }
                auto t2 = chrono::high_resolution_clock::now();
                auto diff=std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/(1.0*test_set.size());
                totalTimes[name] += diff;

                errorRate/=test_set.size();
                for(size_t i=0;i<num_of_classes;++i){
                    if(testClassCount[i]>0)
                        recall+=1-1.*testErrorRates[i] / testClassCount[i];
                }
                recall/=num_of_classes;
                totalErrorRates[name]+=errorRate;
                totalErrorSquares[name] += errorRate*errorRate;
                totalRecalls[name]+=recall;

                //cout<<name.c_str()<<' '<<errorRate;
            }
            for(size_t i=0;i<classifiers.size();++i)
                delete classifiers[i];
        }
        for(size_t i=0;i<num_of_classes;++i){
            std::random_shuffle ( indices[i].begin(), indices[i].end() );
        }
    }
    for ( const auto &timePair : totalTimes) {
        string name=timePair.first;
        totalErrorRates[name]/=num_of_tests;
        totalErrorSquares[name] = sqrt((totalErrorSquares[name] - totalErrorRates[name] * totalErrorRates[name] * num_of_tests) / (num_of_tests - 1));
        totalRecalls[name]/=num_of_tests;
        cout<<"db_size="<<(dataset.size()-test_set.size())<<"\t"<<name.c_str()<<"\terror="<< (100 * totalErrorRates[name]) << "\tsigma=" << (100 * totalErrorSquares[name])<<"\tavg time(us)="<<(timePair.second/num_of_tests)<<"\trecall="<< (100 * totalRecalls[name]);
        print_endl;
    }
    cout<<' ';print_endl;

}
