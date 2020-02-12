#include "ann.h"
#include "db_features.h"

#include <set>
#include <iostream>
#include <algorithm>
#include <locale>
#include <string>
#include <chrono>
#include <functional>
using namespace std;


#ifdef QT_BUILD
#include <QtCore>
#include <QDebug>
#undef cout
#define cout qDebug()
#define print_endl
#else
#define print_endl cout<<endl;
#endif

void testANN(){
    ImagesDatabase totalImages;
    unordered_map<string, int> person2indexMap;
#if defined(USE_PCA) && 0
    ImagesDatabase orig_database;
    loadImages(orig_database, FEATURES_FILE_NAME,person2indexMap);
    extractPCA(orig_database, totalImages);
#else
    ImagesDatabase orig_database;
    loadImages(orig_database, FEATURES_FILE_NAME,person2indexMap);
    for (auto& features:orig_database){
        if (features.size() > 1)
            totalImages.push_back(features);
    }
    cout << "total size=" << totalImages.size() << " o=" << orig_database.size();
    print_endl;
#endif
    int num_of_classes = totalImages.size();

    const int TESTS = 1;
    std::vector<ImageInfo> dbImages, testImages;
    double totalTestsErrorRate = 0, errorRateVar = 0;
    for (int testCount = 0; testCount < TESTS; ++testCount){
        getTrainingAndTestImages(totalImages, dbImages, testImages);

        cout<< "test=" << testCount<<" dbSize=" << dbImages.size() <<" testSize=" << testImages.size();
        print_endl;

        BruteForce bf(dbImages);
        bf.testSetRecognition(testImages);

        vector<ClassificationMethod*> methods;
        methods.push_back(new FlannMethod(dbImages));
#ifdef USE_NMSLIB
        methods.push_back(new NmslibMethod(dbImages));
#endif
        methods.push_back(new DirectedEnumeration(dbImages));
        for (double ratio = 0.025; ratio <= 0.5/*1.001*/; ratio += 0.025)
        {
            int imageCountToCheck = (int)(ratio*dbImages.size());
            cout<<"ratio"<<ratio;
            print_endl;
            for(int i=0;i<methods.size();++i){
                methods[i]->setImageCountToCheck(imageCountToCheck);
                methods[i]->testSetRecognition(testImages);
            }
        }
        for(int i=0;i<methods.size();++i)
            delete methods[i];
        /*totalTestsErrorRate += errorRate;
        errorRateVar += errorRate * errorRate;*/

    }
    /*totalTestsErrorRate /= TESTS;
    total_time /= TESTS;
    errorRateVar = sqrt((errorRateVar - totalTestsErrorRate * totalTestsErrorRate * TESTS) / (TESTS - 1));
    cout << "Avg error=" << totalTestsErrorRate << " Sigma=" << errorRateVar << " time(ms)=" << total_time << endl;*/
}


float ClassificationMethod::getThreshold(vector<float>& otherClassesDists, float falseAcceptRate)
{
    int ind = (int)(otherClassesDists.size()*falseAcceptRate);
    std::nth_element(otherClassesDists.begin(), otherClassesDists.begin() + ind, otherClassesDists.end());
    float threshold = otherClassesDists[ind];
    cout << threshold << " " << *std::min_element(otherClassesDists.begin(), otherClassesDists.end()) << " " << *std::max_element(otherClassesDists.begin(), otherClassesDists.end());
    print_endl;

    return threshold;
}
void ClassificationMethod::testSetRecognition(std::vector<ImageInfo>& testImages){
    int errorsCount=0;
    avgCheckedPercent=0;
    auto t1 = chrono::high_resolution_clock::now();
    for (ImageInfo testImageInfo : testImages){
        int bestInd = recognize(testImageInfo);
        if(bestInd == -1 || testImageInfo.classNo != dbImages[bestInd].classNo)
            ++errorsCount;
    }
    auto t2 = chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

    double errorRate = 100.*errorsCount / testImages.size();
    cout << method_name.c_str()<<" error=" << errorRate << "% total_time (ms)" << total_time/testImages.size()<<" checkedPercent="<<(avgCheckedPercent>0?avgCheckedPercent/testImages.size():-1);
    print_endl;
}


//nearest neighbor (brute force)
int BruteForce::recognize(ImageInfo& testImage){
    distanceCalcCount = 0;
    int bestInd = -1;
    double bestDist = 100000,dist;
    for (int j = 0; j < dbImages.size(); ++j){
            dist =distance(testImage,j);
            if (dist < bestDist){
                    bestDist = dist;
                    bestInd = j;
            }
    }
    //cout<<bestDist;
    return bestInd;
}

//svm
using namespace cv;
using namespace cv::ml;
SvmClassifier::SvmClassifier(std::vector<ImageInfo>& dbImages):ClassificationMethod("SVM",dbImages)
{
    int num_of_cont_features=FEATURES_COUNT;
    if(dbImages.size()>0)
        num_of_cont_features=dbImages[0].features.size();
    Mat labelsMat(dbImages.size(), 1, CV_32S);
    Mat trainingDataMat(dbImages.size(), num_of_cont_features, CV_32FC1);
    for(int i=0;i<dbImages.size();++i){
        for(int fi=0;fi<num_of_cont_features;++fi){
            trainingDataMat.at<float>(i,fi)= dbImages[i].features[fi];
        }
        labelsMat.at<int>(i,0)=dbImages[i].classNo;
    }

    // Set up SVM's parameters
    svmClassifier = SVM::create();
    svmClassifier->setType(SVM::C_SVC);
    svmClassifier->setKernel(SVM::LINEAR);
    //svmClassifier->setKernel(SVM::RBF);
    svmClassifier->setGamma(1.0 / num_of_cont_features);
    //params.term_crit = TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6);

    // Train the SVM
    svmClassifier->train(trainingDataMat, ROW_SAMPLE, labelsMat);
}

int SvmClassifier::recognize(ImageInfo& testImage){
    Mat queryMat(1, testImage.features.size(), CV_32FC1);
    for(int fi=0;fi<testImage.features.size();++fi){
        queryMat.at<float>(0,fi)=testImage.features[fi];
    }
    float response = svmClassifier->predict(queryMat);
    for(int i=0;i<dbImages.size();++i){
        if(fabs(dbImages[i].classNo-response)<0.1)
            return i;
    }
    return -1;
}


//k-d trees
FlannMethod::FlannMethod(std::vector<ImageInfo>& dbImages):ClassificationMethod("flann",dbImages)
{
    dictionary_features = new float[dbImages.size()*FEATURES_COUNT];
    for(int j=0;j<dbImages.size();++j){
        for (int k = 0; k<FEATURES_COUNT; ++k)
            dictionary_features[j*FEATURES_COUNT + k] = dbImages[j].features[k];
    }
    cvflann::Matrix<float> samplesMatrix((float*)dictionary_features, dbImages.size(), FEATURES_COUNT);
    flann_index=new cvflann::Index<CURRENT_DISTANCE<float>>(samplesMatrix, cvflann::KDTreeIndexParams(4));
    flann_index->buildIndex();
}
FlannMethod::~FlannMethod(){
    delete flann_index;
    delete dictionary_features;
}
int FlannMethod::recognize(ImageInfo& testImage){
    int bestInd=-1;
    float closestDistance;
    cvflann::Matrix<int> indices(&bestInd, 1, 1);
    cvflann::Matrix<float> dists(&closestDistance, 1, 1);
    cvflann::Matrix<float> query((float*)&testImage.features[0], 1, testImage.features.size());
    flann_index->knnSearch(query, indices, dists, 1,
           cvflann::SearchParams(imageCountToCheck==0?dbImages.size():imageCountToCheck));
    return bestInd;
}

#ifdef USE_NMSLIB
//methods from NMSLib
using namespace similarity;
NmslibMethod::NmslibMethod(std::vector<ImageInfo>& dbImages):ClassificationMethod("nmslib",dbImages)
{
    initLibrary(LIB_LOGNONE, NULL);
    queryData.resize(FEATURES_COUNT);
    vector<vector<float>> rawData(dbImages.size());
    for (int i = 0; i < dbImages.size(); ++i){
        rawData[i].resize(FEATURES_COUNT);
        for (int j = 0; j < FEATURES_COUNT; ++j)
            rawData[i][j] = dbImages[i].features[j];
    }
    customSpace.CreateDataset(dataSet, rawData);

    index =
        MethodFactoryRegistry<float>::Instance().
        CreateMethod(false,
#if 0
        "small_world_rand",
#elif 1
        "proj_incsort",
#endif
        "custom", customSpace,dataSet
        );

    index->CreateIndex(
                AnyParams({
#if 0
                      "NN=25",//11 //15
                      "initIndexAttempts=5",
                      "initSearchAttempts=1",
                      "indexThreadQty=4", /* 4 indexing threads */
#else
                      "projDim=32",
                              "projType=perm",
#endif
                 }));
}
void NmslibMethod::setImageCountToCheck(int imageCountToCheck)
{
    ClassificationMethod::setImageCountToCheck(imageCountToCheck);
    double ratio = 1.0;
    if (imageCountToCheck > 0){
        ratio = ((double)imageCountToCheck) / dbImages.size();
        char params[16];
        sprintf(params, "dbScanFrac=%.2f", ratio);
        index->SetQueryTimeParams(AnyParams({ params }));
    }
}

int NmslibMethod::recognize(ImageInfo& testImage){
    unsigned int K = 1; // 1-NN query
    for (int j = 0; j < FEATURES_COUNT; ++j)
        queryData[j] = testImage.features[j];
    const Object*   queryObj = customSpace.CreateObjFromVect(-1, 0,queryData);
    KNNQuery<float>   knnQ(customSpace, queryObj, K);
    index->Search(&knnQ);
    KNNQueue<float>* res = knnQ.Result()->Clone();
    int bestInd=-1;
    if (!res->Empty()){
        //closestDistance = res->TopDistance();
        bestInd=res->Pop()->id();
    }
    avgCheckedPercent += 100.*knnQ.DistanceComputations() / dbImages.size();
    knnQ.Reset();
    delete queryObj;
    return bestInd;
}
#endif

//directed enumeration/maximal-likelihood ANN methods
#define PIVOT
DirectedEnumeration::DirectedEnumeration(vector<ImageInfo>& faceImages, float falseAcceptRate/*=0.05*/, float threshold/*=0*/, int imageCountToCheck/*=0*/) :
ClassificationMethod("dem",faceImages)
{
    auto t1 = std::chrono::high_resolution_clock::now();
    init(imageCountToCheck);

    if (threshold>0){
        this->threshold = threshold;
    }

    int dbSize=dbImages.size();
    vector<float> otherClassesDists;
#ifndef PIVOT
    P_matrix = new ImageDist[dbSize*dbSize];

    for (int i = 0; i<dbSize; ++i){
        for (int j = 0; j<dbSize; ++j){
            float dist = distance(dbImages[j], i, false);
            P_matrix[i*dbSize + j] = ImageDist(dist, j);
        }
        std::sort(&P_matrix[i*dbSize], &P_matrix[(i + 1)*dbSize]);
        for (int idInd = 0; idInd<dbSize; ++idInd){
            ImageDist& id = P_matrix[i*dbSize + idInd];
            if (dbImages[i].classNo != dbImages[id.imageNum].classNo){
                otherClassesDists.push_back(id.dist);
                break;
            }
        }
        std::sort(&P_matrix[i*dbSize], &P_matrix[(i + 1)*dbSize], ImageDist::ComparerByNumber);
    }
#else
    P_matrix = new ImageDist[startIndices.size()*dbSize];
    for (int ii = 0; ii < startIndices.size(); ++ii){
        int i = startIndices[ii];
        int mostFarModel = -1;
        double maxFarDist = 0;
        float min_other_dist = numeric_limits<float>::max();
        for (int j = 0; j < dbSize; ++j){
            float d = distance(dbImages[j], i, false);

            P_matrix[ii*dbSize + j] = ImageDist(d, j);
            if (dbImages[i].classNo != dbImages[j].classNo && P_matrix[ii*dbSize + j].dist<min_other_dist){
                min_other_dist = P_matrix[ii*dbSize + j].dist;
            }
            double currentFarDist = 0;
            for (int ind = 0; ind <= ii; ++ind){
                if (startIndices[ind] == j)
                    currentFarDist = -1000000;
                else
                    currentFarDist += P_matrix[ind * dbSize + j].dist;
            }
            if (currentFarDist > maxFarDist){
                maxFarDist = currentFarDist;
                mostFarModel = j;
            }
        }
        otherClassesDists.push_back(min_other_dist);
        if (ii < startIndices.size() - 1)
            //;
            startIndices[ii + 1] = mostFarModel;
    }
    if(startIndices.size()>32)
        startIndices.resize(32);
    /*if (threshold <= 0){
        cout << "min_other_dist=" << min_other_dist;
        print_endl;
        this->threshold = min_other_dist;
    }(*/
#endif
    if (threshold <= 0){
        this->threshold = getThreshold(otherClassesDists, falseAcceptRate);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    cout << "init took "
        << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
        << " milliseconds";
    print_endl;
}


DirectedEnumeration::~DirectedEnumeration(){
    delete[] P_matrix;

    delete[] likelihoods;
    delete[] likelihood_indices;
}
void DirectedEnumeration::init(int imageCountToCheck){
    likelihoods = NULL;
    likelihood_indices = NULL;
    int dbSize = dbImages.size();
    likelihoods = new float[dbSize];
    likelihood_indices = new int[dbSize];

    setImageCountToCheck(imageCountToCheck);

    vector<int> indices(dbSize);
    for (int i = 0; i < dbSize; ++i)
        indices[i] = i;
    random_shuffle(indices.begin(), indices.end());
    //srand(time(0));
    int N =
#ifdef PIVOT
        (int)(dbSize*0.015);//0.03 32;
    if (N<5)
        N = 5;
    /*else if (N>32)
        N = 32;*/
    //N=5;
    cout<<N;
#else
        1;
#endif
    startIndices.resize(N);
    for (int i = 0; i<N; ++i)
        startIndices[i] = indices[i];
}



#define CHECK_FOR_BEST_DIST                            \
        tmpDist = distance(testImage, imageNum);        \
        if(tmpDist<bestDistance){                       \
            bestItemUpdated=1;		                    \
            bestDistance=tmpDist;                       \
            bestIndex=imageNum;                         \
            if(bestDistance<threshold){                 \
                isFoundLessThreshold=true;              \
                goto end;                               \
                        }                                           \
                }


class LikelihoodsComparator :public std::binary_function<int, int, bool>{
public:
    LikelihoodsComparator(float* l) :
        likelihoods(l)
    {
    }
    bool operator() (int lhsIndex, int rhsIndex){
        return likelihoods[lhsIndex]<likelihoods[rhsIndex];
    }

private:
    float* likelihoods;
};
int DirectedEnumeration::recognize(ImageInfo& testImage)
{
    int i, nu, k;
    int imageNum, bestImageNum;
    int bestIndex = -1;
    int bestItemUpdated = 1;
    float tmpDist = 0;
    float modelsDist, tmp;
    float bestLikelihood;
    ImageDist* P_row;

    isFoundLessThreshold = false;
    bestDistance = numeric_limits<float>::max();
    distanceCalcCount = 0;
    int dbSize = dbImages.size();

    for (i = 0; i<dbSize; ++i){
        likelihoods[i] = 0;
        likelihood_indices[i] = i;
    }
    int start_index = 0;
    LikelihoodsComparator likelihoodsComparator(likelihoods);

    bestImageNum = -1;
    bestLikelihood = numeric_limits<float>::max();
    for (i = 0; i<startIndices.size(); ++i){
        imageNum = startIndices[i];
        CHECK_FOR_BEST_DIST
        likelihood_indices[imageNum] = likelihood_indices[start_index];
        likelihood_indices[start_index++] = imageNum;

        P_row = P_matrix + dbSize*
#ifdef PIVOT
            i;
#else
            imageNum;
#endif
        for (int ii = start_index; ii<dbSize; ++ii){
            nu = likelihood_indices[ii];
            modelsDist = P_row[nu].dist;
            if (modelsDist >= 0){
                tmp = tmpDist - modelsDist;
                likelihoods[nu] += tmp*tmp;
                //likelihoods[nu] += tmp*tmp / modelsDist;//+log(modelsDist)/2;
            }
        }
    }
    {
        int TRIALS =
#ifndef PIVOT
            2;
#else
            dbSize - start_index;
        std::partial_sort(likelihood_indices + start_index, likelihood_indices + imageCountToCheck,
            likelihood_indices + dbSize, likelihoodsComparator);
#endif
        while ((distanceCalcCount < imageCountToCheck)/*|| countOfLoopsWithNoUpdate<100) && distanceCalcCount <dbSize*/){
            bestItemUpdated = 0;
            if ((start_index + TRIALS) >= dbSize){
                imageNum = likelihood_indices[start_index++];
                CHECK_FOR_BEST_DIST
            }
            else{
                std::partial_sort(likelihood_indices + start_index, likelihood_indices + start_index + TRIALS,
                    likelihood_indices + dbSize, likelihoodsComparator);
                for (int i = 0; i < TRIALS; ++i){
                    bestImageNum = imageNum = likelihood_indices[start_index++];
                    CHECK_FOR_BEST_DIST
                    P_row = P_matrix + imageNum*dbSize;
                    /*if (bestItemUpdated)
                        countOfLoopsWithNoUpdate = 0;
                    else
                        ++countOfLoopsWithNoUpdate;
                        */
                    for (int ii = start_index; ii < dbSize; ++ii){
                        nu = likelihood_indices[ii];
                        modelsDist = P_row[nu].dist;
                        if (modelsDist >= 0){
                            tmp = tmpDist - modelsDist;
                            likelihoods[nu] += tmp*tmp;
                            //likelihoods[nu] += tmp*tmp / modelsDist;//+log(modelsDist)/2;
                        }
                    }
                }
            }
        }
    }
end:
    //startIndices[0]=bestIndex;
    avgCheckedPercent+=100.*distanceCalcCount / dbImages.size();
    return bestIndex;
}
