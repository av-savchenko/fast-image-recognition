#ifndef ANN_H
#define ANN_H

#include "db_features.h"

#include <vector>
#include <string>

class ClassificationMethod{
public:
    ClassificationMethod(std::string name,std::vector<ImageInfo>& db):method_name(name),dbImages(db),avgCheckedPercent(0)
    {
        imageCountToCheck=dbImages.size();
    }
    virtual ~ClassificationMethod(){}
    virtual int recognize(ImageInfo& testImageInfo)=0;

    void testSetRecognition(std::vector<ImageInfo>& testImages);

    virtual void setImageCountToCheck(int imageCountToCheck){
        this->imageCountToCheck = (imageCountToCheck>0 && imageCountToCheck<dbImages.size()) ? imageCountToCheck : dbImages.size();
    }

    static float getThreshold(std::vector<float>& otherClassesDists, float falseAcceptRate);

protected:
    std::string method_name;
    std::vector<ImageInfo>& dbImages;
    int distanceCalcCount;
    float avgCheckedPercent;
    int imageCountToCheck;

    float distance(ImageInfo& testImage, int modelInd, bool updateCounters = true){
        if (updateCounters){
            ++distanceCalcCount;
        }
        return testImage.distance(dbImages[modelInd]);
    }
};

//concrete methods
class BruteForce : public ClassificationMethod
{
public:
    BruteForce(std::vector<ImageInfo>& dbImages):ClassificationMethod("BF",dbImages){}
    int recognize(ImageInfo& testImage);
};

#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

class SvmClassifier : public ClassificationMethod
{
public:
    SvmClassifier(std::vector<ImageInfo>& dbImages);
    int recognize(ImageInfo& testImage);
private:
    cv::Ptr<cv::ml::SVM> svmClassifier;
};

class DirectedEnumeration : public ClassificationMethod
{
public:
    DirectedEnumeration(std::vector<ImageInfo>& faceImages, float falseAcceptRate = 0.01f, float threshold = 0, int imageCountToCheck = 0);
    ~DirectedEnumeration();

    int recognize(ImageInfo& testImage);

    bool isFoundLessThreshold;
    float bestDistance;

private:
    class ImageDist{
    public:
        float dist;
        int imageNum;

        ImageDist(float d = -1, int iNum = 0) :
            dist(d), imageNum(iNum)
        {}

        bool operator<(const ImageDist& rhs) const{
            return dist<rhs.dist;
        }

        static inline bool ComparerByNumber(const ImageDist& lhs, const ImageDist& rhs){
            return lhs.imageNum<rhs.imageNum;
        }
    };

    void init(int imageCountToCheck);

    float threshold;

    ImageDist* P_matrix;
    std::vector<int> startIndices;

    float* likelihoods;
    int* likelihood_indices;
};

#include <opencv2/core.hpp>
#include <opencv2/flann.hpp>

#ifdef USE_L2_DISTANCE
#define CURRENT_DISTANCE cvflann::L2
#else
#define CURRENT_DISTANCE cvflann::ChiSquareDistance
#endif
class FlannMethod : public ClassificationMethod
{
public:
    FlannMethod(std::vector<ImageInfo>& dbImages);
    ~FlannMethod();
    int recognize(ImageInfo& testImage);
private:
    float* dictionary_features;
    cvflann::Index<CURRENT_DISTANCE<float> > *flann_index;
};

//#define USE_NMSLIB

#ifdef USE_NMSLIB
#include "object.h"
#include "space/space_vector_gen.h"
#include "init.h"
#include "index.h"
#include "params.h"
#include "rangequery.h"
#include "knnquery.h"
#include "knnqueue.h"
#include "methodfactory.h"
#include "ztimer.h"

struct DistL2 {
  /*
   * Important: the function is const and arguments are const as well!!!
   */
  float operator()(const float* x, const float* y, size_t qty) const {
    float res = 0;
    for (size_t i = 0; i < qty; ++i) res+=(x[i]-y[i])*(x[i]-y[i]);
    return sqrt(res);
  }
};
class NmslibMethod : public ClassificationMethod
{
public:
    NmslibMethod(std::vector<ImageInfo>& dbImages);
    int recognize(ImageInfo& testImage);
    virtual void setImageCountToCheck(int imageCountToCheck);
private:
    similarity::VectorSpaceGen<float, CURRENT_DISTANCE<float>>  customSpace;
    similarity::ObjectVector    dataSet;
    similarity::Index<float>*   index;
    std::vector<float> queryData;
};
#endif // USE_NMSLIB
#endif // ANN_H
