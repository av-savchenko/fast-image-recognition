#ifndef DB_FEATURES_H
#define DB_FEATURES_H

#include <vector>
#include <string>
#include <unordered_map>

#include "db.h"

//#define USE_PCA

#define USE_L2_DISTANCE

typedef std::vector<float> FeaturesVector;
typedef std::vector<std::vector<FeaturesVector> > ImagesDatabase;

float feature_distance(const FeaturesVector& lhs, const FeaturesVector& rhs, int start_pos = 0, int end_pos = FEATURES_COUNT);

class ImageInfo{
public:
        ImageInfo(int no, int ind, const FeaturesVector& feat) :classNo(no), indexInDatabase(ind), features(feat)
        {
        }
        float distance(const ImageInfo& rhs, int start_pos = 0, int end_pos = FEATURES_COUNT) const{
                return feature_distance(features, rhs.features,start_pos,end_pos);
        }
        const int classNo, indexInDatabase;
        const FeaturesVector& features;
};

int loadImages(ImagesDatabase& imagesDb, std::string features_file, std::unordered_map<std::string, int>& person2indexMap, bool early_stop=false);
void getTrainingAndTestImages(const ImagesDatabase& totalImages, std::vector<ImageInfo>& dbImages, std::vector<ImageInfo>& testImages, bool randomize=true);
int recognize_image_bf(const std::vector<ImageInfo>& dbImages, const ImageInfo& testImageInfo, int max_features=0);
void extractPCA(const ImagesDatabase& orig_database, ImagesDatabase& new_database);

#endif // DB_FEATURES_H
