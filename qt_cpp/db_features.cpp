#include "db_features.h"

#include <unordered_map>
#include <map>
#include <fstream>
#include <sstream>

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


float feature_distance(const FeaturesVector& lhs, const FeaturesVector& rhs, int start_pos, int end_pos){
        float dist = 0;
        for (int i = start_pos; i < end_pos; ++i)
#ifdef USE_L2_DISTANCE
            dist += (lhs[i] - rhs[i])*(lhs[i] - rhs[i]);
            //dist += abs(lhs[i] - rhs[i]);
#else
            if((lhs[i] + rhs[i])>0){
#if 1
                dist += (lhs[i] - rhs[i])*(lhs[i] - rhs[i])/(lhs[i] + rhs[i]);
#else
                if(lhs[i]>0)
                    dist += lhs[i] *log(2*lhs[i]/(lhs[i] + rhs[i]));
                if(rhs[i]>0)
                    dist+=rhs[i] *log(2*rhs[i]/(lhs[i] + rhs[i]));
#endif
            }
#endif
        dist /= (end_pos-start_pos);
        return dist;
}

int loadImages(ImagesDatabase& imagesDb, std::string features_file, unordered_map<string, int>& person2indexMap, bool early_stop)
{
        person2indexMap.clear();
        ifstream ifs(features_file);
        int total_images=0;
        if (ifs){
            while (ifs){
                    std::string fileName, personName, feat_str;
                    if (!getline(ifs, fileName))
                            break;
                    if (!getline(ifs, personName))
                            break;
                    if (!getline(ifs, feat_str))
                            break;
                    //cout << fileName.c_str() << ' ' << personName.c_str() << endl;
                    personName.erase(0, personName.find_first_not_of(" \t\n\r\f\v\r\n"));
#ifdef USE_CALTECH
                    if(personName.find("BACKGROUND_Google")!=string::npos ||
                            personName.find("257.clutter")!=string::npos)
                        continue;
#endif
                    if (person2indexMap.find(personName) == person2indexMap.end()){
#ifdef USE_CASIA
                        if(/*early_stop && */person2indexMap.size()>=1000)
                        //if(person2indexMap.size()>=256)
                            break;
#endif
                        person2indexMap.insert(std::make_pair(personName, person2indexMap.size()));
                        imagesDb.push_back(std::vector<FeaturesVector>());
                    }
                    vector<FeaturesVector>& currentPersonImages = imagesDb[person2indexMap[personName]];
                    currentPersonImages.push_back(FeaturesVector());
                    FeaturesVector& features = currentPersonImages.back();
                    features.resize(FEATURES_COUNT);

                    istringstream iss(feat_str);
                    //of << fileName << endl << personName << endl;
                    float dfeature, sum = 0;
                    for (int i = 0; i < FEATURES_COUNT; ++i){
                            iss >> dfeature;
                            //#ifdef USE_VGG
                            if (abs(dfeature) < 0.0001)
                                    dfeature = 0;
                            //#endif
                            //dfeature+=0.001;
                            features[i] = dfeature;
#if defined(USE_L2_DISTANCE) && 1
                            sum += dfeature*dfeature;
#else
                            sum+=dfeature;
#endif
                    }
//#ifdef USE_PCA
#if defined(USE_L2_DISTANCE) && 1
                    sum = sqrt(sum);
#endif
                    for (int i = 0; i < FEATURES_COUNT; ++i)
                        features[i] /= sum;
//#endif
                    ++total_images;
            }
            ifs.close();

            double avg_count=0;
            for(int i=0;i<imagesDb.size();++i)
                avg_count+=imagesDb[i].size();
            avg_count/=imagesDb.size();
            cout << "total size=" << imagesDb.size() << " totalImages=" << total_images<<" avg_count="<<avg_count;
            print_endl;
            //cout << imagesDb[0][0][0] << ' ' << imagesDb[0][0][FEATURES_COUNT - 1] << ' ' << imagesDb.back().back()[0] << ' ' << imagesDb.back().back()[FEATURES_COUNT - 1];
    }
    return total_images;
}
void getTrainingAndTestImages(const ImagesDatabase& totalImages, std::vector<ImageInfo>& dbImages, std::vector<ImageInfo>& testImages, bool randomize)
{
        const int INDICES_COUNT = 400;
        static int indices[INDICES_COUNT];
        for (int i = 0; i<INDICES_COUNT; ++i)
                indices[i] = i;
        if (randomize)
                std::random_shuffle(indices, indices + INDICES_COUNT);

        dbImages.clear();
        testImages.clear();
        int indexInDatabase=0;
        for (int class_ind=0;class_ind<totalImages.size();++class_ind){
                int currentFaceCount = totalImages[class_ind].size();
                //cout << currentFaceCount << endl;
#if defined(USE_CALTECH) && 1
                int db_size = 30;
#else
                float size_f = currentFaceCount*FRACTION;
                int db_size = (int)(ceil(size_f));
                //int db_size = (int)(floor(size_f));
                //if (rand() & 1) db_size = (int)size_f;
                if (db_size == currentFaceCount)
                        db_size = currentFaceCount - 1;
                if (db_size == 0)
                        db_size = 1;
#endif
                //std::vector<FaceImage*> dbFaces(iter->second.begin(),iter->second.begin()+db_size);
                //std::vector<FaceImage*> testFaces(iter->second.begin()+db_size,iter->second.end());
                int ind = 0;
                for (int i = 0; i < INDICES_COUNT; ++i)
                {
                        if (indices[i] < currentFaceCount){
                                ImageInfo info(class_ind, indexInDatabase+indices[i], totalImages[class_ind][indices[i]]);
                                if (ind < db_size)
                                        dbImages.push_back(info);
                                else
                                        testImages.push_back(info);
                                ++ind;
                        }
                }
                //cout << db_size << ' ' << size_f<<' '<<currentFaceCount << ' '<<ind<<'\n';
                indexInDatabase+=currentFaceCount;
        }
        //cout << "total: " << dbImages.size() << ' ' << testImages.size() << '\n';
}

#include<opencv2/core.hpp>
using namespace cv;

#ifdef USE_PCA
const string PCA_FILENAME=
#ifdef USE_LCNN
"E:\\avsavchenko\\images\\tmp\\pca_casia.xml";
#else
"E:\\avsavchenko\\images\\tmp\\pca_vgg_casia.xml";
#endif
void save(const string &file_name,cv::PCA& pca_)
{
        FileStorage fs(file_name,FileStorage::WRITE);
        //pca_.write(fs);
        fs << "mean" << pca_.mean;
        fs << "e_vectors" << pca_.eigenvectors;
        fs << "e_values" << pca_.eigenvalues;
        fs.release();
        cout<<"pca saved\n";
}

int load(const string &file_name,cv::PCA& pca_)
{
        int success= 0;
        FileStorage fs(file_name,FileStorage::READ);
        if(fs.isOpened()){
                //pca_.read(fs);
                fs["mean"] >> pca_.mean;
                fs["e_vectors"] >> pca_.eigenvectors ;
                fs["e_values"] >> pca_.eigenvalues ;
                fs.release();
                success=1;
                cout<<"pca loaded "<<pca_.mean.cols<<"\n";
        }
        return success;
}
void saveImages(ImagesDatabase& imagesDb){
        ofstream of(PCA_FEATURES_FILE_NAME);
        if (of){
                cout << "begin write file\n";
                for (auto& iter : imagesDb){
                        for (FeaturesVector& features : iter.second){
                                of << "filename "<<iter.first << endl;
                                of << iter.first << endl;
                                for (int i = 0; i < FEATURES_COUNT; ++i)
                                        of << features[i] << ' ';
                                of << endl;
                        }
                }
                cout << "end write file\n";
                of.close();
        }
}

void extractPCA(const ImagesDatabase& orig_database, ImagesDatabase& new_database){
        int total_images_count = 0;
        for (auto& person : orig_database){
                total_images_count += person.second.size();
        }
        Mat mat_features(total_images_count, FEATURES_COUNT, CV_32F);
        int ind = 0;
        for (auto& person : orig_database){
                for (const FeaturesVector& features : person.second){
                        for (int j = 0; j < FEATURES_COUNT; ++j){
                                //db_features[i1*featuresCount + j] =
                                mat_features.at<float>(ind, j) =
                                        features[j];
                        }
                        ++ind;
                }
        }

        PCA pca;
        if(!load(PCA_FILENAME,pca)){
                pca=pca(mat_features, Mat(), CV_PCA_DATA_AS_ROW, 0);
                save(PCA_FILENAME,pca);
        }
        else{
                /*Mat mean_features;
                reduce(mat_features, mean_features, 0, REDUCE_AVG);
                cout << "mean rows=" << mean_features.rows << " cols=" << mean_features.cols << endl;
                pca.mean = mean_features;*/
        }
        Mat mat_projection_result=pca.project(mat_features);
        cout << "rows="<<mat_projection_result.rows << " cols=" << mat_projection_result.cols << endl;
        /*cout << "example: " << mat_features.at<float>(0, 0) << " " << mat_projection_result.at<float>(0, 0) << endl;
        cout << "example: " << mat_features.at<float>(0, 1) << " " << mat_projection_result.at<float>(0, 1) << endl;
        cout << "example: " << mat_features.at<float>(0, FEATURES_COUNT) << " " << mat_projection_result.at<float>(0, FEATURES_COUNT) << endl;*/

        ind = 0;
        for (auto& person : orig_database){
                ImagesDatabase::mapped_type class_features_list;
                class_features_list.reserve(person.second.size());
                for (const FeaturesVector& features : person.second){
                        class_features_list.push_back(FeaturesVector());
                        class_features_list.back().resize(FEATURES_COUNT);
                        for (int j = 0; j < FEATURES_COUNT; ++j){
                                //db_features[i1*featuresCount + j] =
                                class_features_list.back()[j] = mat_projection_result.at<float>(ind, j);
                        }

                        ++ind;
                }
                new_database.insert(make_pair(person.first, class_features_list));
        }
        saveImages(new_database);
}
#else
void extractPCA(const ImagesDatabase& orig_database, ImagesDatabase& new_database){
        int total_images_count = 0;
        for (auto& person : orig_database){
                total_images_count += person.size();
        }
        Mat mat_features(total_images_count, FEATURES_COUNT, CV_32F);
        int ind = 0;
        for (auto& person : orig_database){
                for (const FeaturesVector& features : person){
                        for (int j = 0; j < FEATURES_COUNT; ++j){
                                //db_features[i1*featuresCount + j] =
                                mat_features.at<float>(ind, j) =
                                        features[j];
                        }
                        ++ind;
                }
        }

        PCA pca(mat_features, Mat(), PCA::DATA_AS_ROW, 0);
        Mat mat_projection_result=pca.project(mat_features);
        qDebug() << "rows="<<mat_projection_result.rows << " cols=" << mat_projection_result.cols;

        ind = 0;
        new_database.reserve(orig_database.size());
        for (auto& person : orig_database){
                ImagesDatabase::value_type class_features_list;
                class_features_list.reserve(person.size());
                for (int i=0;i<person.size();++i){
                        class_features_list.push_back(FeaturesVector());
                        class_features_list.back().resize(FEATURES_COUNT);
                        for (int j = 0; j < FEATURES_COUNT; ++j){
                                //db_features[i1*featuresCount + j] =
                                class_features_list.back()[j] = mat_projection_result.at<float>(ind, j);
                        }

                        ++ind;
                }
                new_database.push_back(class_features_list);
        }
        //cout << "rows="<<mat_projection_result.rows << " cols=" << mat_projection_result.cols << endl;
        /*cout << "example: " << mat_features.at<float>(0, 0) << " " << mat_projection_result.at<float>(0, 0) << endl;
        cout << "example: " << mat_features.at<float>(0, 1) << " " << mat_projection_result.at<float>(0, 1) << endl;
        cout << "example: " << mat_features.at<float>(0, FEATURES_COUNT) << " " << mat_projection_result.at<float>(0, FEATURES_COUNT) << endl;*/
}
#endif


int recognize_image_bf(const vector<ImageInfo>& dbImages, const ImageInfo& testImageInfo, int max_features){
    if(max_features==0)
        max_features=FEATURES_COUNT;
    int bestInd = -1;
    double bestDist = 100000;
    vector<double> distances(dbImages.size());
    for (int j = 0; j < dbImages.size(); ++j){
            distances[j] =
                    //testImageInfo.distance(dbImages[j]);
                    testImageInfo.distance(dbImages[j],0,max_features);
            if (distances[j] < bestDist){
                    bestDist = distances[j];
                    bestInd = j;
            }
    }
    return bestInd;
}
