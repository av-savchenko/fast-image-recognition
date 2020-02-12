#include "ann.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <unordered_map>
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


typedef std::map<std::string, std::vector<std::vector<FeaturesVector>>> MapOfVideos;

#ifdef USE_LCNN
const std::string VIDEO_FEATURES_FILE =
"dnn_features_my_best_rate5.txt";
const std::string TRAIN_FEATURES_FILE =
"dnn_features_my_all_best.txt";
#else
const std::string VIDEO_FEATURES_FILE =
"vgg_mean_dnn_features.txt";
const std::string TRAIN_FEATURES_FILE =
"dnn_vgg_features_all_mean.txt";
#endif
void loadVideos(MapOfVideos& dbVideos){
    int total_images = 0, total_videos = 0;
    ifstream ifs(VIDEO_FEATURES_FILE);
    if (ifs){
        while (ifs){
            std::string fileName, personName, feat_str;
            if (!getline(ifs, personName))
                break;
            personName.erase(0, personName.find_first_not_of(" \t\n\r\f\v\r\n"));
            int videos_count;
            ifs >> videos_count;
            /*cout << "["<<personName.c_str() << "] video_cnt=" << videos_count;
            print_endl;*/
            dbVideos.insert(std::make_pair(personName, vector<vector<FeaturesVector>>()));
            vector<vector<FeaturesVector>>& person_videos = dbVideos[personName];
            person_videos.resize(videos_count);

            for (int i = 0; i < videos_count; ++i){
                int frames_count;
                ifs >> frames_count;
                //cout << frames_count << endl;
                person_videos[i].resize(frames_count);
                //read end line
                if (!getline(ifs, fileName))
                    break;

                for (int j = 0; j < frames_count; ++j){
                    if (!getline(ifs, fileName))
                        break;
                    if (!getline(ifs, feat_str))
                        break;
                    istringstream iss(feat_str);
                    person_videos[i][j].resize(FEATURES_COUNT);

                    float dfeature, sum = 0;
                    for (int k = 0; k < FEATURES_COUNT; ++k){
                        iss >> dfeature;
                        //#ifdef USE_VGG
                        if (abs(dfeature) < 0.0001)
                            dfeature = 0;
                        //#endif
                        person_videos[i][j][k] = dfeature;
                        sum += dfeature*dfeature;
                    }
//#if defined(USE_PCA)
#ifdef USE_L2_DISTANCE
                    sum = sqrt(sum);
#endif
                    for (int k = 0; k < FEATURES_COUNT; ++k)
                        person_videos[i][j][k] /= sum;
//#endif
                    //cout << fileName << ' ' << personName << ' ' << features[0] << ' ' << features[FEATURES_COUNT - 1] << '\n';
                    ++total_images;
                }
            }
            total_videos += videos_count;
            dbVideos.insert(std::make_pair(personName, person_videos));
        }
        ifs.close();
        cout << "total size=" << dbVideos.size() << " totalVideos=" << total_videos << " totalImages=" << total_images;
        print_endl;
    }

#if 0
    //write pca
    Mat mat_features(total_images, FEATURES_COUNT, CV_32F);
    int ind = 0;
    for (MapOfVideos::iterator iter = dbVideos.begin(); iter != dbVideos.end(); ++iter){
        for (vector<FeaturesVector>& video : iter->second){
            for (FeaturesVector& features : video){
                for (int j = 0; j < FEATURES_COUNT; ++j){
                    //db_features[i1*featuresCount + j] =
                    mat_features.at<float>(ind, j) =
                        features[j];
                }
                ++ind;
            }

        }
    }

    PCA pca;
    if (!load(PCA_FILENAME, pca)){
        std::cout << "error loading PCA\n";
        exit(0);
    }
    Mat mat_projection_result = pca.project(mat_features);
    cout << "rows=" << mat_projection_result.rows << " cols=" << mat_projection_result.cols << endl;

    ind = 0;
    for (MapOfVideos::iterator iter = dbVideos.begin(); iter != dbVideos.end(); ++iter){
        for (vector<FeaturesVector>& video : iter->second){
            for (FeaturesVector& features : video){
                for (int j = 0; j < FEATURES_COUNT; ++j){
                    features[j] = mat_projection_result.at<float>(ind, j);
                }
                ++ind;
            }

        }
    }
    ofstream of("E:\\avsavchenko\\images\\tmp\\ytf_dnn_features_my_best_pca_rate5.txt");
    if (of){
        cout << "begin write file\n";
        for (MapOfVideos::iterator iter = dbVideos.begin(); iter != dbVideos.end(); ++iter){
            of << iter->first << endl << iter->second.size() << endl;
            for (vector<FeaturesVector>& video : iter->second){
                of << video.size() << endl;
                for (FeaturesVector& face : video){
                    of << "fileName_" << iter->first << endl;
                    for (int i = 0; i < FEATURES_COUNT; ++i)
                        of << face[i] << ' ';
                    of << endl;
                }
            }
        }
        cout << "end write file\n";
        of.close();
    }
#endif
}
void testYTFRecognition(){
    ImagesDatabase totalImages;
    unordered_map<string, int> person2indexMap;
#if defined(USE_PCA) && 0
    ImagesDatabase orig_database;
    loadImages(orig_database, TRAIN_FEATURES_FILE, person2indexMap);
    extractPCA(orig_database, totalImages);
#else
    loadImages(totalImages,TRAIN_FEATURES_FILE,person2indexMap);
#endif

    vector<string> dbNames;
    dbNames.reserve(person2indexMap.size());
    for (auto keyValue : person2indexMap){
        dbNames.push_back(keyValue.first);
    }

    MapOfVideos videos;
    loadVideos(videos);

    vector<string> videoNames;
    videoNames.reserve(videos.size());
    for (auto keyValue : videos){
        videoNames.push_back(keyValue.first);
    }

    vector<string> commonNames(videos.size());
    sort(videoNames.begin(), videoNames.end());
    sort(dbNames.begin(), dbNames.end());
    /*for (auto name : dbNames){
        cout << '[' << name << "] ";
    }
    cout << endl;
    for (auto name : videoNames){
        cout << '[' << name << "] ";
    }
    cout << endl;
    */
    auto it = std::set_intersection(videoNames.begin(), videoNames.end(), dbNames.begin(), dbNames.end(), commonNames.begin());
    commonNames.resize(it - commonNames.begin());
    /*cout << "res\n";
    for (auto name : commonNames){
        cout << '[' << name << "] ";
    }
    cout << endl;*/

    cout << "lfw names size=" << dbNames.size() << " YTF names size=" << videoNames.size() << " common names size=" << commonNames.size();
    print_endl;

    vector<string> listToRemove;
    std::set_symmetric_difference(videoNames.begin(), videoNames.end(), dbNames.begin(), dbNames.end(), back_inserter(listToRemove));
    for (string needToRemove : listToRemove){
        person2indexMap.erase(needToRemove);
        videos.erase(needToRemove);
    }

    unordered_map<string, int> person2indexMapNew;
    std::vector<ImageInfo> testImages;
    int class_index = 0;
    for (MapOfVideos::iterator iter = videos.begin(); iter != videos.end(); ++iter){
        person2indexMapNew.insert(make_pair(iter->first, class_index));
        for (vector<FeaturesVector>& video : iter->second){
            //for (int ind = 0; ind < video.size();++ind)
            for (int ind = 0; ind < video.size();ind+=10)
            {
                testImages.push_back(ImageInfo(class_index, ind, video[ind]));
            }
        }
        ++class_index;
    }

    std::vector<ImageInfo> dbImages;
    int lfw_size = 0;
    for (auto& person_index:person2indexMap){
        int new_class_ind = person2indexMapNew[person_index.first];
        int class_ind = person_index.second;
        for (int ind = 0; ind < totalImages[class_ind].size();++ind){
            dbImages.push_back(ImageInfo(new_class_ind, ind, totalImages[class_ind][ind]));
        }
        ++lfw_size;
    }
    cout << "lfw names size=" << lfw_size << " YTF names size=" << videos.size() << " removed=" << listToRemove.size();
    print_endl;

    cout<< "dbSize=" << dbImages.size() <<" testSize=" << testImages.size();
    print_endl;

    BruteForce bf(dbImages);
    bf.testSetRecognition(testImages);

    SvmClassifier svm_c(dbImages);
    svm_c.testSetRecognition(testImages);

    vector<ClassificationMethod*> methods;
    //methods.push_back(new FlannMethod(dbImages));
    //methods.push_back(new NmslibMethod(dbImages));
    methods.push_back(new DirectedEnumeration(dbImages,0.01));
    //for (double ratio = 0.025; ratio <= 0.5/*1.001*/; ratio += 0.025)
    for (double ratio = 0.1; ratio <= 0.7/*1.001*/; ratio += 0.1)
    //for (double ratio = 1.0; ratio >= 0.1/*1.001*/; ratio -= 0.1)
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
}
