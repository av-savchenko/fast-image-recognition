QT += core
QT -= gui

CONFIG += c++11
#QMAKE_CXXFLAGS_WARN_ON = -Wall -Wno-sign-compare -Wno-unused-variable

TARGET = recognition_testing_3
CONFIG += console
CONFIG -= app_bundle

DEFINES += QT_BUILD

#SETUP CORRECT PATH TO OPENCV AND NonMetricSpaceLib!!!
INCLUDEPATH += C:/Users/avsavchenko/Downloads/opencv/build/include/ #/Users/avsavchenko/Documents/my_soft/github/nmslib/similarity_search/include/
LIBS += -LC:/Users/avsavchenko/Downloads/opencv/build/x64/vc14/lib -LC:/Users/avsavchenko/Downloads/opencv/build/x64/vc14/bin \
    -lopencv_world412 #-L/usr/lib \#-L/usr/local/lib \
    #-L/Users/avsavchenko/Documents/my_soft/github/nmslib/similarity_search/release \
    #-lNonMetricSpaceLib -lopencv_calib3d -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_shape -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videoio -lopencv_videostab

TEMPLATE = app

SOURCES += main.cpp \
    ImageTesting.cpp \
    classification.cpp \
    db_features.cpp \
    ann.cpp \
    video.cpp

HEADERS += \
    db.h \
    db_features.h \
    ann.h
