//  Project.h: interface of CProject (main project class you will write)
//    Dept. Software Convergence, Kyung Hee University
//    Prof. Daeho Lee, nize@khu.ac.kr
//

#pragma once
#include "cv_dnn_ultraface.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>

class CProject
{
    cv::Mat m_PreviousImage;
public:
    char m_ExePath[256];
    wchar_t m_ExePathUnicode[256];

    CProject();
    ~CProject();
    void GetExecutionPath();
    void Run(cv::Mat Input, cv::Mat& Output, bool bFirstRun, bool bVerbose);

    UltraFace *m_pUltraface;
    cv::dnn::Net m_MobileNet;
};


class Point{
public:
    float x,y;
    Point(){}
    Point(float a, float b): x(a), y(b){}
    Point operator-(Point other){return {x-other.x, y-other.y};}
    Point operator+(Point other){return {x+other.x, y+other.y};}
    Point operator*(float a){return {x * a, y * a};}
    bool operator==(Point other){return (other.x==x && other.y==y) ? true : false;}
    double mag(){return sqrt(x*x + y*y);}
};

class Rect{
public:
    Point LT,RB;
    Rect(){}
    Rect(Point lt, Point rb): LT(lt), RB(rb){}
    float iou(Rect other){ // iou가 0을 가질 경우, 이 사각형은 inter을 갖지 않음. 즉, 겹치지 않음.
        Rect inter = this->intersection(other);
        float thisArea = this->width() * this->height();
        float otherArea = other.width() * other.height();
        float interArea = inter.width() * inter.height();
        return interArea/(thisArea+otherArea-interArea);
    }
    Rect intersection(Rect other){
        float x1 = std::max(LT.x, other.LT.x);
        float y1 = std::max(LT.y, other.LT.y);
        float x2 = std::min(RB.x, other.RB.x);
        float y2 = std::min(RB.y, other.RB.y);
        return {{x1,y1},{x2,y2}};
    }
    float width(){
        return RB.x-LT.x >= 0 ? RB.x-LT.x : 0;
    }
    float height(){
        return RB.y-LT.y >= 0 ? RB.y-LT.y : 0;
    }
    Point center(){
        return {(LT.x + RB.x)/2, (LT.y + RB.y)/2};
    }
};


class Tracker{
public:
    static int cnt;
    int T_id;
    int UnTracked = 0;
    
    Point offset;
    Rect rt;
    std::vector<cv::Mat> featureList;
    
    Tracker(){}
    Tracker(Rect current, cv::Mat cvFeature):T_id(cnt++), rt(current), offset(0,0){
        featureList.push_back(cvFeature);
    }
    double GetCosineSimilarity(cv::Mat feature){
        double maxSimilarity = 0;
        for(int i = featureList.size() - 1; i >= 0 && i >= (int)(featureList.size()) - 5;--i){
            double Similarity = 0;
            double A = 0, B = 0;
            for(int row = 0; row < featureList[i].rows; ++row)
                for(int col = 0; col < featureList[i].cols; ++col){
                    Similarity += featureList[i].at<float>(row,col) * feature.at<float>(row,col); // 내적
                    A += featureList[i].at<float>(row, col) * featureList[i].at<float>(row, col); // A크기
                    B += feature.at<float>(row, col) * feature.at<float>(row,col); // B크기
                }
            if(sqrt(A) * sqrt(B) > 0) Similarity /= sqrt(A) * sqrt(B);
            else Similarity = 0;
            
            if(Similarity > maxSimilarity) maxSimilarity = Similarity;
        }
        return maxSimilarity;
    }
};
