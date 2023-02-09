// 1. Project.cpp
//  Project.cpp: implementation of CProject (main project class you will write)
//    Dept. Software Convergence, Kyung Hee University
//    Prof. Daeho Lee, nize@khu.ac.kr
//
#include "KhuCvApp.h"
#include "Project.h"

#ifdef _MSC_VER
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new DEBUG_NEW
#endif
#endif

CProject::CProject() {
    GetExecutionPath();
    
    char m_ExePt[256];
    strcat(m_ExePt, m_ExePath);
    
    strcat(m_ExePath, "/version-RFB-320_without_postprocessing.onnx");
    strcat(m_ExePt, "/mobilenetv2-7.onnx");
    
    m_pUltraface = new UltraFace(m_ExePath, 320, 240);
    m_MobileNet = cv::dnn::readNetFromONNX(m_ExePt);
}

CProject::~CProject() {
    delete m_pUltraface;
}

void CProject::GetExecutionPath() {
    wxFileName f(wxStandardPaths::Get().GetExecutablePath());
    wxString appPath(f.GetPath());

    wcscpy(m_ExePathUnicode, appPath);
    strcpy(m_ExePath, appPath.c_str());
}

// tracking vector initialize
std::vector<Tracker> m_idTrackers;
int Tracker::cnt = 0;

void CProject::Run(cv::Mat Input, cv::Mat& Output, bool bFirstRun, bool bVerbose) {
    if(bFirstRun){
        Tracker::cnt = 0;
        m_idTrackers.clear();
        
        for(auto layer : m_MobileNet.getLayerNames()){
            DlgPrintf("%s", layer.c_str());
        }
    }
    
    std::vector<FaceInfo> faceList;
    m_pUltraface->detect(Input, faceList);
 
    cv::Mat OutImage = Input.clone();
    
    for(auto list : faceList) {
        cv::Mat Roi = Input(cv::Rect(cv::Point(list.x1, list.y1), cv::Point(list.x2, list.y2)));
        cv::Mat DnnInput;
        cv::resize(Roi, DnnInput, cv::Size(224, 224), 0, 0, cv::INTER_AREA);
        cv::Mat inputBlob = cv::dnn::blobFromImage(DnnInput, 1 / 255., cv::Size(224, 224), cv::Scalar(128, 128, 128), false);
        m_MobileNet.setInput(inputBlob);
        // features cv::Mat 받아들이기
#ifndef  __APPLE__
        cv::Mat features = m_MobileNet.forward("mobilenetv20_features_pool0_fwd");
#else
        cv::Mat features = m_MobileNet.forward("onnx_node!mobilenetv20_features_pool0_fwd");
#endif
        
        cv::Mat cvFeature(40, 32, CV_32FC1);
        memcpy(cvFeature.data, features.data, 1280 * sizeof(float));
        
        Rect currentRt({list.x1,list.y1},{list.x2,list.y2});
        bool has_id = false;
        bool identified = false;
        
        double maxSimilarity = 0;
        Tracker *maxTracker;
        
        for(int i = 0; i < m_idTrackers.size(); ++i){
            if(m_idTrackers[i].rt.iou(currentRt) > 0.15) {
                identified = true;
                maxTracker = &m_idTrackers[i];
                break;
            }
            else{
                double similarity = m_idTrackers[i].GetCosineSimilarity(cvFeature);
                if(similarity > maxSimilarity) {
                    maxSimilarity = similarity;
                    maxTracker = &m_idTrackers[i];
                }
            }
        }
        if(maxSimilarity > 0.945) identified = true;
        else if(maxSimilarity > 0.93 && maxTracker->UnTracked > 5) identified = true;
        
        if(identified){
            has_id = true;
            maxTracker->featureList.push_back(cvFeature);
            maxTracker->UnTracked = (maxTracker->UnTracked > 1) ? -1 : 0;
            maxTracker->offset = maxTracker->offset * 0.5 + (currentRt.center() - maxTracker->rt.center()) * 0.5;
            maxTracker->rt = currentRt;
        }
        
        if(!has_id){
            Tracker tracker(currentRt, cvFeature);
            m_idTrackers.push_back(tracker);
        }
    }
    
    for(int i = 0; i < m_idTrackers.size(); ++i){
        // 삭제
        if(m_idTrackers[i].UnTracked > 18 || (m_idTrackers[i].offset == Point{0,0} && m_idTrackers[i].UnTracked > 5)) m_idTrackers.erase(m_idTrackers.begin() + i);
        
        // id 부여
        cv::Scalar color;
        std::stringstream m_id;
        m_id << m_idTrackers[i].T_id << "-" << m_idTrackers[i].UnTracked;
        
        // 초기화
        if(m_idTrackers[i].UnTracked == -1) {
            color = cv::Scalar(255,0,255);
            m_idTrackers[i].UnTracked = 1;
        }
        else if(m_idTrackers[i].UnTracked == 0) {
            color = cv::Scalar(0,0,255);
            m_idTrackers[i].UnTracked = 1;
        }
        else {
            color = cv::Scalar(255,255,0);
            Rect currentRt = m_idTrackers[i].rt;
            m_idTrackers[i].rt.LT = m_idTrackers[i].rt.LT + m_idTrackers[i].offset;
            m_idTrackers[i].rt.RB = m_idTrackers[i].rt.RB + m_idTrackers[i].offset;
            m_idTrackers[i].offset = m_idTrackers[i].offset * 0.5 + (m_idTrackers[i].rt.center()-currentRt.center()) * 0.5;
            m_idTrackers[i].UnTracked++;
        }
        
        // rect, id 그리기
        cv::rectangle(OutImage, cv::Point(m_idTrackers[i].rt.LT.x,m_idTrackers[i].rt.LT.y), cv::Point(m_idTrackers[i].rt.RB.x,m_idTrackers[i].rt.RB.y),color, 2);
        cv::putText(OutImage, m_id.str(), cv::Point(m_idTrackers[i].rt.LT.x,m_idTrackers[i].rt.LT.y-10),1,2,color,3);
    }
    
    /// m_idTrackers Vector의 capacity 맞춰주기
    m_idTrackers.shrink_to_fit();
    
    if(bVerbose)
        DisplayImage(OutImage, Input.cols, 0, false, true);
}
