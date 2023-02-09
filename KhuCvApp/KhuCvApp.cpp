//  KhuCvApp.cpp: implementation of CKhuCvApp
//	Dept. Software Convergence, Kyung Hee University
//	Prof. Daeho Lee, nize@khu.ac.kr
//

#include "KhuCvApp.h"

#ifdef _MSC_VER
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new DEBUG_NEW
#endif
#endif

#include <wx/display.h>

bool CKhuCvApp::OnInit()
{
    int language = wxLANGUAGE_KOREAN;
    m_locale.Init(language);

    m_pMainFrame = new CMainFrame(wxT("KhuCV App"));
#ifndef  __APPLE__ 
    m_pMainFrame->Maximize(true);
#endif
    m_pMainFrame->Show();

    m_pMainFrame->DragAcceptFiles(true);
    
    wxSize MainFrameSize = m_pMainFrame->GetSize();
    
    
    wxPoint DlgPos;
    wxSize DlgSize(350, 350);
#ifdef __APPLE__
    DlgPos = {0, 100+480};//{5, wxDisplay().GetGeometry().GetHeight()-DlgSize.y};
#else
    DlgPos = {5, MainFrameSize.y-200-DlgSize.y};
#endif
    
    m_pMainDialog = new CMainDialog(m_pMainFrame, wxID_ANY, wxT("Main Dialog"), DlgPos, DlgSize, wxDEFAULT_DIALOG_STYLE);
    m_pMainDialog->Show();
   
    return true;
}

CKhuCvApp::~CKhuCvApp() {
    
}

IMPLEMENT_APP(CKhuCvApp)

void NewFileOpen(const char* fileName, cv::Mat cvImage, int nPosX, int nPosY) {
    CMainFrame* pMainFrame = wxGetApp().m_pMainFrame;
    CChildFrame* pChildFrame = new CChildFrame(pMainFrame, wxID_ANY, fileName);

    CKcImage kcImage = CKcImage(cvImage, wxPoint(nPosX, nPosY));
    pChildFrame->m_ImageList.push_back(kcImage);
    pChildFrame->Show();
}

void DisplayImage(cv::Mat cvImage, int nPosX, int nPosY, bool bErase, bool bClearPos) {
    CMainFrame* pMainFrame = wxGetApp().m_pMainFrame;
    CChildFrame* pChildFrame = (CChildFrame*)(pMainFrame->GetActiveChild());
    cv::Mat cvCloneImage = cvImage.clone();

    CKcImage kcImage;
    if (cvCloneImage.type() != CV_8UC3) {
        if (cvCloneImage.channels() == 1) {
            cv::Mat cvNewImage(cvCloneImage.rows, cvCloneImage.cols, CV_8UC3);
            cv::normalize(cvCloneImage, cvCloneImage, 0, 255, cv::NORM_MINMAX);
            cvCloneImage.convertTo(cvCloneImage, CV_8UC1);
            cv::cvtColor(cvCloneImage, cvNewImage, cv::COLOR_GRAY2BGR);
            kcImage = CKcImage(cvNewImage, wxPoint(nPosX, nPosY));
        }
        else if(cvCloneImage.channels() == 3) {
            cv::Mat cvNewImage;
            cv::Mat BGR[3];

            cv::split(cvCloneImage, BGR);
            for(int i = 0 ; i < 3 ; ++i) {
                //cv::normalize(BGR[i], BGR[i], 0, 255, cv::NORM_MINMAX);
                BGR[i].convertTo(BGR[i], CV_8UC1);
            }
            cv::merge(BGR, 3, cvNewImage);

            kcImage = CKcImage(cvNewImage, wxPoint(nPosX, nPosY));
        }
        else
            return;
    }
    else
        kcImage = CKcImage(cvCloneImage, wxPoint(nPosX, nPosY));

    if(bClearPos) {
        auto NewEnd = std::remove_if(pChildFrame->m_ImageList.begin(), pChildFrame->m_ImageList.end(), [=](CKcImage kcImage){return kcImage.pos == wxPoint(nPosX, nPosY);});
        pChildFrame->m_ImageList.erase(NewEnd, pChildFrame->m_ImageList.end());
    }

    pChildFrame->m_ImageList.push_back(kcImage);
    pChildFrame->m_pClientView->m_nLastSelImageNum = pChildFrame->m_ImageList.size()-1;
    pChildFrame->m_pClientView->Refresh(bErase);
    pChildFrame->m_pClientView->Update();
}

CKcImage GetLastSelImage() {
    CMainFrame* pMainFrame = wxGetApp().m_pMainFrame;
    CChildFrame* pChildFrame = (CChildFrame*)(pMainFrame->GetActiveChild());

    int nCurrentGrabImageNum = pChildFrame->m_pClientView->m_nLastSelImageNum;

    if (nCurrentGrabImageNum < 0) {

        if(pChildFrame->m_ImageList.size() > 0)
            return pChildFrame->m_ImageList[pChildFrame->m_ImageList.size()-1];

        return CKcImage();
    }
    else if (nCurrentGrabImageNum >= pChildFrame->m_ImageList.size()) {

        if (pChildFrame->m_ImageList.size() > 0)
            return pChildFrame->m_ImageList[pChildFrame->m_ImageList.size() - 1];

        return CKcImage();
    }

    return pChildFrame->m_ImageList[nCurrentGrabImageNum];
}

CMainDialog* GetMainDialog() {
    return wxGetApp().m_pMainDialog;
}

void DlgPrintf(const char* ptr, ...) {
    unsigned int Num;

    char ach[1024];
    va_list args;
    va_start(args, ptr);
    vsnprintf(ach, 1024, ptr, args);

    wxString msg = ach;
    
    CMainFrame* pMainFrame = wxGetApp().m_pMainFrame;

    Num = pMainFrame->GetPrintListBox()->GetCount();
    pMainFrame->GetPrintListBox()->InsertItems(1, &msg, Num);
    pMainFrame->GetPrintListBox()->SetSelection(Num);
}
