# KhuCv-FaceTracking
<p>Adding Face Tracking Function on KhuCv. I modified and added codes only in 'Project.cpp' and in 'Project.h'. </p>

<h3> <프로젝트 설명> </h3>

- 진행 기간 : 2023.01.01 ~ 2023.02.20(약 2개월)

<p>KhuCv는 실시간 영상 혹은 동영상에서 사람 얼굴을 인식하고 추적합니다. 얼굴 추적은 DeepSORT방식을 이용하였습니다. 기본적으로 IOU를 이용하여 추적하였으며 IOU탐지를 못한 경우, 이전 프레임의 Face Rectangle과 얼굴 유사도가 가장 크면서, 해당 얼굴과 먼 거리에 있지 않은 Face Rectangle을 얼굴에 부여하였습니다. 얼굴 특징은 classfication mobileNet에서 프레임의 특징을 잡아내는 부분까지만 foward하여 추출해 사용하였으며, 얼굴 유사도 비교는 cosine Similarity로 비교하였습니다.</p>

<h3> <기능 설명> </h3>

<p> 새로 나타난 얼굴에는 id를 부여하고 추적합니다. Face Rectangle의 바로 위를 보시면 'id-UnTracked유무'가 나타나 있습니다. 여기서 UnTracked와 Face Rectangle의 색깔은 다음과 같은 의미를 지닙니다.</p>

- 해당 얼굴을 연속적으로 추적하는 경우 -> UnTracked : 0 , Face Rectangle : 빨강색
- 추적이 안 됐을 경우 -> UnTracked : 추적 못한 프레임 수 만큼의 숫자 1~25 , Face Rectangle : 민트색
- 추적 못한 Face Rectangle을 추적한 경우 -> UnTracked : -1 , Face Rectangle : 보라색


<h4> Flow Chart : </h4>
<div style="display:flex;">
<img src="https://github.com/WKlee0607/KhuCv-FaceTracking/blob/main/Flow%20Charts/Tracking%20Vector%20Update.png" style="height:300px;width:250px;"/>
<img src="https://github.com/WKlee0607/KhuCv-FaceTracking/blob/main/Flow%20Charts/Drawing%20Face%20Rect%20in%20Video.png" style="height:300px;width:250px;"/>


<h3> <사용법> </h3>

1. 참고 링크에 따라 KhuCv를 빌드한다.
2. 참고 링크에 있는 Face Detection DNN 파일과 Face feature DNN 을 다운받아 KhuCv의 Run파일에 넣어준다.
3. KhuCv의 KhuCvApp 파일을 위의 KhuCvApp 파일로 변경하여 사용한다.
4. KhuCv어플리케이션을 동작시킨 뒤, 사람이 나오는 동영상을 작동시킨다.


<h3> -참고- </h3> 

- <a href="https://github.com/NizeLee/KhuCv_mdi">How to build KhuCv</a>
- <a href="https://github.com/NizeLee/KhuCv_mdi/tree/main/Samples/01_Face_detection_opencv"> How to apply Face Detection DNN on KhuCv</a>
- <a href="https://velog.io/@wklee0607_/6.-KhuCv-FaceTracking-Using-DeepSORT"> How to apply Face feature DNN on KhuCv </a>


<h3> -Project 총정리- </h3>

- <a href="https://velog.io/@wklee0607_/series/2022-23-WVacation-CppStudy"> WKlee0607's Velog address of FaceTracking Project</a>




<br><br>

<h3> - Face Tracking Video Preview- </h3>
<img src="https://github.com/WKlee0607/KhuCv-FaceTracking/blob/main/Previews.gif"/>
