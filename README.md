# Hard-Hat-Detact-Model  
Google Colab에서 제작하였습니다   
기본 콘텐츠로 건너뛰기  
안전모 탐지 모델  
2024년 04월 30일 로봇응용SW전문가 과정_유현우  
안전모 탐지 모델이란?  
작업 현장에서 안전모를 착용하고 있는지를 탐지  
  
안전모 미착용한 사람을 발견하여 착용하도록 함  
  
안전모 탐지를 위한 데이터 세트  
https://public.roboflow.com/object-detection/hard-hat-workers
사용 AI 모델
yolo5 . 가장 빠른 인스톨로 실시간 탐지에 유리하다. . ON 알고리즘 중 가장 활발하게 연구활동이 이루어지고 있다.  
YOLOv5u는 객체 감지 방법론의 진보를 나타냅니다. 기본 아키텍처에서 출발한 YOLOv5 모델( Ultralytics)의 기본 아키텍처에서 유래한  
 YOLOv5u는 이전에 소개된 앵커가 없고 물체가 없는 분할 헤드를 통합하여 YOLOv8 모델에 도입된 기능입니다. 이러한 조정은 모델의 아키텍처 
 를 개선하여 물체 감지 작업에서 정확도와 속도 간의 트레이드 오프가 개선되었습니다. 경험적 결과와 그로부터 도출된 특징을 고려할 때,  
 YOLOv5u는 연구와 실제 애플리케이션 모두에서 강력한 솔루션을 찾는 사람들에게 효율적인 대안을 제공합니다.  
AI 응용 SW를 개발하는 5단계  
What? 내가 어떤 모델을 개발할 것인지  
DataSet을 준비한다. - 양질의 DataSet은 SW 품질을 좌우한다.  
적절한 Pre-Trainer된 AI 모델을 선택한다.  
오늘 공부할 내용  
Fire-Training : 2에서 준비한 dataSet을 이용해서 3을 Customaizing한다.  
마지막으로  
응용 SW를 제작한다. . webAPP - 이전에 실습했던 TM을 기억한다. . nativeAPP -python과 pyQT로 작성했던 기억.  
더블클릭 또는 Enter 키를 눌러 수정  

YOLO Clon from github  
작업 폴더를 준비하고 이동한다.  
[ ]
%cd /content
/content
모델 다운로드
[ ]
%pwd

github에서 복사해오기
!git clone https://github.com/ultralytics/yolov5

[ ]
!git clone https://github.com/ultralytics/yolov5
Cloning into 'yolov5'...
remote: Enumerating objects: 16582, done.
remote: Counting objects: 100% (60/60), done.
remote: Compressing objects: 100% (42/42), done.
remote: Total 16582 (delta 30), reused 39 (delta 18), pack-reused 16522
Receiving objects: 100% (16582/16582), 15.11 MiB | 23.24 MiB/s, done.
Resolving deltas: 100% (11383/11383), done.
[ ]
%cd yolov5
/content/yolov5
[ ]
# install dependencies as necessary
!pip install -qr requirements.txt  # install dependencies (ignore errors)
import torch

from IPython.display import Image, clear_output  # to display images

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 207.3/207.3 kB 2.0 MB/s eta 0:00:00
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.5/4.5 MB 23.3 MB/s eta 0:00:00
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 755.6/755.6 kB 45.7 MB/s eta 0:00:00
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 62.7/62.7 kB 8.5 MB/s eta 0:00:00
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
imageio 2.31.6 requires pillow<10.1.0,>=8.3.2, but you have pillow 10.3.0 which is incompatible.
Setup complete. Using torch 2.2.1+cu121 _CudaDeviceProperties(name='Tesla T4', major=7, minor=5, total_memory=15102MB, multi_processor_count=40)
데이터 셋 다운로드
폴더 생성 후 이동
[ ]
%mkdir hat
%cd hat
/content/yolov5/hat
[ ]
!curl -L "https://public.roboflow.com/ds/Arw2aWqEp0?key=BgKXyrr7JL" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
스트리밍 출력 내용이 길어서 마지막 5000줄이 삭제되었습니다.
 extracting: train/labels/000605_jpg.rf.0c32e7a0bc3cb35efb487e563bd531d2.txt  
 extracting: train/labels/000343_jpg.rf.0c2f20c5673eb0132c8f6d63906c5f2d.txt  
 extracting: train/labels/004238_jpg.rf.0c3465deac398afe856c752cc9d98852.txt  
 extracting: train/labels/004029_jpg.rf.0c5c615d8dc20133e108e813eca4234a.txt  
 extracting: train/labels/003332_jpg.rf.ffb6b4eb5348096c8a79689feaba7689.txt  
 extracting: train/labels/004344_jpg.rf.ff6aa909ab613eafa922a8296c5e5c98.txt  
 extracting: train/labels/003973_jpg.rf.ff5e4e96869ba73feacd6e24ac5eac28.txt  
 extracting: train/labels/002421_jpg.rf.ff98e8a8b8121dc1bf6600e5b89420da.txt  
 extracting: train/labels/002005_jpg.rf.ffabe992edd18facb28e74aca9253c05.txt  
 extracting: train/labels/002468_jpg.rf.ff839ee9e2323cecb932865b51188da6.txt  
 extracting: train/labels/000874_jpg.rf.fae34e320237dacc785391997f808126.txt  
 extracting: README.roboflow.txt     
 extracting: README.dataset.txt      
원본 데이터셋에 구조를 알아보기
[ ]
%cat /content/yolov5/hat/data.yaml
train: ../train/images
val: ../valid/images

nc: 3
names: ['head', 'helmet', 'person']
[ ]
from glob import glob
train_img_list = glob('/content/yolov5/hat/train/images/*.jpg')
test_img_list = glob('/content/yolov5/hat/valid/images/*.jpg')
valid_img_list = glob('/content/yolov5/hat/valid/images/*.jpg')
print(len(train_img_list), len(test_img_list), len(valid_img_list))
5269 0 0
리스트를 텍스트 파일로 저장하기
[ ]
import yaml
with open('/content/yolov5/hat/train.txt','w') as f:
    f.write('\n'.join(train_img_list) + '\n')
with open('/content/yolov5/hat/test.txt','w') as f:
    f.write('\n'.join(test_img_list) + '\n')
with open('/content/yolov5/hat/val.txt','w') as f:
    f.write('\n'.join(valid_img_list) + '\n')
함수 선언하기
[ ]
from IPython.core.magic import register_line_cell_magic

@register_line_cell_magic
def writetemplate(line,cell):
    with open(line, 'w') as f:
        f.write(cell.format(**globals()))
[ ]
%%writetemplate /content/yolov5/hat/data.yaml

train: /content/yolov5/hat/train/images
val: /content/yolov5/hat/valid/images

nc: 3
names: ['head', 'helmet', 'person']
모델 구성
[ ]
import yaml

with open('/content/yolov5/hat/data.yaml','r') as stream:
  num_classes = str(yaml.safe_load(stream)['nc'])

%cat /content/yolov5/models/yolov5s.yaml
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license

# Parameters
nc: 80 # number of classes
depth_multiple: 0.33 # model depth multiple
width_multiple: 0.50 # layer channel multiple
anchors:
  - [10, 13, 16, 30, 33, 23] # P3/8
  - [30, 61, 62, 45, 59, 119] # P4/16
  - [116, 90, 156, 198, 373, 326] # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [
    [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
    [-1, 3, C3, [128]],
    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
    [-1, 6, C3, [256]],
    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
    [-1, 9, C3, [512]],
    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
    [-1, 3, C3, [1024]],
    [-1, 1, SPPF, [1024, 5]], # 9
  ]

# YOLOv5 v6.0 head
head: [
    [-1, 1, Conv, [512, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 6], 1, Concat, [1]], # cat backbone P4
    [-1, 3, C3, [512, False]], # 13

    [-1, 1, Conv, [256, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 4], 1, Concat, [1]], # cat backbone P3
    [-1, 3, C3, [256, False]], # 17 (P3/8-small)

    [-1, 1, Conv, [256, 3, 2]],
    [[-1, 14], 1, Concat, [1]], # cat head P4
    [-1, 3, C3, [512, False]], # 20 (P4/16-medium)

    [-1, 1, Conv, [512, 3, 2]],
    [[-1, 10], 1, Concat, [1]], # cat head P5
    [-1, 3, C3, [1024, False]], # 23 (P5/32-large)

    [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
  ]
[ ]
%%writetemplate /content/yolov5/models/custom_yolov5s.yaml

# Parameters
nc: {num_classes}  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
[ ]
%cat /content/yolov5/models/custom_yolov5s.yaml

# Parameters
nc: 3  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 6, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 3, C3, [1024]],
   [-1, 1, SPPF, [1024, 5]],  # 9
  ]

# YOLOv5 v6.0 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
데이터셋 준비 완료
학습 시작
img: 입력 이미지 크기 정의

batch: 배치 크기 결정

epochs: 학습 기간 개수 정의

data: yaml 파일 경로

cfg: 모델 구성 지정

weights: 가중치에 대한 경로 지정

name: 결과 이름

nosave: 최종 체크포인트만 저장

cache: 빠른 학습을 위한 이미지 캐시

[ ]
%%time
%cd /content/yolov5/
!python train.py --img 416 --batch 16 --epochs 50 --data ./hat/data.yaml --cfg ./models/custom_yolov5s.yaml --weights '' --name hat_result --cache
/content/yolov5
2024-05-07 00:45:35.790347: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-05-07 00:45:35.790398: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-05-07 00:45:35.791817: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
train: weights=, cfg=./models/custom_yolov5s.yaml, data=./hat/data.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=50, batch_size=16, imgsz=416, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, evolve_population=data/hyps, resume_evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=hat_result, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest, ndjson_console=False, ndjson_file=False
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 🚀 v7.0-307-g920c721e Python-3.10.12 torch-2.2.1+cu121 CUDA:0 (Tesla T4, 15102MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/

Dataset not found ⚠️, missing paths ['/content/yolov5/hat/valid/images']
Traceback (most recent call last):
  File "/content/yolov5/train.py", line 848, in <module>
    main(opt)
  File "/content/yolov5/train.py", line 623, in main
    train(opt.hyp, opt, device, callbacks)
  File "/content/yolov5/train.py", line 176, in train
    data_dict = data_dict or check_dataset(data)  # check if None
  File "/content/yolov5/utils/general.py", line 561, in check_dataset
    raise Exception("Dataset not found ❌")
Exception: Dataset not found ❌
CPU times: user 75.8 ms, sys: 4.66 ms, total: 80.5 ms
Wall time: 9.94 s
