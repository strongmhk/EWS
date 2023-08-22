from django.shortcuts import render
from .models import FileUpload
from django.http import HttpResponse, Http404, HttpResponseRedirect
from django.core.exceptions import BadRequest
from django.core import serializers
from django.shortcuts import render, get_object_or_404
import os
import pandas as pd
from .techList import *
import json
from time import sleep
# Create your views here.

def fileUpload(request):
  '''
  ALLOW METHOD: POST, GET
  URL: /files/
  1. POST: title, content, file을 입력받고 데이터베이스에 저장
  2. GET: 현재 DB에 저장된 파일들의 목록을 반환
  '''
  ## 파일 업로드
  if(request.method == 'POST'):
    title = request.POST['title']
    content = request.POST['content']
    img = request.FILES['imgfile']
    fileupload = FileUpload(
      title=title,
      content=content,
      imgfile=img,
    )
    fileupload.save()
    print(fileupload)
    return HttpResponse() # TODO 응답을 성공 페이지로 보내줘야함
  
  ## 전체 파일 정보 받기, GET /files/
  elif(request.method == 'GET'):
    AllFiles = FileUpload.objects.all()
    data = serializers.serialize("json", AllFiles)
    return HttpResponse(
      data,
      headers={
        "Content-Type": "applictaion/json"
      }
      )
    
  else :
    raise BadRequest('Invalid request')
  
def fileDetail(request, file_id):
  '''
  ALLOW METHOD: GET, DELETE
  URL: /files/{id}/
  1. DELETE: 파일을 삭제함
  2. GET: 파일의 5개의 데이터를 반환함
  '''
  if(request.method == 'DELETE') :
    file = get_object_or_404(FileUpload, pk=file_id)
    # 경로에서 파일 삭제
    try :
      path = getMediaURI()
      os.remove(os.path.join(path, file.imgfile.name))
    except:
      print('파일을 찾을 수 없습니다.')
  
    # db에서 파일 삭제
    file.delete()
    
    return HttpResponse("success")
  
  elif(request.method == 'GET'):
    jsonData = readCsvById_json(file_id)
    
    return HttpResponse(
      jsonData,
      headers={
        "Content-Type": "applictaion/json"
      }
      )

  
  else :
    return BadRequest('Invalid request')
  
def fileMetaData(request, file_id):
  '''
  ALLOW METHOD: GET
  URL: /files/{id}/metadata/
  1. GET: file의 column 정보를 받고 반환
  '''
  if(request.method == 'GET') :
    jsonData = readMetaData(file_id)
    
    return HttpResponse(
      jsonData,
      headers={
        "Content-Type": "applictaion/json"
      }
      )
  else:
    return BadRequest('Invalid request')
  


## TODO 데이터 분석을 하는 함수
def analyze(request, file_id) :
  '''
  ALLOW METHOD: POST
  URL: /files/{id}/analyze/
  1. POST: 분석을 진행할 column(feature)과 반응변수(target)을 입력받고 데이터 분석을 진행,
    분석된 대시보드의 URL을 json으로 반환
  '''
  if(request.method == 'POST'):
    targets = request.POST['targets']
    features = request.POST['features']
    
    # response = foo(df, targets, features)
    
    print(features)
    print(targets)
    
    sleep(20)
    
    ## 데이터 분석 시작
    
    ## 데이터 분석 끝
    return HttpResponse()
  
  else:
    return BadRequest('Invalid request')
  


#### ####
def getMediaURI() :
  # 현재 스크립트 파일의 경로 (절대 경로)
  current_path = os.path.abspath(__file__)
  # 현재 스크립트 파일이 위치한 디렉토리 (상위 경로)
  current_directory = os.path.dirname(current_path)
  # 현재 디렉토리의 상위 경로를 구하려면 다시 dirname을 사용합니다.
  parent_directory = os.path.dirname(current_directory)
  # 상위 경로의 다른 폴더 경로를 구하기 위해 os.path.join을 사용합니다.
  other_directory = os.path.join(parent_directory, 'media')
  return other_directory


def readCsvById_json(file_id) :
  '''
  csv파일을 읽고 5행의 데이터를 json으로 파싱해서 반환
  '''
  file = get_object_or_404(FileUpload, pk=file_id)
  path = os.path.join(getMediaURI(), file.imgfile.name)
  df = pd.read_csv(path)
  return df.head().to_json()

def readMetaData(file_id) :
  '''
  csv파일의 column 종류와 분석 가능한 기법들을 반환
  {
    "column" : [컬럼 데이터 리스트],
    "tect" : [분석 기법 이름 리스트]
  }
  '''
  file = get_object_or_404(FileUpload, pk=file_id)
  path = os.path.join(getMediaURI(), file.imgfile.name)
  df = pd.read_csv(path)
  column = df.columns.tolist()
  tech =[item.value for item in list(AnalysisTech_forClient)]
  
  data = {
    "column" : column,
    "tech": tech
  }
  
  return json.dumps(data)

def readCsvById(file_id):
  file = get_object_or_404(FileUpload, pk=file_id)
  path = os.path.join(getMediaURI(), file.imgfile.name)
  df = pd.read_csv(path)
  
  return df
