from django.db import IntegrityError
from django.shortcuts import render
from .models import RawData, Output
from django.http import HttpResponse, Http404, HttpResponseRedirect, JsonResponse
from django.core.exceptions import BadRequest
from django.core import serializers
from django.shortcuts import render, get_object_or_404
import os
import pandas as pd
from .techList import *
import json
from time import sleep
# Create your views here.


## RawData 관련 api

def fileUpload(request):
  '''
  ALLOW METHOD: POST, GET
  URL: /files/
  1. POST: describe, file_name 을 입력받고 데이터베이스에 저장
  2. GET: 현재 DB에 저장된 파일들의 목록을 반환
  '''
  ## 파일 업로드
  if request.method == 'POST':
    describe = request.POST['describe']
    file_name = request.FILES['file_name']

    response_data = createCsvFile(file_name, describe)

    return JsonResponse(response_data)   # TODO 응답을 성공 페이지로 보내줘야함


  ## 전체 파일 정보 받기, GET /files/
  elif(request.method == 'GET'):
    AllFiles = RawData.objects.all()
    data = serializers.serialize("json", AllFiles)
    return HttpResponse(
      data,
      headers={
        "Content-Type": "application/json"
      }
      )
    
  else :
    raise BadRequest('Invalid request')





def fileDetail(request, file_id):
  '''
  ALLOW METHOD: GET, DELETE
  URL: /files/{id}/
  1. DELETE: 파일을 삭제함
  2. GET: 파일명과 함께 파일의 5개의 데이터를 반환함
  '''
  if(request.method == 'DELETE') :
    file = get_object_or_404(RawData, pk=file_id)
    # 경로에서 파일 삭제
    try :
      path = getMediaURI()
      os.remove(os.path.join(path, file.file_name.name))
    except:
      print('파일을 찾을 수 없습니다.')
  
    # db에서 파일 삭제
    file.delete()
    
    return HttpResponse("파일 삭제 완료")
  
  elif(request.method == 'GET'):
    json_data = readCsvById_json(file_id)

    return HttpResponse(
      json_data,
      headers={
        "Content-Type": "application/json"
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
        "Content-Type": "application/json"
      }
      )
  else:
    return BadRequest('Invalid request')
  


## 여기부터 Output 관련 api

def outputCreate(request):
    '''
    ALLOW METHOD: POST, GET
    URL: /files/outputs/
    1. POST: raw_data_id, describe, file_name 을 입력받고 Output 데이터베이스에 저장
    2. GET: 현재 DB에 저장된 파일들의 목록을 반환
    '''
    if (request.method == 'POST'):

        raw_data_id = int(request.POST['raw_data_id'])
        describe = request.POST['describe']
        file_name = request.FILES['file_name']

        response_data = createOutputFile(raw_data_id, describe, file_name)

        return JsonResponse(response_data)


    ## 전체 파일 정보 받기, GET /files/ouputs/
    elif (request.method == 'GET'):

      AllFiles = Output.objects.all()
      data = serializers.serialize("json", AllFiles)

      return HttpResponse(
        data,
        headers={
          "Content-Type": "application/json"
        }
      )
    else:
      raise BadRequest('Invalid request')



'''
  ALLOW METHOD: GET, DELETE
  URL: /files/outputs/{id}/
  1. DELETE: 파일을 삭제함
  2. GET: 해당 파일 응답 페이지로 반환
  '''
def outputGetOrDelete(request, output_id):
  if (request.method == 'DELETE'):
    output = get_object_or_404(Output, pk=output_id)
    # 경로에서 파일 삭제
    try:
      path = getMediaURI()
      os.remove(os.path.join(path, output.file_name.name))
    except:
      print('파일을 찾을 수 없습니다.')

    # db에서 파일 삭제
    output.delete()

    return HttpResponse("파일 삭제 완료")

  elif (request.method == 'GET'):
    json_data = readCsvById_json(output_id)

    return HttpResponse(
      json_data,
      headers={
        "Content-Type": "application/json"
      }
    )



## TODO 데이터 분석을 하는 함수
def analyze(request, file_id) :
  '''
  ALLOW METHOD: POST
  URL: /files/{id}/analyze/
  1. POST: 분석을 진행할 column(feature)과 반응변수(target)을 입력받고 데이터 분석을 진행,
    분석된 대시보드의 URL을 json으로 반환
  2. GET: 분석 결과 파일
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
  
  elif(request.method == 'GET'):
    print("hello")
  else:
    return BadRequest('Invalid request')




# data파일 db에 insert
def createCsvFile(file_name, describe):
  try:
    rawdata = RawData(
      file_name=file_name,
      describe=describe,
    )
    rawdata.save()

    response_data = {
      "message": "파일 업로드 완료",
      "id": rawdata.id,
      "file_name": rawdata.file_name.url,
      "describe": rawdata.describe,
    }

  except IntegrityError:
    response_data = {
      "message": "이미 존재하는 파일 이름입니다."
    }

  return response_data





def createOutputFile(raw_data_id, file_name, describe):
  try:
    raw_data = RawData.objects.get(id=raw_data_id)
    output = Output(
      raw_data_id=raw_data,
      file_name=file_name,
      describe=describe,
    )
    output.save()

    response_data = {
      "message": "결과 파일 생성 완료",
      "id": output.id,
      "file_name": output.file_name.url,
      "describe": output.describe,
    }

  except IntegrityError:
    response_data = {
      "message": "이미 존재하는 파일 이름입니다."
    }

  return response_data




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
  file = get_object_or_404(RawData, pk=file_id)
  path = os.path.join(getMediaURI(), file.file_name.name)
  df = pd.read_csv(path)
  file_name = os.path.basename(path)

  data_as_dict = df.head().to_dict(orient="records")

  response_data = {
    "file_name": file_name,
    "data": data_as_dict
  }

  json_data = json.dumps(response_data)

  return df.head().to_json()



def readMetaData(file_id) :
  '''
  csv파일의 column 종류와 분석 가능한 기법들을 반환
  {
    "column" : [컬럼 데이터 리스트],
    "tech" : [분석 기법 이름 리스트]
  }
  '''
  file = get_object_or_404(RawData, pk=file_id)
  path = os.path.join(getMediaURI(), file.file_name.name)
  df = pd.read_csv(path)
  column = df.columns.tolist()
  tech =[item.value for item in list(AnalysisTech_forClient)]
  
  data = {
    "column" : column,
    "tech": tech
  }
  
  return json.dumps(data)

def readCsvById(file_id):
  file = get_object_or_404(RawData, pk=file_id)
  path = os.path.join(getMediaURI(), file.file_name.name)
  df = pd.read_csv(path)
  
  return df

