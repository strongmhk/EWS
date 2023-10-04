from datetime import datetime
from pathlib import Path

from django.core.files.storage import default_storage
from django.core.serializers.json import DjangoJSONEncoder
from django.db import IntegrityError
from django.shortcuts import render
from .models import RawData, Output
from django.http import HttpResponse, Http404, HttpResponseRedirect, JsonResponse, FileResponse
from django.core.exceptions import BadRequest
from django.core import serializers
from django.shortcuts import render, get_object_or_404
import os
import pandas as pd
from .techList import *
import json
from time import sleep
# Create your views here.


'''
RawData 관련 api
'''

def fileUpload(request):
  '''
  ALLOW METHOD: POST, GET
  URL: /files/
  1. POST: describe, file_name 을 입력받고 데이터베이스에 저장
  2. GET: 현재 DB에 저장된 파일들의 목록을 반환
  '''
  ## 파일 업로드
  if(request.method == 'POST'):
    describe = request.POST['describe']
    file_name = request.FILES['file_name']

    response_data = insertRawDataToDB(file_name, describe)

    return JsonResponse(response_data)


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



def fileDetail(request, raw_data_id):
  '''
  ALLOW METHOD: GET, DELETE
  URL: /files/{raw_data_id}/
  1. DELETE: 파일을 삭제함
  2. GET: 파일의 데이터 5행까지 반환
  '''
  file = get_object_or_404(RawData, pk=raw_data_id)
  if(request.method == 'DELETE') :
    # 경로에서 파일 삭제
    try :
      os.remove(getDataPath(raw_data_id))
    except:
      HttpResponse("파일을 찾을 수 없습니다.")
  
    # db에서 파일 삭제
    file.delete()
    
    return HttpResponse("파일 삭제 완료")
  
  elif(request.method == 'GET'):
    json_data = readCsvById_json(raw_data_id)


    return HttpResponse(
      json_data,
      headers={
        "Content-Type": "application/json"
      }
    )

  
  else :
    return BadRequest('Invalid request')


def fileMetaData(request, raw_data_id):
  '''
  ALLOW METHOD: GET
  URL: /files/{raw_data_id}/metadata/
  1. GET: file의 column 정보를 받고 반환
  '''
  if(request.method == 'GET') :
    jsonData = readMetaData(raw_data_id)

    return HttpResponse(
      jsonData,
      headers={
        "Content-Type": "application/json"
      }
      )
  else:
    return BadRequest('Invalid request')
  




'''
여기부터 Output 관련 api
'''

def createDqReport(request, raw_data_id):
  '''
  ALLOW METHOD: POST
  URL: /files/dq-report/{raw_data_id}
  1. POST: raw_data_id를 받아 해당 파일의 path를 찾고, html 파일 생성
  '''

  if (request.method == 'POST'):
    # 파일 경로를 받아와 ews 인스턴스 초기화
    raw_data_path = getDataPath(raw_data_id)
    # ews = EWS(raw_data_path)
    # ews.setup(feature_list, target_col, date_col,  object_col, object_list, missing_dic, using_col)


    # 해당 파일 db에 저장
    dir = "dq_report/min"
    file_name = str(raw_data_id) + "_dq_report.html"
    # dir, file_name 넘겨주기
    dq_report_path = createOutputPath(dir, file_name)
    # 파일 해당 path안에 생성
    createFileInDirectory(dq_report_path)
    response_data = insertOutputToDB(raw_data_id, dq_report_path)

    return JsonResponse(response_data)







def outputCreate(request, raw_data_id):
  '''
  ALLOW METHOD: POST
  URL: /files/analyze/{raw_data_id}
  1. POST: raw_data_id를 전달받아 Output 데이터베이스에 저장
  '''
  if (request.method == 'POST'):
    dir = 'analyze/min'
    file_name = str(raw_data_id) + "_analyze.html"
    analyze_path = createOutputPath(dir, file_name) # output 파일 경로 생성
    createFileInDirectory(analyze_path)
    response_data = insertOutputToDB(raw_data_id, analyze_path)

    return JsonResponse(response_data)




def getAllOutput(request):
  '''
  ALLOW METHOD: GET
  URL: /files/ouputs/
  1. GET: 모든 output 파일 조회
  '''
  if (request.method == 'GET'):

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




def outputGetOrDelete(request, output_id):
  '''
  ALLOW METHOD: GET, DELETE
  URL: /files/outputs/{output_id}/
  1. DELETE: 파일을 삭제함
  2. GET: 해당 html 파일 응답 페이지로 반환
  '''

  if (request.method == 'DELETE'):
    output = get_object_or_404(Output, pk=output_id)
    # 경로에서 파일 삭제
    try:
      path = os.path.join(getMediaURI(), output.path)
      os.remove(path)
    except:
      print('파일을 찾을 수 없습니다.')

    # db에서 파일 삭제
    output.delete()

    return HttpResponse("파일 삭제 완료")

  elif (request.method == 'GET'):
    output = get_object_or_404(Output, pk=output_id)

    if default_storage.exists(output.path):
      return FileResponse(default_storage.open(output.path, 'rb'), content_type='text/html')
    else:
      raise Http404("존재하지 않는 파일입니다.")




'''
여기까지 api
'''





# RawData 테이블에 insert
def insertRawDataToDB(file_name, describe):
  '''
  form-data로 넘어온
  file_name, describe를 받아 Output 테이블에 해당 정보 insert
  '''
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





# output 테이블에 insert
def insertOutputToDB(raw_data_id, path):
  '''
  raw_data_id와 path를 받아 Output 테이블에 insert
  '''
  try:
    raw_data = RawData.objects.get(id=raw_data_id)
    # ews.setup(feature_list, target_col, date_col,  object_col, object_list, missing_dic, using_col)

    output = Output(
      raw_data_id=raw_data,
      path=path,
    )
    output.save()

    response_data = {
      "message": "결과 파일 생성 완료",
      "id": output.id,
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
  path = getDataPath(file_id)
  df = pd.read_csv(path)

  return df.head().to_json()





def readMetaData(file_id) :
  '''
  csv파일의 column 종류와 분석 가능한 기법들을 반환
  {
    "column" : [컬럼 데이터 리스트],
    "tech" : [분석 기법 이름 리스트]
  }
  '''
  path = getDataPath(file_id)
  df = pd.read_csv(path)
  column = df.columns.tolist()
  tech =[item.value for item in list(AnalysisTech_forClient)]
  
  data = {
    "column" : column,
    "tech": tech
  }
  
  return json.dumps(data)

def readCsvById(file_id):
  '''
  file_id를 받아 해당 csv파일의 데이터 프레임 반환
  '''
  path = getDataPath(file_id)
  df = pd.read_csv(path)
  
  return df


def getDataPath(raw_data_id):
  '''
  raw_data_id를 받아 RawData 테이블에서 해당 파일 조회,
  해당 파일의 file_name 필드 + media 파일 경로를 붙여 절대 경로 반환
  '''
  file = get_object_or_404(RawData, pk=raw_data_id)
  path = os.path.join(getMediaURI(), file.file_name.name)
  return path


def createOutputPath(dir, file_name):
  '''
  file의 path 생성
  dir : 저장하고자하는 디렉토리 이름
  file_name : 파일 이름
  ex) media/dir/년/월/일/file_name

  csv
  dq_report
  Preprocess_merged
  파일마다 디렉토리를 분리해 관리하기 위해
  '''

  # 오늘 날짜
  today = datetime.now()
  # 년/월/일로 포맷팅
  formatted_today = today.strftime("%Y/%m/%d")
  path = os.path.join(getMediaURI(), dir, formatted_today, file_name)

  # pathlib 모듈의 parent.mkdir 메서드 사용하기 위해서 인스턴스 생성
  path = Path(path)

  return path


def createFileInDirectory(output_path):
  '''
  path를 전달받아 해당 path에 파일 생성
  '''
  # 경로가 존재하는지 확인하고, 없으면 생성
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, 'w', encoding='utf-8') as f:
    # HTML 내용 작성
    f.write("<!DOCTYPE html>")
    f.write("<html>")
    f.write("<head>")
    f.write("<title>Title of the document</title>")
    f.write("</head>")
    f.write("<body>")
    f.write("Hello, World!")
    f.write("</body>")
    f.write("</html>")

  f.close()





def a(request):
  '''
  type: list, feature_list = []  # ex) ['계약건수[건]','시점'] 입력받은 df의 전체 col 리스트
  type: str,  target_col = "지급여력비율" target column
  type: str,  date_col = '시점'  date type을 가지고 있는 column
  type: str,  object_col = '회사별'
  type: list, object_list = []  # ex) ['메리츠','KB']
  type: dic,  missing_dic = {}  # ex) {'시점' : 'drop', '지급여력비율' : 'mean'}
  type: list,using_col = self.feature_list.copy() : feature_list 동일
  type: list,using_col.append(self.target_col)
  '''

  feature_list = request.POST['feature_list']
  target_col = request.POST['target_col']
  date_col = request.POST['date_col']
  object_col = request.POST['object_col']
  object_list = request.POST['object_list']
  using_col = request.POST['using_col']
  missing_dic = request.POST['missing_dic']
  raw_data_id = int(request.POST['raw_data_id'])