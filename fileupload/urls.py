from django.urls import path

from . import views

urlpatterns = [
  path('', views.fileUpload, name="fileUpload"),
  path('<int:raw_data_id>/', views.fileDetail, name="fileDetail"),
  path('metadata/<int:raw_data_id>/', views.fileMetaData, name="fileMetaData"),
  path('analyze/<int:raw_data_id>/', views.outputCreate, name="outputCreate"),
  path('outputs/<int:output_id>/', views.outputGetOrDelete, name="outputGetOrDelete"),
  path('outputs/', views.getAllOutput, name="getAllOutput"),
  path('dq-report/<int:raw_data_id>/', views.createDqReport, name="createDqReport"),
  path('test-analyze/<int:raw_data_id>/', views.tempOutputCreate, name="tempOutputCreate"),
]