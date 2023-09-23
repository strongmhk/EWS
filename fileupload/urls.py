from django.urls import path

from . import views

urlpatterns = [
  path('', views.fileUpload, name="fileUpload"),
  path('<int:file_id>/', views.fileDetail, name="fileDetail"),
  path('<int:file_id>/metadata/', views.fileMetaData, name="fileMetaData"),
  path('<int:file_id>/analyze/', views.analyze, name="analyze"),
  path('outputs/', views.outputCreate, name="outputCreate"),
  path('outputs/<int:output_id>', views.outputGetOrDelete, name="outputGetOrDelete"),
]