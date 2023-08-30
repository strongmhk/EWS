from django.db import models
from .techList import AnalysisTech_forServer

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
# Create your models here.

# rawdata entity
class RawData(models.Model):
  file_name = models.FileField(null=True, upload_to="", blank=True)
  describe = models.TextField(max_length=40, null=True)
  created_at = models.DateTimeField(auto_now_add=True)

# output entity
class Output(models.Model):
  raw_data_id = models.ForeignKey("RawData", related_name="analysis", on_delete=models.CASCADE, db_column="raw_data_id")
  file_name = models.FileField(null=True, upload_to="", blank=True)
  created_at = models.DateTimeField(auto_now_add=True)
  analysis_tech = models.CharField(
    max_length=40,
    choices=[(tech.name, tech.value) for tech in AnalysisTech_forServer],
    default=AnalysisTech_forServer.LinearRegression.value,
  )

