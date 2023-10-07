from django.db import models
from .techList import AnalysisTech_forServer

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
# Create your models here.

# rawdata entity
class RawData(models.Model):
  file_name = models.FileField(null=True, upload_to="data/%Y/%m/%d", unique = True)
  describe = models.TextField(max_length=40, null=True)

# output entity
class Output(models.Model):
  raw_data_id = models.ForeignKey("RawData", related_name="analysis", on_delete=models.SET_NULL, db_column="raw_data_id", null=True)
  path = models.FileField(null=True, upload_to="analyze/%Y/%m/%d", unique = True)




