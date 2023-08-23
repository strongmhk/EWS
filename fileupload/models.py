from django.db import models
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
# Create your models here.

# rawdata entity
class RawData(models.Model):
  host = models.TextField(max_length=10, null=False)
  title = models.TextField(max_length=40, null=True)
  imgfile = models.ImageField(null=True, upload_to="", blank=True)
  content = models.TextField()

# output entity
class Output(models.Model):
  

