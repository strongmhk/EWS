from django.db import models
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
# Create your models here.

# rawdata entity
class RawData(models.Model):
  file_name = models.ImageField(null=True, upload_to="", blank=True)
  describe = models.TextField(max_length=40, null=True)
  created_at = models.DateTimeField(auto_now_add=True)

# output entity
class Output(models.Model):
  created_at = models.DateTimeField(auto_now_add=True)
  
