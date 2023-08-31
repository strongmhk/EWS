# Generated by Django 4.2.4 on 2023-08-31 08:10

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='RawData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file_name', models.FileField(blank=True, null=True, upload_to='data/%Y/%m/%d')),
                ('describe', models.TextField(max_length=40, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Output',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file_name', models.FileField(blank=True, null=True, upload_to='data/%Y/%m/%d')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('analysis_tech', models.CharField(choices=[('LinearRegression', 'lr'), ('RidgeRegression', 'ridge'), ('ExtremeGradientBoosting', 'xgboost'), ('LightGBM', 'lightgbm')], default='lr', max_length=40)),
                ('raw_data_id', models.ForeignKey(db_column='raw_data_id', on_delete=django.db.models.deletion.CASCADE, related_name='analysis', to='fileupload.rawdata')),
            ],
        ),
    ]
