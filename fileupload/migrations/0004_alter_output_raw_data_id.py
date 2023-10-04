# Generated by Django 4.2.4 on 2023-10-04 05:26

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("fileupload", "0003_rename_describe_output_path_remove_output_file_name"),
    ]

    operations = [
        migrations.AlterField(
            model_name="output",
            name="raw_data_id",
            field=models.ForeignKey(
                db_column="raw_data_id",
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="analysis",
                to="fileupload.rawdata",
            ),
        ),
    ]