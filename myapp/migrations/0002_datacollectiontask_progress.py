# Generated by Django 5.1.4 on 2025-01-03 03:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='datacollectiontask',
            name='progress',
            field=models.FloatField(default=0),
        ),
    ]
