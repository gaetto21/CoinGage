# Generated by Django 5.1.4 on 2025-01-03 03:25

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DataCollectionTask',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('start_date', models.DateTimeField()),
                ('end_date', models.DateTimeField()),
                ('interval', models.CharField(choices=[('1m', '1분'), ('5m', '5분'), ('10m', '10분'), ('30m', '30분'), ('1h', '1시간'), ('1d', '1일'), ('1w', '1주')], max_length=3)),
                ('status', models.CharField(choices=[('pending', '대기중'), ('running', '수집중'), ('completed', '완료'), ('failed', '실패')], default='pending', max_length=10)),
                ('selected_tickers', models.TextField()),
                ('output_file', models.CharField(max_length=255)),
                ('error_message', models.TextField(blank=True, null=True)),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='CollectedData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ticker', models.CharField(max_length=20)),
                ('datetime', models.DateTimeField()),
                ('open', models.FloatField()),
                ('high', models.FloatField()),
                ('low', models.FloatField()),
                ('close', models.FloatField()),
                ('volume', models.FloatField()),
                ('quote_volume', models.FloatField()),
                ('trade_count', models.IntegerField()),
                ('taker_buy_volume', models.FloatField()),
                ('taker_buy_quote_volume', models.FloatField()),
                ('task', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='myapp.datacollectiontask')),
            ],
            options={
                'unique_together': {('task', 'ticker', 'datetime')},
            },
        ),
    ]
