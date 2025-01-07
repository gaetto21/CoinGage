from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('model/data-collection/', views.data_collection, name='data_collection'),
    path('model/technical-indicators/', views.technical_indicators_view, name='technical_indicators'),
    path('model/generate-indicators/', views.technical_indicators, name='generate_indicators'),
    path('api/tickers/', views.get_tickers, name='get_tickers'),
    path('model/data-preprocessing/', views.data_preprocessing_view, name='data_preprocessing'),
    path('model/data-combination/', views.data_combination_view, name='data_combination'),
    path('model/development/', views.model_development_view, name='model_development'),
    path('api/model/create/', views.create_model, name='create_model'),
    path('api/model/train/', views.train_model, name='train_model'),
    path('api/data/info/', views.get_data_info, name='data_info'),
] 