from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('app1/', views.app1, name='app1'),  # Ruta para app1
    path('app2/', views.app2, name='app2'),  # Ruta para app2
    path('app3/', views.app3, name='app3'),  # Ruta para app3
]
