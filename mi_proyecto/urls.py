# mi_proyecto/urls.py
from django.contrib import admin
from django.urls import include, path
from mi_aplicacion import views  # Importa las vistas de tu aplicación

urlpatterns = [
    path('admin/', admin.site.urls),
    path('mi_aplicacion/', include('mi_aplicacion.urls')),  # Incluye las URLs de tu aplicación
    path('', views.index, name='index'), 
    path('app1/', views.app1, name='app1'),
    path('app2/', views.app2, name='app2'),  
]

