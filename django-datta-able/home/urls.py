from django.urls import path
from django.contrib.auth import views as auth_views

from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
  path(''       , views.index,  name='index'),
  path('conversion/', views.conversion, name='conversion'),
  path('upload/', views.model_form_upload, name='model_form_upload'),
]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
