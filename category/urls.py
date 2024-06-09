from django.urls import path
from category import views

urlpatterns = [
    path('', views.CategoryAPIView.as_view(), name='category')
]
