from django.urls import path

from . import views

app_name = "generation"

urlpatterns = [
    path("index/", views.index, name="index"),
    path("generate/", views.generate, name="generate"),
]
