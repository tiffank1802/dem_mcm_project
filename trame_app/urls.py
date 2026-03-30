from django.urls import path
from . import views

urlpatterns = [
    path("trame-viz/", views.trame_viz, name="trame_viz"),
]
