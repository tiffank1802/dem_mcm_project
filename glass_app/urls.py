from django.urls import path
from . import views

app_name = 'glass_app'

urlpatterns = [
    path('cards/', views.card_glass, name='card_glass'),
]
