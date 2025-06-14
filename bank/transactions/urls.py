from django.urls import path
from . import views

app_name = 'transactions'

urlpatterns = [
    path('',        views.landing_view,        name='landing'),
    path('add/',    views.add_transaction_view, name='add'),
    path('random/', views.add_random_view,      name='random'),
]
