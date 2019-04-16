#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 10:27:03 2019

@author: venkat-pt2636
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    url('all/', views.home),
    url('request/',views.chatbot),
    url('predict/',views.predict)
]
