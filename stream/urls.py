from django.urls import path, include
from .views import Home, frameShow, VideoStream

urlpatterns = [
    path('', Home, name='home'),
    path('stream/', VideoStream, name='video-stream'),
    path('frameShow', frameShow, name='frame-show'),
]
