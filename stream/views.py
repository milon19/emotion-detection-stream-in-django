from django.http import StreamingHttpResponse
from django.shortcuts import render, HttpResponse
from imutils.video import VideoStream
import cv2

from stream.camera import openCamera
cam = openCamera()

def Home(request):

    return render(request, 'home.html', {})

def VideoStream(request):
    params = cam.get_params()
    context = {
        'params': params
    }
    return render(request, 'video-stream.html', context)

def gen(cam):
    while True:
        frame = cam.get_frame_web()
        yield b'--frame\r\n 'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'

def frameShow(request):
    return StreamingHttpResponse(gen(cam), content_type='multipart/x-mixed-replace; boundary=frame')