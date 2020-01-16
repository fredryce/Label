import kivy
kivy.require('1.9.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.camera import Camera
from kivy.uix.screenmanager import ScreenManager, Screen



from kivy.utils import platform
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2

Builder.load_string('''
<CaputeImage>:
    orientation: 'vertical'
    XinCam:
        id: XinCamera
        resolution: (640, 480)
        play: True

    ToggleButton:
        text: 'Play'
        on_press: XinCamera.play = not XinCamera.play
        size_hint_y: None
        height: '48dp'
''')


class XinCam(Image):
    def __init__(self, **kargs):
        super(XinCam, self).__init__(**kargs)

    def start(self, capture, fps=60): #setting the capture need to be called before running on_tex
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps) 

    def update(self, dt):
        return_value, frame = self.capture.read()
        if return_value: self.process_frame(frame)
        self.canvas.ask_update()

    def process_frame(self, frame): #this shouuld process the frame and modify self.texture
        #print(frame)
        h, w = frame.shape[0], frame.shape[1]
        self.texture = Texture.create(size=(w, h))
        self.texture.flip_vertical()
        self.texture.blit_buffer(frame.tobytes(), colorfmt='bgr')


    





class CaputeImage(BoxLayout):
    #wtf init method is called after build. this should only be used to store button functions. never put init method in this function. dk when is it called
    def __init__(self, **kargs):
        super(CaputeImage, self).__init__(**kargs)
        self._request_android_permissions()

    @staticmethod
    def is_android():
        return platform == 'android'

    def _request_android_permissions(self):
        """
        Requests CAMERA permission on Android.
        """
        if not self.is_android():
            return
        from android.permissions import request_permission, Permission
        request_permission(Permission.CAMERA)



class ImageRecognition(App):

    def __init__(self):
        super(ImageRecognition, self).__init__()
        self.capture = cv2.VideoCapture(0)

    def build(self):
        layout = CaputeImage()
        layout.ids["XinCamera"].start(self.capture)
        return layout

    def on_stop(self):
        self.capture.release()

if __name__ == "__main__":
    imagerecognition = ImageRecognition()
    imagerecognition.run()