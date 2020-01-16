import kivy
kivy.require('1.9.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.camera import Camera
from kivy.uix.screenmanager import ScreenManager, Screen



from kivy.utils import platform
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2

Builder.load_string('''
<Browse_file@Button>:
    font_size: 32
    color: 0, 0, 0, 1
    size: 150, 150
<First_screen>:
    orientation: 'vertical'
    Browse_file:
        text: 'Browse'
        on_press: root.manager.browse_click()

    
''')

#root for this button points to first screen

class Manager(ScreenManager):
    def __init__(self, *widgets, **kargs):
        super(Manager, self).__init__(**kargs)
        for wi in widgets:
            self.add_widget(wi())

    def browse_click(self):
        print("im clicked")
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




    

class First_screen(Screen):
    def __init__(self):
        super(First_screen, self).__init__()



class Browse_file(Button):
    pass





class ImageRecognition(App):

    def __init__(self):
        super(ImageRecognition, self).__init__()

    def build(self):
        return Manager(First_screen)

    def on_stop(self):
        pass

if __name__ == "__main__":
    imagerecognition = ImageRecognition()
    imagerecognition.run()