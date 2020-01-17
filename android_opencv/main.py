from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
import cv2
from kivy.clock import Clock

kv_text = """\
#:import WipeTransition kivy.uix.screenmanager.WipeTransition

<MainScreen>:
	transition: WipeTransition()
	id: sm
	FirstScreen:
	SecondScreen:

<FirstScreen>:
	name: "first_screen"
	GridLayout:
		id: sterowanie_serv
		rows: 10
		padding: 10
		spacing: 10
		BoxLayout:
			spacing: 20
			XinCam:
        		id: XinCamera1
        		resolution: (640, 480)
        		play: True
			XinCam:
        		id: XinCamera2
        		resolution: (640, 480)
        		play: True

<SecondScreen>:
	name: "second_screen"
	GridLayout:
		id: Przemo
		rows: 5
		padding: 10
		spacing: 10
		BoxLayout:
			spacing: 20
			Button:
				text:'1'
				on_press: print('Button 1')
			Button:
				text:'x2'
				on_press: print('Button X2')
			Button:
				text:'go to First screen'
				on_press: app.root.current = 'first_screen'
"""


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




class MainScreen(ScreenManager):
	def __init__(self, cap):
		super(MainScreen, self).__init__()
		#print(self.get_screen("first_screen").ids)
		self.get_screen("first_screen").ids["XinCamera1"].start(cap)
		self.get_screen("first_screen").ids["XinCamera2"].start(cap)

class FirstScreen(Screen):
	#some methods
	pass

class SecondScreen(Screen):
	#some methods
	pass

class MyKivyApp(App):
	def __init__(self):
		super(MyKivyApp, self).__init__()
		self.capture = cv2.VideoCapture(0)

	def build(self):
		return MainScreen(self.capture)

	def on_stop(self):
	   self.capture.release()

def main():
	Builder.load_string(kv_text)
	app = MyKivyApp()
	app.run()

if __name__ == '__main__':
	main()