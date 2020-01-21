from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.widget import Widget
from kivy.properties import StringProperty, ListProperty, NumericProperty, ReferenceListProperty, ObjectProperty, BooleanProperty
from kivy.uix.stacklayout import StackLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
import cv2
import numpy as np
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import time
from kivy.uix.scatter import Scatter
from kivy.garden.graph import Graph, MeshLinePlot

import matplotlib
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
import matplotlib.pyplot as plt
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas
from random import gauss
import scipy.stats as ss

from kivy.graphics import Color, Rectangle, Ellipse

from kivy.uix.filechooser import FileChooserListView
from kivy.core.window import Window

import os
from kivy.uix.popup import Popup

kv_text = """\
#:import WipeTransition kivy.uix.screenmanager.WipeTransition
<LoadDialog>:
	BoxLayout:
		size: root.size
		pos: root.pos
		orientation: "vertical"
		FileChooserListView:
			id: filechooser
			path: '.'

		BoxLayout:
			size_hint_y: None
			height: 30
			Button:
				text: "Cancel"
				on_release: root.cancel()

			Button:
				text: "Load"
				on_release: root.load(filechooser.path, filechooser.selection)

<MyBall@Widget>
	size: [50, 50]
	size_hint: None, None
	canvas:
		Color:
			rgb:0.8,0.8,1
		Rectangle:
			pos:self.parent.pos
			size: self.parent.size
		Color:
			rgba: 0, 0, 0, 1
		Ellipse:
			pos: self.pos
			size: self.size


<MainScreen>:
	transition: WipeTransition()
	id: sm
	FirstScreen:
	SecondScreen:

<FirstScreen>:
	name: "first_screen"
	GridLayout:
		orientation:'vertical'
		id: sterowanie_serv
		rows: 5
		padding: 10
		spacing: 10
		Label:
			id:current_status
			text: root.label_value
			size_hint_y: 0.1

		GridLayout:
			orientation: 'horizontal'
			cols:2
			size_hint_y: 0.2
			Button:
				id:browse_live
				text: root.text_value
				on_press: root.switch()

			ToggleButton:
				id:rec_save
				text: root.togglebutton_value
				on_press: root.toggle_switch(self)

		GridLayout:
			id: camera_data
			cols:2
			XinCam:
				id: XinCamera1
				resolution: (320, 240)
				play: True
				size:self.parent.size
			XinCam:
				id: XinCamera2
				resolution: (320, 240)
				play: True
				size: self.parent.size      
			
		BlankCanvas:
			id: modify
			size: self.parent.size
			MyBall
				id: selfball








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
	done_recording = BooleanProperty(False)
	def __init__(self, **kargs):
		super(XinCam, self).__init__(**kargs)
		self.frame = None
		self.fps = 120
		self.brightness = 1
		self.contrast = 0
		self.out = None
		

	def start(self, capture): #setting the capture need to be called before running on_tex
		self.frame_counter = 0
		self.capture = capture
		self.event = Clock.schedule_interval(self.update, 1.0 / self.fps) 
	def stop(self):
		Clock.unschedule(self.event)

	def reset_video(self):
		self.frame_counter = 0
		self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

	def complete_recording(self):
		self.done_recording=True
		self.out.release()
		self.out = None


	def update(self, dt):
		return_value, frame = self.capture.read()

		if return_value:
			self.process_frame(frame)
			
		if self.parent.parent.parent.text_value != "Browse":
			if self.frame_counter == self.capture.get(cv2.CAP_PROP_FRAME_COUNT): #recorded video terminated
				if self.out:
					self.complete_recording()
				self.reset_video() #repeat video if its not streaming
			#im currently not streaming


		self.canvas.ask_update()
		self.frame_counter += 1

	def process_frame(self, frame): #this shouuld process the frame and modify self.texture
		#print(frame)
		frame = cv2.addWeighted(frame, self.contrast, frame, self.brightness, 0) #alpha is contrast, beta is brightnes

		if self.out:
			self.out.write(frame)

		self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten("C")
		h, w = frame.shape[0], frame.shape[1]
		self.texture = Texture.create(size=(w, h))
		self.texture.flip_vertical()
		self.texture.blit_buffer(frame.tostring(), colorfmt='bgr')

	def save(self):
		self.reset_video()
		return_value, frame = self.capture.read()
		self.reset_video()
		print("saving frame shape is ", frame.shape)
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		self.out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame.shape[1],frame.shape[0]))




class MainScreen(ScreenManager):
	def __init__(self, cap):
		super(MainScreen, self).__init__()
		self.cam1 = self.get_screen("first_screen").ids["XinCamera1"]
		self.cam1.start(cap)
		self.cam2 = self.get_screen("first_screen").ids["XinCamera2"]
		self.cam2.start(cap)
		self.graph_handle()


		ig_object = self.get_screen("first_screen").ids["selfball"]
		ig_object.bind(bright_contrast=self.ig_callback)

	def graph_handle(self):
		self.graph1 = HistoHandle(self.cam1,self.cam2)
		self.get_screen("first_screen").ids.sterowanie_serv.add_widget(self.graph1.canv)
		self.graph1.start()

	def ig_callback(self, instance, value):
		print(f"graph changed brightness:{value[0]}, contrast:{value[1]}")
		self.cam2.brightness = value[0]
		self.cam2.contrast = value[1]


class MyBall(Widget): #widget should always be part of a layout, i think the parent is the boxlayout
	bright = ObjectProperty(0)
	contrast = ObjectProperty(0)
	bright_contrast = ReferenceListProperty(bright, contrast)
	scale_b = 2
	scale_c =2
	def __init__(self, **kargs):
		super(MyBall, self).__init__(**kargs)
		Window.bind(on_resize=self.on_window_resize)

		self.event = Clock.schedule_interval(self.adjust_line_space, 1 / 60.) #when encounter Nonetype error you need to schedule it and unschedule it later



	def on_touch_move(self, touch, down=False):
		if (self.findPoint(self.parent.pos[0], self.parent.pos[1], self.parent.pos[0]+self.parent.size[0], self.parent.pos[1]+self.parent.size[1], touch.x, touch.y)):
			self.center_x, self.center_y = touch.pos #because the pos is adjusted base on parents in kv thus this need to modify parents

			x_off = int(self.center_x - self.parent.pos[0])
			y_off = int(self.center_y - self.parent.pos[1])



			#print(type(self.find_index(x_off, self.x_bright_linespace)))

			self.bright = float(self.x_bright_linespace[x_off])  #x is brightness
			self.contrast = float(self.y_contrast_linespace[y_off]) #y is contrast

			if down:
				super(MyBall, self).on_touch_down(touch)

			else:
				super(MyBall, self).on_touch_move(touch)


	def adjust_line_space(self, *dt):
		if self.parent:

			self.pos = [self.parent.center_x - self.size[0]/2, self.parent.center_y - self.size[1]/2]

			b = np.arange(0, self.parent.size[0]+1, 1)
			self.x_bright_linespace = np.interp(b, (b.min(), b.max()), (0, self.scale_b))
			c = np.arange(0, self.parent.size[1]+1, 1)
			self.y_contrast_linespace = np.interp(c, (c.min(), c.max()), (0, self.scale_c))
			Clock.unschedule(self.event)

			#assert len(b[0]) == len(self.x_bright_linespace), f"Size doesnt match {len(b[0])} {len(self.x_bright_linespace)}"


	def on_touch_down(self, touch):
		self.on_touch_move(touch, down=True)


	def on_window_resize(self, window, width, height):
		self.adjust_line_space()
		




	def findPoint(self, x1, y1, x2, y2, x, y):
		if (x > x1 and x < x2 and y > y1 and y < y2):
			return True
		else: 
			return False

class LoadDialog(FloatLayout):
	load = ObjectProperty(None)
	cancel = ObjectProperty(None)


class BlankCanvas(BoxLayout):
	pass


'''
class MyInteract(FigureCanvas):
	fig, ax = plt.subplots(figsize=(100,100))
	canv = fig.canvas
	def __init__(self):
		super(MyInteract, self).__init__(self.fig)
		self.initialize_plot()
		self.points_x = [0, 0.5 ,100] #contains 3 points for x
		self.points_y = [0, 0.5 ,100] #contains 3 points for y
		x_points, y_points = self.find_parab()
		self.ax.plot(x_points, y_points, linestyle='-.')

	def initialize_plot(self):
		#self.ax.set_axisbelow(True)
		self.ax.grid(linestyle='-', linewidth='0.5', color='red')
		#self.ax.autoscale()
		self.ax.set_ylabel('Brightness')
		self.ax.set_xlabel('Contrast')


	def on_touch_move(self, touch):

		print(touch.x, touch.y)
		self.points_x[1] = touch.x
		self.points_y[1] = touch.y
		x_points, y_points = self.find_parab()
		self.ax.clear()
		self.initialize_plot()
		self.ax.plot(x_points, y_points, linestyle='-.')
		self.canv.draw_idle()



	def find_parab(self):
		#def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
		x1, x2, x3 = self.points_x
		y1, y2, y3 = self.points_y
		denom = (x1-x2) * (x1-x3) * (x2-x3)
		A = (x3 * (y2-y1) + x2 * (y1-y3) + x1 * (y3-y2)) / denom
		B = (x3*x3 * (y1-y2) + x2*x2 * (y3-y1) + x1*x1 * (y2-y3)) / denom
		C = (x2 * x3 * (x2-x3) * y1+x3 * x1 * (x3-x1) * y2+x1 * x2 * (x1-x2) * y3) / denom

		x_points = [i for i in np.arange(0, 1.1, 0.1)]
		y_points = [((A*(i**2)) + (B*i) + C) for i in x_points]

		return x_points, y_points	
'''


class HistoHandle(Widget):
	

	def __init__(self, cam1, cam2):
		super(HistoHandle, self).__init__()
		self.cam1=cam1
		self.cam2=cam2
		self.fig, (self.ax1, self.ax2) = plt.subplots(1,2)
		self.canv = self.fig.canvas


	def start(self):
		self.event = Clock.schedule_interval(self.update, 1.0 / 1)
	def stop(self):
		Clock.unschedule(self.event)
	def update(self, *dt):
		self.ax1.set_xlim(0, 255)
		self.ax2.set_xlim(0,255)
		self.ax1.hist(self.cam1.frame, bins="auto", histtype="step")
		self.ax2.hist(self.cam2.frame, bins="auto", histtype="step")
		self.canv.draw_idle()
		self.ax1.clear()
		self.ax2.clear()

	



class FirstScreen(Screen):
	text_value = StringProperty("Browse")
	label_value = StringProperty("Streaming from Camera... ")
	togglebutton_value = StringProperty("Click to Record")


	#file system

	loadfile = ObjectProperty(None)


	def switch(self):

		self.ids.XinCamera1.stop()
		self.ids.XinCamera2.stop()
		self.manager.graph1.stop()
		if self.text_value == "Browse": #if button is browse that means currently streaming
			self.show_load()
		else:
			self.label_value = "Streaming from Camera... "
			self.text_value = "Browse"
			self.ids.XinCamera1.start(cv2.VideoCapture(0))
			self.ids.XinCamera2.start(cv2.VideoCapture(0))
			self.manager.graph1.start()
	def recording_complete(self, instance, param):
		self.ids.rec_save.state= "normal"
		self.togglebutton_value = "Click to Record"
		self.ids.XinCamera2.done_recording= False
		popup = Popup(title="Video Saved complete",
				content=Label(text="SUCCESS"),
				size=(100, 100),
				size_hint=(0.3, 0.3),
				auto_dismiss=True)
		popup.open()




	def toggle_switch(self, tg_object):
		if tg_object.state.strip() == "down":
			self.togglebutton_value = "Recording... "
			self.ids.XinCamera1.reset_video()
			self.ids.XinCamera2.save()
			self.ids.XinCamera2.bind(done_recording=self.recording_complete)
		else:
			self.togglebutton_value = "Click to Record"
			self.ids.XinCamera2.complete_recording()



	def show_load(self):
		content = LoadDialog(load=self.load, cancel=self.cancel)
		self._popup = Popup(title="Load file", content=content,size_hint=(0.9, 0.9))
		self._popup.open()

	def load(self, path, filename):

		self.label_value = f"Video {os.path.basename(filename[0])} Loaded"
		self.text_value = "Stream"

		self.ids.XinCamera1.start(cv2.VideoCapture(filename[0]))
		self.ids.XinCamera2.start(cv2.VideoCapture(filename[0]))

		self.dismiss_popup()

	def cancel(self):
		self.ids.XinCamera1.start(self.ids.XinCamera1.capture)
		self.ids.XinCamera2.start(self.ids.XinCamera2.capture)
		self.dismiss_popup()

	def dismiss_popup(self):
		self._popup.dismiss()
		self.manager.graph1.start()


		
		

class SecondScreen(Screen):
	#some methods
	pass

class MyKivyApp(App):
	def __init__(self):
		super(MyKivyApp, self).__init__()
		self.capture = cv2.VideoCapture(0)

	def build(self):
		Builder.load_string(kv_text)
		return MainScreen(self.capture)

	def on_stop(self):
	   self.capture.release()

def main():
	app = MyKivyApp()
	app.run()

if __name__ == '__main__':
	main()