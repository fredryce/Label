import cv2
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import time
from concurrent.futures import ThreadPoolExecutor as Pool
import os, glob, sys
import threading
import multiprocessing
import itertools
import argparse
from skimage.measure import compare_ssim as ssim
#what can change the contour and the location of the object can change
#issues, error when clicking next button too fast many times, in find next mask function
#issue check on inserting ovals in the next frame



class Image_root(tk.Tk):
	def __init__(self):
		super(Image_root, self).__init__()

		self.size = 5
		self.padding = 0
		self.threads = []
		self.difference = 0
		self.tolerance_setting = {}

		self.label_path = "./label_image_output/labels"
		self.image_path = "./label_image_output/originals"
		parser = argparse.ArgumentParser()
		parser.add_argument('--option', help='choose label, or populate', required=False)
		parser = parser.parse_args()
		if parser.option== 'label' or parser.option == None:
			self.image_mod_class = Image_modify(direct = self.image_path)
		elif parser.option == 'populate':
			self.createDir()
			self.image_mod_class = Image_modify('skate2.mp4')
			self.image_mod_class.separate(self.image_path)
		elif parser.option == 'fill':
			self.image_mod_class = Image_modify(direct = self.label_path, fill=True)

		
		self.lk_params = dict(winSize = (30, 30), maxLevel = 4,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)) #default 15,15
		self.create_canvas()
		self.create_label()
		self.create_button_clear()
		self.create_position_label()
		self.create_button_next()
		self.create_button_next100()
		self.create_check_button()
		self.pool = Pool(4)
		self.bind('<Return>', lambda event: self.enter_pressed(event))
	def enter_pressed(self, event):
		self.update_image()
		#print("i pressed")

	def create_check_button(self):
		self.check_var = tk.IntVar()
		self.show_verify = tk.Checkbutton(self.label, variable=self.check_var, text = 'SHOW')
		self.show_verify.pack(anchor=tk.W, side='left')



	def createDir(self):
		if not os.path.isdir("./label_image_output"):
			os.mkdir("./label_image_output")
			os.mkdir(self.label_path)
			os.mkdir(self.image_path)


	def change_oval(self, oval, x, y):
		x1, y1, x2, y2 = self.canvas.coords(oval)
		x1, y1, x2, y2 = x - self.size, y -self.size, x+self.size, y+self.size

		if x1 < self.padding/2:
			x1 = self.padding/2
		if y1 < self.padding/2:
			y1 = self.padding/2
		if x2 > self.image_mod_class.width + self.padding/2:
			x2 = self.image_mod_class.width + self.padding/2
		if y2 > self.image_mod_class.height +self.padding/2:
			y2 = self.image_mod_class.height+self.padding/2


		self.canvas.coords(oval, x1,y1,x2,y2)

	def find_next_points(self,image1, image2, previous_mask): #causes the previous polygon not removed
		#points = itertools.chain(*[(self.canvas.coords(oval)[0] + self.size, self.canvas.coords(oval)[1] + self.size) for oval in self.oval_shapes])
		#print(list(points))

		gray_image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
		gray_image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
		gray = cv2.cvtColor(previous_mask, cv2.COLOR_RGB2GRAY)
		ret, thresh = cv2.threshold(gray, 127, 255,0)
		im2, cont, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		points = np.array([(self.canvas.coords(oval)[0] + self.size, self.canvas.coords(oval)[1] + self.size) for oval in self.oval_shapes]).astype(np.float32) #points needs to be float32

		
		new_points, status, error = cv2.calcOpticalFlowPyrLK(gray_image1, gray_image2, points, None, **self.lk_params)
		#print(new_points)
		for i in range(len(self.oval_shapes)):
			self.change_oval(self.oval_shapes[i], new_points[i][0], new_points[i][1])

		#threading.Thread(target=self.update_polygon).start()
		self.update_polygon()
		'''
		mask = Image.new('RGB', (self.image_mod_class.width,self.image_mod_class.height)) #create a new image for drawing
		#mask = Image.fromarray(random.astype('uint8')).convert('RGB')
		draw = ImageDraw.Draw(mask)
		draw.polygon([(self.canvas.coords(oval)[0] + self.size, self.canvas.coords(oval)[1] + self.size) for oval in self.oval_shapes], fill = 'white')
		mask = np.array(mask)
		mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
		return mask
		'''

	def _magicwand(self, image, tolerance, seed_point, connectivity = 4):
		self._flood_mask[:] = 0
		flags = connectivity | 255 << 8   # bit shift
		flags |= cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
		flood_image = image.copy()
		cv2.floodFill(flood_image, self._flood_mask, seed_point, 0, tolerance, tolerance, flags)
		return self._flood_mask[1:-1, 1:-1].copy()

	def find_tolerance(self, old_image, new_image, mask):
		completed_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
		#for every points in the mask, find the tolerance that will recreate the portion of mask from the image. use that data and apply to the next image
		#maybe: append the result of magicwand to the completed mask, if the next point is already on the completed mask, then move on.
		#multi processing



		





	def show(self, numpy_images, mask = False):
		if type(numpy_images) == str:
			numpy_images = cv2.imread(os.path.join(self.image_path, os.path.basename(numpy_images))).astype(np.uint8)
			if mask:
				read_mask = cv2.imread(self.image_name).astype(np.uint8)
				get_mask = cv2.medianBlur(read_mask,15)
				numpy_images[get_mask == 255] = 255
			cv2.imshow("previous label",numpy_images)



	def process_image(self):


		if len(self.oval_shapes) > 2:
			self.draw.polygon([(self.canvas.coords(oval)[0] + self.size, self.canvas.coords(oval)[1] + self.size) for oval in self.oval_shapes], fill = 'white')		
		mask_array = np.array(self.mask)
		cv2.imwrite(self.image_name, mask_array.astype(np.uint8))

		if self.check_var.get() == 1:
			self.show(self.image_name, mask=True)

		#the above code is for previous image, the below code is for preparing for next image

		image_np, image_name = self.image_mod_class.read_image(self.image_path, self.label_path)

		current_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
		previous_gray = cv2.cvtColor(self.image_np, cv2.COLOR_RGB2GRAY)

		difference = ssim(previous_gray, current_gray)
		#print(difference)
		if abs(difference - self.difference) > 0.05:
			pass
			#print("changing to a completely new frame")
		self.difference = difference

		if len(self.oval_shapes)>0:
			new_mask = self.find_next_points(self.image_np, image_np, mask_array)
			self.find_tolerance(self.image_np, image_np, mask_array)


		return image_np, image_name


		

	def create_canvas(self): #canvas being created first function 
		self.canvas = tk.Canvas(self, width=self.image_mod_class.width,height=self.image_mod_class.height, bg='black')

		self.canvas.bind("<Button-1>", self.canvas_location)
		self.canvas.bind("<Button-3>", lambda event: self.canvas_location(event, modi = True))

		self.canvas.bind("<Motion>", self.canvas_motion)
		self.canvas.bind('<Control-Button-1>')
		self.canvas.bind('<Control-Button-3>')


		self.canvas.pack(expand = tk.YES, fill=tk.BOTH)
	def create_label(self):
		self.label = tk.Label(self, bg='blue')
		self.label.pack(expand = tk.YES, fill = 'both')

	def create_position_label(self): 
		self.position_label = tk.Label(self.label) 
		self.position_label.pack(side = 'left', fill='x',expand = True)



	def create_button_next(self):
		self.button_next = tk.Button(self.label, bg='red', text='NEXT', command=self.update_image)
		self.button_next.pack(anchor=tk.E, side='right')


	def create_button_clear(self):
		self.button_next = tk.Button(self.label, bg='red', text='CLEAR', command=self.clear_dots)
		self.button_next.pack(anchor=tk.W, side='left')

	def create_button_next100(self):
		self.button_next = tk.Button(self.label, bg='red', text='NEXT100', command=self.update_image_100)
		self.button_next.pack(anchor=tk.E, side='right')

	def update_image_100(self):
		for i in range(100): #change this value to skip how many frames at once
			self.update_image()

		print(self.image_mod_class.frame_num)


	def clear_dots(self, single = False, oval = None):
		try:
			if single:
				self.canvas.unbind("<Button-3>")
				self.canvas.delete(oval)
				self.oval_shapes.remove(oval)
				self.update_polygon()
				
				
			else:
				try:
					self.pool.map(lambda row: self.canvas.delete(row), self.oval_shapes)
					self.canvas.delete(self.polygon)
					del self.polygon
				except Exception:
					print('im in exception')
				self.oval_shapes = []
				self.mode = None
		except AttributeError:
			self.update_image()


	def update_image(self):

		try:  #means the user is moving from one image to the next
			self.pointsClicked
			cv2.destroyAllWindows()
			if len(self.oval_shapes) > 0:
				self.mode = "modify"
			self.image_np, self.image_name = self.process_image()
			self._flood_mask = np.zeros((self.image_np.shape[0]+2, self.image_np.shape[1]+2), dtype=np.uint8)

		except AttributeError as e: #starting from the first image
			print('im in here')
			print(e)
			self.mode = None
			self.oval_shapes = []
			self.image_np, self.image_name = self.image_mod_class.read_image(self.image_path, self.label_path)

		

		self.title("{}/{}".format(self.image_mod_class.frame_num, self.image_mod_class.max_frame))

		self.pointsClicked = []
		
		#img = next(self.image_mod)

		#self.image_np, self.image_name = self.image_mod_class.read_image(self.image_path, self.label_path)



		self.img = Image.fromarray(self.image_np, 'RGB')

		self.mask = Image.new('RGB', (self.image_mod_class.width,self.image_mod_class.height)) #create a new image for drawing
		self.draw = ImageDraw.Draw(self.mask)


		self.new_photo= ImageTk.PhotoImage(self.img)
		try:
			self.canvas.itemconfig(self.canvas_image, image = self.new_photo)

		except AttributeError:

			self.canvas_image = self.canvas.create_image(self.padding/2, self.padding/2,image=self.new_photo,anchor=tk.NW)
			self.canvas.config(height= self.image_mod_class.height + self.padding, width= self.image_mod_class.width +self.padding)
			self.canvas.itemconfig(self.canvas_image, image = self.new_photo)

		
		#cv2.imshow('out', img)





	def canvas_location(self, event, modi = False): #click points on image modi means modify existing
		try:
			self.pointsClicked.append((event.x, event.y))
		except AttributeError:
			self.update_image()


		x1, y1 = (event.x - self.size), (event.y - self.size)
		x2, y2 = (event.x + self.size), (event.y + self.size)
		oval = self.canvas.create_oval(x1, y1, x2, y2, fill='red', tags='ovals')
		self.canvas.tag_bind(oval, '<Enter>', lambda event: self.hover_oval(oval))
		self.canvas.tag_bind(oval, '<Leave>', lambda event: self.hover_oval(oval, out=True))
		self.canvas.tag_bind(oval, '<Button-3>', lambda event: self.clear_dots(single= True, oval = oval))

		if modi and len(self.oval_shapes) > 2:
			#binary search
			self.find_closest(oval)
		else:
			if not self.mode:
				self.oval_shapes.append(oval)
			else:
				#insert in right place
				self.find_closest(oval, insert=True)
		self.update_polygon()

	def find_closest(self, oval, insert = False):
		mod_array = []
		min_distance =None
		final_point = None
		index = 0

		new_point = np.array([self.canvas.coords(oval)[0] + self.size, self.canvas.coords(oval)[1] + self.size])
		for i in range(len(self.oval_shapes)):
			
			point = np.array([self.canvas.coords(self.oval_shapes[i])[0] + self.size, self.canvas.coords(self.oval_shapes[i])[1] + self.size])

			if not min_distance:
				min_distance = np.linalg.norm(new_point- point)
			else:
				if np.linalg.norm(new_point - point) < min_distance:
					min_distance = np.linalg.norm(new_point - point)
					index = i
					final_point = point

				
			mod_array.append(self.oval_shapes[i])

		if not insert:	
			self.canvas.delete(self.oval_shapes[index])
			mod_array.remove(self.oval_shapes[index])
		else:
			pass
			'''
			try:
				#distance doesnt work in order to fit in right spot
				previous_point = np.array([self.canvas.coords(self.oval_shapes[index-1])[0] + self.size, self.canvas.coords(self.oval_shapes[index-1])[1] + self.size])
				next_point = np.array([self.canvas.coords(self.oval_shapes[index +1])[0] + self.size, self.canvas.coords(self.oval_shapes[index + 1])[1] + self.size])

				prev_distance = np.linalg.norm(new_point - previous_point)
				next_distance = np.linalg.norm(new_point - next_point)

				if next_distance < prev_distance:
					index = index +1

			except Exception as e:
				print('im not in bound')
			'''

		mod_array.insert(index, oval)


		
		self.oval_shapes = mod_array


	
	def update_polygon(self):
		if len(self.oval_shapes) >2:
			t = threading.Thread(target=self.update_thread)
			self.threads.append(t)
			t.start()
			
	def update_thread(self):
		try:
			self.canvas.delete(self.polygon)
		except Exception:
			print("deleting previous poly failed")
		pp = [(self.canvas.coords(oval)[0] + self.size, self.canvas.coords(oval)[1] + self.size) for oval in self.oval_shapes]
		self.polygon = self.canvas.create_polygon(pp, fill = '', outline='red')
		self.canvas.bind("<Button-3>", lambda event: self.canvas_location(event, modi = True))




	def hover_oval(self, oval, out=False):
		x1,y1,x2,y2 = self.canvas.coords(oval)
		if out:
			x1 +=4
			y1 +=4
			x2 -=4
			y2 -=4
		else:
			x1 -=4
			y1 -=4
			x2 +=4
			y2 +=4
		self.canvas.coords(oval, x1,y1,x2,y2)




	def canvas_motion(self, event, move=False, oval=None):
		value = 'X: {} Y:{}'.format(event.x, event.y)
		self.position_label.config(text=value)

	
class Image_modify():
	def __init__(self,file_name = None, direct = None, fill = False):
		self.file_name = file_name
		self.width = None
		self.height = None
		self.frame_num = 0
		self.mask = None
		if direct:
			self.max_frame = self.check_last(direct)
		self.fill = fill
		if fill:
			self.missing_num = self.find_gap(direct)
			print(self.missing_num)

	def read_image(self, img_dir, label_dir):
		if not self.fill:
			max_value = self.check_last(label_dir)
			
		else:
			max_value = self.missing_num.pop(0)




		if max_value != 0:
			self.frame_num = max_value
			print('Labeling Image {}'.format(max_value))
			image_name = os.path.join(img_dir, str(max_value) + '.jpg')
			image = cv2.imread(image_name)
			if not image.any():
				print("All Frames been labeled")
				exit()
			self.height, self.width = image.shape[:2]
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		return image, os.path.join(label_dir, str(max_value) + '.jpg')

	def find_gap(self, direct):
		print('max frame is', self.max_frame)
		miss_array = []
		file_array = [0] * self.max_frame
		for files in glob.iglob(os.path.join(direct, r'*.jpg')):
			base = int(os.path.basename(files).split('.')[0])
			file_array[base-1] = 1
		for i in range(len(file_array)):
			if file_array[i] != 1:
				miss_array.append(i+1)

		return miss_array

	def separate(self, direct):
		counter = self.check_last(direct) #keep track of image name
		start = 0 #keep track of number within a video
		if counter == 0:
			width = 1280
			height = 720
		else:
			previous_image = cv2.imread(os.path.join(direct, str(counter-1)+'.jpg'))
			width = previous_image.shape[0]
			height = previous_image.shape[1] 


		print('starting from image number {}'.format(counter))
		vidcap = cv2.VideoCapture(self.file_name)
		length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
		sucess = 1
		while sucess:
			
			percentValue = int((start/length)*100)
			sys.stdout.write('\r')
			sys.stdout.write("%s[%s%s] %i/%i\r" % ("processing: ", "#"*percentValue, "."*(100 - percentValue), percentValue, 100))
			sys.stdout.flush()

			sucess, frame = vidcap.read()
			#frame = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
			if start % 3 == 0:
				frame = cv2.resize(frame, (height, width)) #wtf cv2 takein height then width?
				cv2.imwrite(os.path.join(direct, str(counter)+'.jpg'), frame)
				counter +=1
			start += 1
			
	def findMax(self, values):
		return int(os.path.basename(values).split('.')[0])

	def check_last(self, direct):
		max_value = 0
		files =  glob.glob(os.path.join(direct, r'*.jpg'))
		if not files:
			return max_value
		max_value = int(os.path.basename(max(files,key=self.findMax)).split('.')[0])
		if not max_value:
			return 1
		return max_value + 1




if __name__ == '__main__':
	root = Image_root()
	root.mainloop()
