import keras
import numpy as np
import os
import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

class CamApp(App):

    def build(self):
        self.webCam = Image(size_hint=(1, 0.8))
        self.button = Button(text="Take Photo", on_press=self.Classify, size_hint=(1, 0.1))
        self.label = Label(text="Pending", size_hint=(1, 0.1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.webCam)
        layout.add_widget(self.button)
        layout.add_widget(self.label)

        self.model = keras.models.load_model(os.path.join('App', 'Appdata', 'TabletIdentifier_CNN.h5'))

        self.capture = cv2.VideoCapture(1)
        Clock.schedule_interval(self.update, 1.0/33.0)

        self.Labels = {0: 'coldact', 1: 'Dolo', 2: 'Ultracet'}

        return layout

    def update(self, *args):
        ret, frame = self.capture.read()

        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.webCam.texture = img_texture

    def preprocess(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))
        image = np.reshape(image, [1, 480, 640, 3])
        return image

    def Classify(self, *args):
        ret, frame = self.capture.read()

        cv2.imwrite(os.path.join('App/Appdata', 'Input_Images', 'input_Image.jpg'), frame)
        img = self.preprocess(os.path.join('App/Appdata', 'Input_Images', 'input_Image.jpg'))
        self.label.text = self.Labels[np.array(self.model.predict(img)).argmax()]


if __name__ == '__main__':
    CamApp().run()
