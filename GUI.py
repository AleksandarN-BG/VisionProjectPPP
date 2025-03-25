from kivy.graphics import Color, Rectangle
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label


class CameraView(BoxLayout):
    gesture_text = StringProperty("No gesture detected")

    def __init__(self, **kwargs):
        super(CameraView, self).__init__(orientation='vertical', **kwargs)

        # Main layout
        self.main_layout = BoxLayout(orientation='vertical')

        # Camera image view
        self.img = Image()
        self.main_layout.add_widget(self.img)

        # Bottom bar with info label and camera button
        self.bottom_bar = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))

        # Info label
        self.info_label = Label(
            text=self.gesture_text,
            size_hint=(0.8, 1),
            color=(1, 1, 1, 1),
            font_size='24sp',
            halign='center',
            valign='middle'
        )
        self.info_label.bind(size=self._update_rect, pos=self._update_rect)

        with self.info_label.canvas.before:
            Color(0, 0.5, 0.5, 0.8)  # Teal background
            self.rect = Rectangle(size=self.info_label.size, pos=self.info_label.pos)

        # Camera selection button
        self.cam_button = Button(
            text='Camera:',
            size_hint=(0.2, 1),
            background_color=(0.3, 0.3, 0.9, 0.8),
            font_size='24sp'
        )

        # Add components to bottom bar
        self.bottom_bar.add_widget(self.info_label)
        self.bottom_bar.add_widget(self.cam_button)

        # Add everything to main layout
        self.main_layout.add_widget(self.bottom_bar)
        self.add_widget(self.main_layout)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def update_gesture_text(self, text):
        self.info_label.text = text
        self.gesture_text = text
