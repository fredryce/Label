from jnius import autoclass, PythonJavaClass, java_method



Camera = autoclass('android.hardware.Camera')
SurfaceTexture = autoclass('android.graphics.SurfaceTexture')

GL_TEXTURE_EXTERNAL_OES = 36197


class CameraPreview(Image):
    play = BooleanProperty(False)

    resolution = ListProperty([640, 480])

    _camera = None
    _previewCallback = None
    _previewTexture = None

    secondary_texture = None

    def __init__(self, **kwargs):
        self.canvas = RenderContext()
        super(CameraPreview, self).__init__(**kwargs)
        self.bind(size=self.size_changed)

        # (2)
        self.canvas.shader.fs = '''
            #extension GL_OES_EGL_image_external : require
            #ifdef GL_ES
                precision highp float;
            #endif

            /* Outputs from the vertex shader */
            varying vec4 frag_color;
            varying vec2 tex_coord0;

            /* uniform texture samplers */
            uniform sampler2D texture0;
            uniform samplerExternalOES texture1;

            void main()
            {
                gl_FragColor = texture2D(texture1, tex_coord0);
            }
        '''
        # This is needed for the default vertex shader.
        self.canvas['projection_mat'] = Window.render_context['projection_mat']

        with self.canvas.before:
            # (4)
            Callback(self.draw_callback)
        with self.canvas:
            # (3)
            BindTexture(texture=self.secondary_texture, index=1)
        self.canvas['secondary_texture'] = 1

        # (1)
        tex_id = kivy.graphics.opengl.glGenTextures(1)[0]
        kivy.graphics.opengl.glBindTexture(GL_TEXTURE_EXTERNAL_OES, tex_id)
        width, height = self.resolution
        self.secondary_texture = Texture(width=width, height=height, target=GL_TEXTURE_EXTERNAL_OES, texid=int(tex_id), colorfmt='rgba')
        # (6)
        self._camera = Camera.open()
        self._previewTexture = SurfaceTexture(int(tex_id))
        self._camera.setPreviewTexture(self._previewTexture)

    def draw_callback(self, instr):
        if self._previewTexture:
            self._previewTexture.updateTexImage()

    def config_camera(self, surface):
        self._camera.setPreviewTexture(surface)

    def update_canvas(self, dt):
        self.canvas.ask_update()

    def size_changed(self, *largs):
        pass


    def on_play(self, instance, value):
        if not self._camera:
            return
        if value:
            self._camera.startPreview()

            # (5)
            Clock.schedule_interval(self.update_canvas, 1.0/30)
        else:
            Clock.unschedule(self.update_canvas)
            self._camera.stopPreview()
