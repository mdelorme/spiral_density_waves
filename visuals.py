import numpy as np
import vispy
from vispy import visuals, gloo, scene, io

from utils import *

vs_galaxy = '''
#version 120
void main() {
    gl_Position = vec4($position, 1.0);
    gl_PointSize = $psize;
}
'''

fs_galaxy = '''
#version 120
void main() {
    float x = 2.0*gl_PointCoord.x - 1.0;
    float y = 2.0*gl_PointCoord.y - 1.0;
    float a = 1.0 - (x*x + y*y);
    gl_FragColor = vec4($color.rgb, a*$color.a);
}
'''

vs_traj = '''
#version 120
attribute vec3 a_position;
void main(void) {
  gl_Position = vec4(a_position, 1.0);
  gl_PointSize = 4.0f;
}
'''

fs_traj = '''
#version 120
void main() {
  gl_FragColor = $color;
}
'''

class TrajVisual(visuals.Visual):
    def __init__(self, pos, color):
        visuals.Visual.__init__(self, vcode=vs_galaxy, fcode=fs_galaxy)

        self.unfreeze()
        self._draw_mode = 'points'
        
        self.positions = pos
        self.psize     = 5.0
        self.color     = color
        self.length    = 1800

        p = np.array(pos, dtype=np.float32)
        self.pos_vbo = gloo.VertexBuffer(p.copy())

        self.freeze()

    def add_point(self, point):
        self.positions.append(point)
        self.positions = self.positions[-self.length:]
        p = np.array(self.positions, dtype=np.float32)
        self.pos_vbo = gloo.VertexBuffer(p.copy())

    def _prepare_transforms(self, view):
        pass

    def _prepare_draw(self, view):
        self.shared_program.vert['position'] = self.pos_vbo
        self.shared_program.vert['psize']    = self.psize
        self.shared_program.frag['color']    = self.color


class GalaxyVisual(visuals.Visual):
    def __init__(self, mode, stype='trailing'):
        # Super class initialisation
        visuals.Visual.__init__(self, vcode=vs_galaxy, fcode=fs_galaxy)

        self.unfreeze()
        self.stype = stype
        self.mode = mode
        self.N = 100000
        self._draw_mode = 'points'
        
        self.pos_vbo = None

        self.positions = None
        self.psize     = None
        self.color     = None

        # Orbital parameters
        self.a           = None
        self.b           = None
        self.angle       = None
        self.orbit_angle = None
        self.e           = 0.6

        self.alpha             = 2.0

        if mode == 'solid':
            self.angular_velocity = 0.01
        elif mode == 'differential_1':
            self.angular_velocity = 0.02
        elif mode == 'differential_2':
            self.angular_velocity = 0.1
        else:
            self.angular_velocity = 0.01

        self.pattern_speed = 0.001
            
        self.angular_dampening = 0.3
        
        self.generate_data()

        # Texture and drawing mode
        self._draw_mode = 'points'
        self.freeze()

    def generate_data(self):
        self.a = np.abs(np.random.randn(self.N) * 0.5)
        self.b = self.a * np.sqrt(1.0 - self.e**2.0)
        y = np.zeros((self.N,))

        # Angle of the orbit
        if self.stype == 'trailing':
            self.orbit_angle = -self.alpha * np.log(self.a)
        else:
            self.orbit_angle = self.alpha * np.log(self.a)

        # Base position of the particle on the orbit
        self.angle = np.random.rand(self.N) * np.pi * 2.0

        # Generating positions
        pts = ellipse(self.b, self.e, self.angle)

        # ... and rotating them along with the orbit
        pts = rotation(pts, self.orbit_angle)

        self.positions = np.zeros((self.N, 3), dtype=np.float32)
        self.positions[:,:2] = pts
        self.psize = 4#  * self.pixel_scale
        self.color = 0.8, 0.8, 1, 0.1

        self.pos_vbo = gloo.VertexBuffer(self.positions.copy())

    def iterate(self):
        if self.mode == 'solid':
            r = np.linalg.norm(self.positions, axis=1)
            pr = self.positions
            self.angle = np.arctan2(pr[:,1], pr[:,0])
            self.angle += self.angular_velocity

            self.positions[:,0] = r * np.cos(self.angle)
            self.positions[:,1] = r * np.sin(self.angle)
        elif self.mode == 'differential_1':
            r = np.linalg.norm(self.positions, axis=1)
            self.angle = np.arctan2(self.positions[:,1], self.positions[:,0])
            self.angle += self.angular_velocity / r**self.angular_dampening

            self.positions[:,0] = r * np.cos(self.angle)
            self.positions[:,1] = r * np.sin(self.angle)
        elif self.mode == 'differential_2':
            r = np.linalg.norm(self.positions, axis=1)
            max_r = r.max() * 1.1
            self.angle = np.arctan2(self.positions[:,1], self.positions[:,0])
            self.angle += self.angular_velocity / (max_r - r)**(1.0 / self.angular_dampening)

            self.positions[:,0] = r * np.cos(self.angle)
            self.positions[:,1] = r * np.sin(self.angle)            
        elif self.mode == 'density_wave':
            if self.stype == 'trailing':
                self.angle = self.angle + self.angular_velocity
            else:
                self.angle = self.angle - self.angular_velocity
                
            pts = ellipse(self.b, self.e, self.angle)
            self.positions[:,:2] = rotation(pts, self.orbit_angle)
        else:
            if self.stype == 'trailing':
                self.angle = self.angle + self.angular_velocity
            else:
                self.angle = self.angle - self.angular_velocity

            self.orbit_angle += self.pattern_speed
                
            pts = ellipse(self.b, self.e, self.angle)
            self.positions[:,:2] = rotation(pts, self.orbit_angle)
            
        self.pos_vbo.set_data(self.positions.copy())

    def _prepare_transforms(self, view):
        pass

    def _prepare_draw(self, view):
        self.shared_program.vert['position'] = self.pos_vbo
        self.shared_program.vert['psize']    = self.psize
        self.shared_program.frag['color']    = self.color

Trajectory = scene.visuals.create_visual_node(TrajVisual)
Galaxy = scene.visuals.create_visual_node(GalaxyVisual)
