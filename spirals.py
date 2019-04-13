import numpy as np
import time
import shutil
import sys, os
from vispy import gloo, app, scene, visuals, io
from visuals import Galaxy, Trajectory

if '--mode' in sys.argv:
    i = sys.argv.index('--mode')
    mode = sys.argv[i+1]
else:
    mode = 'solid'

available_modes = ('solid', 'differential_1', 'differential_2', 'density_wave', 'density_wave_pattern')
if mode not in available_modes:
    print('Error, unknown mode {}'.format(mode))
    print('Available modes are :', ', '.join(available_modes))
    exit(1)

render_mode = '--render' in sys.argv
plot_traj = '--plot-traj' in sys.argv

if '--n-frames' in sys.argv:
    i = sys.argv.index('--n-frames')
    n_frames = int(sys.argv[i+1])
else:
    n_frames = 1800 # 30s

def rotation(pts, angle):
    x = pts[:,0] * np.cos(angle) - pts[:,1] * np.sin(angle)
    y = pts[:,0] * np.sin(angle) + pts[:,1] * np.cos(angle)

    #y[angle < 0.0] *= -1
            
    return np.stack((x, y)).T

def ellipse(b, e, angles):
    r = b / np.sqrt(1 - (e * np.cos(angles)) ** 2.0)
    return np.stack((r * np.cos(angles), r * np.sin(angles))).T


### Based on the Boids example of Vispy

class Canvas(scene.SceneCanvas):
    def __init__(self, mode):
        scene.SceneCanvas.__init__(self, size=(1024, 1024), keys='interactive')

        self.unfreeze()

        self.view = self.central_widget.add_view()
        self.visuals = []

        ps = self.pixel_scale
        self.iteration = 0
        
        # Create particles
        self.galaxy = Galaxy(mode, parent=self.view.scene)
        self.visuals.append(self.galaxy)

        if plot_traj:
            self.traj1_id = 1000
            self.traj1_pts = [self.galaxy.positions[self.traj1_id,:]]
            self.traj2_id = 10000
            self.traj2_pts = [self.galaxy.positions[self.traj2_id,:]]
            self.line1 = Trajectory(self.traj1_pts, color=(0.7, 0, 0,   1), parent=self.view.scene)
            self.line2 = Trajectory(self.traj2_pts, color=(0.2, 0, 0.7, 1), parent=self.view.scene)
            
        # Time
        self._t = time.time()

        width, height = self.physical_size
        gloo.set_viewport(0, 0, width, height)

        self.dest = 'render_' + mode
        if render_mode:
            if os.path.exists(self.dest):
                shutil.rmtree(self.dest)
            os.mkdir(self.dest)

        gloo.set_state(clear_color=(0, 0, 0, 1), blend=True,
                       blend_func=('src_alpha', 'one'))

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self._timer.start()

        self.freeze()

        self.show()

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    def on_timer(self, event):
        self.galaxy.iterate()

        if plot_traj:
            self.line1.add_point(tuple(self.galaxy.positions[self.traj1_id,:]))
            self.line2.add_point(tuple(self.galaxy.positions[self.traj2_id,:]))

        self.update()
        self._draw_scene()

        if render_mode:
            img = gloo.util._screenshot()
            filename = 'img_{:04d}.png'.format(self.iteration)
            print('Rendering img {}/{}'.format(self.iteration+1, n_frames))
            io.imsave(os.path.join(self.dest, filename), img[:,:,:3])
            self.iteration += 1
            if self.iteration == n_frames:
                exit(0)

        
if __name__ == '__main__':
    c = Canvas(mode)
    app.run()
