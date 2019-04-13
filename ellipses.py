import numpy as np
import matplotlib.pyplot as plt
import shutil, os

angles = np.arange(0.0, 2.0*np.pi, 0.01)
angles = np.array(list(angles) + [0.0])

def rotation(pts, angle):
    x = pts[:,0] * np.cos(angle) - pts[:,1] * np.sin(angle)
    y = pts[:,0] * np.sin(angle) + pts[:,1] * np.cos(angle)

    return np.stack((x, y)).T

def ellipse(b, e):
    r = b / np.sqrt(1 - (e * np.cos(angles)) ** 2.0)
    return np.stack((r * np.cos(angles), r * np.sin(angles))).T

spiral_param  = 0.2
spiral_length = np.pi

da = 0.05
var_a = np.arange(1.0, spiral_length, da)
e  = 0.5

ite = 0
alpha_max = 5.0
alpha_v = np.arange(0.0, alpha_max, 1.0)

if os.path.exists('render_orbits'):
    shutil.rmtree('render_orbits')
os.mkdir('render_orbits')

for alpha in alpha_v:
    print('Rendering iteration {}/{}'.format(ite+1, alpha_v.shape[0]))
    C = 0

    plt.close('all')
    fig = plt.figure(figsize=(10, 10))

    for a in var_a:
        b = a * np.sqrt(1.0 - e**2.0)

        angle = -alpha * np.log(a) + C

        pts = ellipse(b, e)
        pts = rotation(pts, angle)
        plt.plot(pts[:,0], pts[:,1], '-k', alpha=0.5)

    plt.xlim(-3.5, 3.5)
    plt.ylim(-3.5, 3.5)
    plt.savefig('render_orbits/img_{:04}.png'.format(ite))
    ite+=1

plt.show()
