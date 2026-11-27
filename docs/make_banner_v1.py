"""Compose the README banner and gallery images from renders made with render_figures_v1.py.

usage: python3 docs/make_banner_v1.py RENDER_DIR
RENDER_DIR must hold sphere_clip.png, cube_clip.png, cube_corner.png, sphere_zoom.png (1600 x 900 renders) and sphere_banner.png,
cube_banner.png (rendered with --res 1024,1000 --bg 0.059,0.090,0.125 so they blend into the banner).
Writes docs/figures/banner.png and docs/figures/3d_*.png (downscaled to 1200 px wide).
"""
import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as im

src = sys.argv[1]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(out, exist_ok=True)


def crop(img, x0, x1, y0, y1):
    h, w = img.shape[:2]
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


# banner: title block on the left, two 3D cut-away renders on the right
fig = plt.figure(figsize=(16, 5), dpi=100)
fig.patch.set_facecolor('#0f1720')
t = fig.add_axes([0.0, 0.0, 0.36, 1.0]); t.axis('off'); t.set_facecolor('#0f1720')
t.text(0.08, 0.66, 'mantle', color='white', fontsize=64, fontweight='bold', family='DejaVu Sans')
t.text(0.08, 0.50, 'shell-first hybrid meshing for CFD', color='#e8a060', fontsize=17)
t.text(0.08, 0.36, 'boundary layer first, bulk second\nOpenFOAM + SU2  ·  dirty STL in  ·  Rust core', color='#c9d3dc', fontsize=13,
       linespacing=1.6)
for k, name in enumerate(['sphere_banner', 'cube_banner']):
    a = fig.add_axes([0.36 + k * 0.32, 0.0, 0.32, 1.0]); a.axis('off')
    a.imshow(crop(im.imread(os.path.join(src, name + '.png')), 0.0, 1.0, 0.0, 1.0))
fig.savefig(os.path.join(out, 'banner.png'), facecolor=fig.get_facecolor())
plt.close(fig)

# gallery images
for name, new in [('sphere_clip', '3d_sphere_cutaway'), ('cube_clip', '3d_cube_cutaway'), ('cube_corner', '3d_cube_edge_fan'),
                  ('sphere_zoom', '3d_sphere_layers')]:
    img = im.imread(os.path.join(src, name + '.png'))
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(12, 12 * h / w), dpi=100)
    a = fig.add_axes([0, 0, 1, 1]); a.axis('off'); a.imshow(img)
    fig.savefig(os.path.join(out, new + '.png'))
    plt.close(fig)
print('wrote', out)
