"""Render README figures from mantle 3D cases with ParaView (pvpython).

usage: pvpython docs/render_figures_v1.py CASE_DIR OUT_PREFIX [--slice-z Z] [--focus x,y,z] [--zoom S] [--view slice|clip]

slice: planar cut of the volume mesh (polygon cells, edges drawn), coloured by log10 cell volume, so the thin
boundary layers, the pillow layer and the octree core are visible at a glance.
clip:  3D view: wall surface plus a crinkle clip of the cells (half the domain removed), same colouring.
Writes OUT_PREFIX.png (1600 x 900).
"""
import argparse, math, os
from paraview.simple import *

ap = argparse.ArgumentParser()
ap.add_argument('case'); ap.add_argument('out')
ap.add_argument('--slice-z', type=float, default=0.0)
ap.add_argument('--axis', choices=['x', 'y', 'z'], default='z', help='slice normal axis (slice view)')
ap.add_argument('--focus', default='0,0,0'); ap.add_argument('--zoom', type=float, default=1.0)
ap.add_argument('--view', choices=['slice', 'clip'], default='slice')
ap.add_argument('--res', default='1600,900')
ap.add_argument('--bg', default='1,1,1', help='background RGB in 0..1')
a = ap.parse_args()
foam = os.path.join(a.case, 'case.foam')
open(foam, 'a').close()
fx, fy, fz = map(float, a.focus.split(','))
W, H = map(int, a.res.split(','))

r = OpenFOAMReader(FileName=foam)
r.MeshRegions = ['internalMesh', 'patch/wall']
r.CellArrays = []
r.UpdatePipeline()
internal = ExtractBlock(Input=r); internal.Selectors = ['/Root/internalMesh']
wall = ExtractBlock(Input=r); wall.Selectors = ['/Root/boundary/wall']
size = CellSize(Input=internal); size.ComputeVolume = 1
logv = Calculator(Input=size); logv.AttributeType = 'Cell Data'; logv.ResultArrayName = 'logV'; logv.Function = 'log10(abs(Volume)+1e-30)'

view = CreateView('RenderView'); view.ViewSize = [W, H]; view.OrientationAxesVisibility = 0
view.Background = [float(x) for x in a.bg.split(',')]
try:
    view.UseColorPaletteForBackground = 0
except AttributeError:
    pass
lut = GetColorTransferFunction('logV'); lut.ApplyPreset('Viridis (matplotlib)', True)

if a.view == 'slice':
    k = 'xyz'.index(a.axis); nrm = [0.0, 0.0, 0.0]; nrm[k] = 1.0
    org = [fx, fy, fz]; org[k] = a.slice_z
    s = Slice(Input=logv); s.SliceType = 'Plane'; s.SliceType.Origin = org; s.SliceType.Normal = nrm
    s.Triangulatetheslice = 0
    d = Show(s, view); d.Representation = 'Surface With Edges'; ColorBy(d, ('CELLS', 'logV'))
    d.EdgeColor = [0.15, 0.15, 0.15]; d.LineWidth = 0.6
    d.RescaleTransferFunctionToDataRange(True, False)
    cam = list(org); cam[k] += 10
    view.CameraPosition = cam; view.CameraFocalPoint = org; view.CameraViewUp = [0, 0, 1] if k != 2 else [0, 1, 0]
    view.CameraParallelProjection = 1; view.CameraParallelScale = 1.0 / a.zoom
else:
    c = Clip(Input=logv); c.ClipType = 'Plane'; c.ClipType.Origin = [fx, fy, fz]; c.ClipType.Normal = [0, 0, 1]; c.Crinkleclip = 1
    near = Clip(Input=c); near.ClipType = 'Sphere'; near.ClipType.Center = [fx, fy, fz]; near.ClipType.Radius = 1.2 / a.zoom; near.Crinkleclip = 1; near.Invert = 1
    d = Show(near, view); d.Representation = 'Surface With Edges'; ColorBy(d, ('CELLS', 'logV'))
    d.EdgeColor = [0.12, 0.12, 0.12]; d.LineWidth = 0.5
    d.RescaleTransferFunctionToDataRange(True, False)
    wd = Show(wall, view); wd.Representation = 'Surface'; wd.ColorArrayName = ['POINTS', '']; wd.AmbientColor = wd.DiffuseColor = [0.80, 0.42, 0.20]
    view.CameraFocalPoint = [fx, fy, fz]; view.CameraViewUp = [0, 0, 1]
    dist = 3.2 / a.zoom
    view.CameraPosition = [fx + dist * math.cos(0.9) * 0.8, fy - dist * math.sin(0.9) * 0.8, fz + dist * 0.55]
view.EnableRenderOnInteraction = 0
SaveScreenshot(a.out + '.png', view, ImageResolution=[W, H], TransparentBackground=0)
print('wrote', a.out + '.png')
