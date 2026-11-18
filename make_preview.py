"""One-off documentation screenshot with simulated audio; no microphone."""
import tkinter as tk
import numpy as np
from PIL import ImageGrab
from retro_autotune_gui import StudioApp
root = tk.Tk()
app = StudioApp(root)
root.update()
t = np.arange(2048) / 48000
signal = (0.35*np.sin(2*np.pi*440*t)+0.18*np.sin(2*np.pi*880*t)+0.07*np.sin(2*np.pi*1320*t)).astype('float32')
app._draw_visuals({'samples':signal,'note':'A4','chord':'Am', 'follow_voice':True,
                   'confidence':.98,'input_db':-16,'output_db':-20,'beat':1})
app.status_var.set('PREVIEW · Simulated vocal, no microphone')
root.update()
root.update_idletasks()
# Capture current window with the GUI toolbar and new guitar/follow controls.
ImageGrab.grab(bbox=(0,0,1140,850)).save('assets/gui_preview.png')
root.destroy()
