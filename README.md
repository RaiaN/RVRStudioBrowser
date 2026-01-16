# RVR Studio - 3D Photo Viewer

An interactive browser-based 3D photo viewer that creates a stereoscopic-like experience from 2D images using depth maps. Built with Three.js for real-time rendering.

![Preview](preview.png)

## Features

✨ **Depth-Displaced Rendering**
- Maps images onto subdivided mesh geometry
- Displaces vertices based on depth map values
- Adjustable depth intensity for dramatic or subtle effects

🎯 **Mouse-Driven Parallax**
- Smooth camera movement following mouse position
- Configurable parallax strength
- Creates convincing 3D depth perception

👁️ **Stereoscopic Mode**
- Side-by-side stereo rendering for VR/3D displays
- Adjustable eye separation
- Toggle on/off via UI

🖼️ **Multi-Image Navigation**
- Smooth transitions between images
- Keyboard and click navigation
- Thumbnail strip for quick access

⚙️ **Real-Time Controls**
- Depth intensity slider
- Parallax strength adjustment
- Mesh resolution selector
- Wireframe debug mode
- Auto camera dolly animation

## Quick Start

### Option 1: Direct Browser (Recommended)
Simply serve the files with any local HTTP server:

```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx serve

# Using PHP
php -S localhost:8000
```

Then open `http://localhost:8000` in your browser.

### Option 2: VS Code Live Server
1. Install the "Live Server" extension
2. Right-click `index.html` → "Open with Live Server"

## Adding Your Own Images

### File Structure
```
assets/
├── sample1.jpg           # Your photo
├── sample1_depth.png     # Corresponding depth map
├── sample2.jpg
├── sample2_depth.png
└── ...
```

### Update `js/main.js`
Edit the `SAMPLE_IMAGES` array:

```javascript
const SAMPLE_IMAGES = [
    {
        image: 'assets/your_photo.jpg',
        depth: 'assets/your_photo_depth.png',
        name: 'Your Photo Title'
    },
    // Add more images...
];
```

## Creating Depth Maps

### Method 1: AI Depth Estimation (Recommended)
Use the included Python script with MiDaS:

```bash
cd tools
pip install torch torchvision opencv-python pillow
python generate_depth_map.py ../assets/photo.jpg ../assets/photo_depth.png --method midas
```

### Method 2: iPhone Spatial Photos
1. Take a photo with iPhone 15 Pro/Pro Max in Spatial mode
2. Export the depth map using apps like:
   - Halide
   - Focos
   - ProCamera

### Method 3: Manual Creation
Use Photoshop/GIMP to paint depth maps:
- **White** = Close to camera
- **Black** = Far from camera
- **Gray** = Middle distance

Use gradients and layers for best results.

### Method 4: Simple Edge-Based (Fast)
```bash
python tools/generate_depth_map.py photo.jpg depth.png --method edges
```

## Depth Map Guidelines

| Aspect | Recommendation |
|--------|----------------|
| Format | PNG (grayscale) |
| Resolution | Match source image |
| Bit Depth | 8-bit minimum |
| Values | 0 (far) to 255 (close) |

**Tips for better depth maps:**
- Use smooth gradients, avoid hard edges
- Consider scene geometry (floor recedes, sky is distant)
- Foreground subjects should be brightest
- Subtle variations create more realistic effect

## Controls

### Mouse
- **Move** - Camera parallax
- **Scroll** - Zoom in/out

### Keyboard
- **← →** - Previous/Next image
- **R** - Reset camera

### UI Panel
- **Depth Intensity** - Amplify Z displacement
- **Parallax Strength** - Camera movement sensitivity
- **Mesh Resolution** - Geometry detail level
- **Stereo Toggle** - Side-by-side 3D mode
- **Auto Dolly** - Automated camera movement
- **Wireframe** - Debug mesh visualization

## Technical Details

### Architecture
```
├── index.html          # Main HTML with UI
├── css/
│   └── style.css       # Styling and themes
├── js/
│   ├── main.js         # Entry point, UI wiring
│   ├── DepthViewer.js  # Core 3D viewer class
│   └── ImageScene.js   # Per-image mesh management
├── assets/             # Images and depth maps
└── tools/
    └── generate_depth_map.py  # Depth generation script
```

### Rendering Pipeline
1. Load image texture + depth map
2. Create subdivided PlaneGeometry
3. Sample depth at each vertex UV
4. Displace vertex Z based on depth value
5. Apply image texture to mesh
6. Render with parallax camera movement

### Performance Considerations
- Default mesh resolution: 128x128 vertices
- Higher resolution = more detail but slower
- Depth sampling uses bilinear interpolation
- Textures are compressed and cached

## Browser Support

| Browser | Support |
|---------|---------|
| Chrome 90+ | ✅ Full |
| Firefox 88+ | ✅ Full |
| Safari 15+ | ✅ Full |
| Edge 90+ | ✅ Full |

Requires WebGL 2.0 support.

## API Reference

### Loading Custom Images via Console

```javascript
// Access the viewer
const viewer = window.RVRViewer.getViewer();

// Load new images programmatically
window.RVRViewer.loadImages([
    { image: 'path/to/img.jpg', depth: 'path/to/depth.png', name: 'Title' }
]);

// Adjust settings
viewer.setDepthIntensity(0.8);  // 0-1
viewer.setParallaxStrength(0.5);  // 0-1
viewer.setStereoMode(true);
viewer.setAutoDolly(true);
```

## Stretch Features

- [x] Depth exaggeration slider
- [x] Stereoscopic side-by-side mode
- [x] Auto camera dolly animation
- [x] Wireframe debug mode
- [ ] Depth-based object highlighting (hover)
- [ ] Video/animated depth maps
- [ ] Multi-layer parallax (foreground/background separation)
- [ ] VR headset support (WebXR)

## Credits

- **Three.js** - 3D rendering engine
- **MiDaS** - Depth estimation model (optional)
- **RVR Studio Team** - Development

## License

MIT License - Feel free to use in personal and commercial projects.

---

Made with ❤️ for immersive photo viewing experiences.
