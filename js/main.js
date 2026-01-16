/**
 * main.js
 * Entry point for RVR Studio 3D Photo Viewer
 * 
 * Initializes the depth viewer and connects UI controls.
 * Supports both depth-based parallax and BLADE-derived 3D mesh viewing.
 */

import { DepthViewer } from './DepthViewer.js';
import { RenderMode, DebugMode } from './ImageScene.js';
import { MeshDisplayMode, LightingPreset } from './MeshScene.js';

// =============================================================================
// Sample Image Data
// =============================================================================
// Add your own images and depth maps to the assets folder and update this array
// Format: { image: 'path/to/image.jpg', depth: 'path/to/depth.png', name: 'Display Name' }

// Only image + depth needed for parallax effect!
const SAMPLE_IMAGES = [
    {
        image: 'assets/IMG_20260116_161032_363.jpg',
        depth: 'assets/IMG_20260116_161032_363_depth.png',
        name: 'He'
    },
    {
        image: 'assets/IMG_20260116_161405_615.jpg',
        depth: 'assets/IMG_20260116_161405_615_depth.png',
        name: 'Cat'
    },
    {
        image: 'assets/IMG_20260116_163328_560.jpg',
        depth: 'assets/IMG_20260116_163328_560_depth.png',
        name: 'She'
    }
];

// Sample 3D meshes (BLADE/SOTA humanoid meshes from glTF/GLB files)
// Generate these using: 
//   python tools/generate_mesh.py image.jpg mesh.glb --method hmr2
//   OR with depth: python tools/generate_mesh.py image.jpg mesh.glb --method depth --depth depth.png
// Batch: python tools/generate_mesh.py --batch ./assets/ ./assets/meshes/ --method depth --depth ./depths/
const SAMPLE_MESHES = [
    {
        mesh: 'assets/meshes/IMG_20260116_161032_363.glb',
        name: 'He (3D Mesh)'
    },
    {
        mesh: 'assets/meshes/IMG_20260116_161405_615.glb',
        name: 'Cat (3D Mesh)'
    },
    {
        mesh: 'assets/meshes/IMG_20260116_163328_560.glb',
        name: 'She (3D Mesh)'
    }
];

// =============================================================================
// Application State
// =============================================================================

let viewer = null;

// DOM Elements
const elements = {
    container: null,
    loadingOverlay: null,
    depthSlider: null,
    depthValue: null,
    parallaxSlider: null,
    parallaxValue: null,
    meshResolution: null,
    stereoToggle: null,
    autoDollyToggle: null,
    wireframeToggle: null,
    resetCameraBtn: null,
    prevBtn: null,
    nextBtn: null,
    currentIndex: null,
    totalImages: null,
    thumbnails: null,
    depthInfo: null,
    // Debug controls
    renderMode: null,
    debugMode: null,
    diagnoseBtn: null,
    diagnosticsPanel: null,
    diagnosticsContent: null,
    closeDiagnostics: null,
    // Multi-layer controls
    layerCountGroup: null,
    layerCount: null,
};

// =============================================================================
// Initialization
// =============================================================================

/**
 * Initialize the application
 */
async function init() {
    console.log('🎬 RVR Studio - 3D Photo Viewer');
    console.log('Initializing...');
    
    // Cache DOM elements
    cacheElements();
    
    // Initialize the 3D viewer
    viewer = new DepthViewer(elements.container);
    
    // Set up UI event listeners
    setupUIControls();
    
    // Set up viewer event listeners
    setupViewerEvents();
    
    // Load sample images
    try {
        // First try to load actual sample images
        await loadSampleImages();
    } catch (error) {
        console.warn('Could not load sample images, generating placeholders...', error);
        // Generate placeholder images if samples don't exist
        await loadPlaceholderImages();
    }
    
    // Hide loading overlay
    hideLoading();
    
    console.log('✅ Initialization complete');
}

/**
 * Cache DOM element references
 */
function cacheElements() {
    elements.container = document.getElementById('viewer-container');
    elements.loadingOverlay = document.getElementById('loading-overlay');
    elements.depthSlider = document.getElementById('depth-slider');
    elements.depthValue = document.getElementById('depth-value');
    elements.parallaxSlider = document.getElementById('parallax-slider');
    elements.parallaxValue = document.getElementById('parallax-value');
    elements.meshResolution = document.getElementById('mesh-resolution');
    elements.stereoToggle = document.getElementById('stereo-toggle');
    elements.autoDollyToggle = document.getElementById('auto-dolly-toggle');
    elements.wireframeToggle = document.getElementById('wireframe-toggle');
    elements.resetCameraBtn = document.getElementById('reset-camera');
    elements.prevBtn = document.getElementById('prev-btn');
    elements.nextBtn = document.getElementById('next-btn');
    elements.currentIndex = document.getElementById('current-index');
    elements.totalImages = document.getElementById('total-images');
    elements.thumbnails = document.getElementById('thumbnails');
    elements.depthInfo = document.getElementById('depth-info');
    // Debug controls
    elements.renderMode = document.getElementById('render-mode');
    elements.debugMode = document.getElementById('debug-mode');
    elements.diagnoseBtn = document.getElementById('diagnose-btn');
    elements.diagnosticsPanel = document.getElementById('diagnostics-panel');
    elements.diagnosticsContent = document.getElementById('diagnostics-content');
    elements.closeDiagnostics = document.getElementById('close-diagnostics');
    // Multi-layer controls
    elements.layerCountGroup = document.getElementById('layer-count-group');
    elements.layerCount = document.getElementById('layer-count');
    // Hybrid controls
    elements.parallaxAmountGroup = document.getElementById('parallax-amount-group');
    elements.parallaxAmount = document.getElementById('parallax-amount');
    elements.parallaxAmountValue = document.getElementById('parallax-amount-value');
    elements.debugLayersGroup = document.getElementById('debug-layers-group');
    elements.debugLayersToggle = document.getElementById('debug-layers-toggle');
    
    // Mesh mode controls
    elements.meshModeToggle = document.getElementById('mesh-mode-toggle');
    elements.meshControlsGroup = document.getElementById('mesh-controls-group');
    elements.meshLightingPreset = document.getElementById('mesh-lighting-preset');
    elements.meshDisplayMode = document.getElementById('mesh-display-mode');
    elements.meshAutoRotate = document.getElementById('mesh-auto-rotate');
    elements.meshRimLight = document.getElementById('mesh-rim-light');
    elements.meshRimLightValue = document.getElementById('mesh-rim-light-value');
    elements.meshStatsBtn = document.getElementById('mesh-stats-btn');
    elements.meshStatsPanel = document.getElementById('mesh-stats-panel');
    elements.meshStatsContent = document.getElementById('mesh-stats-content');
    elements.closeMeshStats = document.getElementById('close-mesh-stats');
}

/**
 * Set up UI control event listeners
 */
function setupUIControls() {
    // Depth intensity slider
    elements.depthSlider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        const normalizedValue = value / 100;
        viewer.setDepthIntensity(normalizedValue);
        elements.depthValue.textContent = `${value}%`;
    });
    
    // Parallax strength slider
    elements.parallaxSlider.addEventListener('input', (e) => {
        const value = parseInt(e.target.value);
        const normalizedValue = value / 100;
        viewer.setParallaxStrength(normalizedValue);
        elements.parallaxValue.textContent = `${value}%`;
    });
    
    // Mesh resolution dropdown
    elements.meshResolution.addEventListener('change', (e) => {
        const resolution = parseInt(e.target.value);
        viewer.setMeshResolution(resolution);
    });
    
    // Stereo mode toggle
    elements.stereoToggle.addEventListener('change', (e) => {
        viewer.setStereoMode(e.target.checked);
    });
    
    // Auto dolly toggle
    elements.autoDollyToggle.addEventListener('change', (e) => {
        viewer.setAutoDolly(e.target.checked);
    });
    
    // Wireframe toggle
    elements.wireframeToggle.addEventListener('change', (e) => {
        viewer.setWireframe(e.target.checked);
    });
    
    // Reset camera button
    elements.resetCameraBtn.addEventListener('click', () => {
        viewer.resetCamera();
    });
    
    // Navigation buttons
    elements.prevBtn.addEventListener('click', () => {
        viewer.previousImage();
    });
    
    elements.nextBtn.addEventListener('click', () => {
        viewer.nextImage();
    });
    
    // === DEBUG CONTROLS ===
    
    // Render mode dropdown
    if (elements.renderMode) {
        elements.renderMode.addEventListener('change', (e) => {
            const mode = e.target.value;
            viewer.setRenderMode(mode);
            
            // Show/hide layer count control (multi-layer only)
            if (elements.layerCountGroup) {
                elements.layerCountGroup.style.display = mode === 'multiLayer' ? 'block' : 'none';
            }
            
            // Show/hide hybrid controls (hybrid only)
            const hybridGroups = document.querySelectorAll('.hybrid-group');
            hybridGroups.forEach(group => {
                group.style.display = mode === 'hybrid' ? 'block' : 'none';
            });
        });
        
        // Trigger initial visibility update
        const initialMode = elements.renderMode.value;
        const hybridGroups = document.querySelectorAll('.hybrid-group');
        hybridGroups.forEach(group => {
            group.style.display = initialMode === 'hybrid' ? 'block' : 'none';
        });
    }
    
    // Layer count dropdown (for multi-layer mode)
    if (elements.layerCount) {
        elements.layerCount.addEventListener('change', (e) => {
            const count = parseInt(e.target.value);
            viewer.setLayerCount(count);
        });
    }
    
    // === HYBRID MODE CONTROLS ===
    
    // Parallax amount slider (for hybrid UV-based parallax)
    if (elements.parallaxAmount) {
        elements.parallaxAmount.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            const normalizedValue = value / 100;  // Convert 1-10 to 0.01-0.10
            viewer.setParallaxAmount(normalizedValue);
            elements.parallaxAmountValue.textContent = `${value}%`;
        });
    }
    
    // Debug depth toggle (show depth visualization)
    if (elements.debugLayersToggle) {
        elements.debugLayersToggle.addEventListener('change', (e) => {
            viewer.setDebugLayers(e.target.checked);
        });
    }
    
    // Debug mode dropdown
    if (elements.debugMode) {
        elements.debugMode.addEventListener('change', (e) => {
            viewer.setDebugMode(e.target.value);
        });
    }
    
    // Diagnose button
    if (elements.diagnoseBtn) {
        elements.diagnoseBtn.addEventListener('click', () => {
            showDiagnostics();
        });
    }
    
    // Close diagnostics
    if (elements.closeDiagnostics) {
        elements.closeDiagnostics.addEventListener('click', () => {
            elements.diagnosticsPanel.classList.add('hidden');
        });
    }
    
    // === MESH MODE CONTROLS ===
    
    // Mesh mode toggle
    if (elements.meshModeToggle) {
        elements.meshModeToggle.addEventListener('change', async (e) => {
            const enabled = e.target.checked;
            
            if (enabled) {
                // Check if meshes are loaded, if not try to load them
                if (!viewer.hasMeshes()) {
                    console.log('Loading 3D meshes...');
                    showLoading();
                    try {
                        await viewer.loadMeshes(SAMPLE_MESHES);
                    } catch (error) {
                        console.warn('Failed to load meshes:', error);
                        e.target.checked = false;
                        hideLoading();
                        return;
                    }
                    hideLoading();
                }
                
                viewer.enableMeshMode();
            } else {
                viewer.disableMeshMode();
            }
            
            // Show/hide mesh controls
            if (elements.meshControlsGroup) {
                elements.meshControlsGroup.style.display = enabled ? 'block' : 'none';
            }
            
            // Update instructions
            updateInstructions(enabled);
        });
    }
    
    // Mesh lighting preset
    if (elements.meshLightingPreset) {
        elements.meshLightingPreset.addEventListener('change', (e) => {
            viewer.setMeshLightingPreset(e.target.value);
        });
    }
    
    // Mesh display mode
    if (elements.meshDisplayMode) {
        elements.meshDisplayMode.addEventListener('change', (e) => {
            viewer.setMeshDisplayMode(e.target.value);
        });
    }
    
    // Mesh auto-rotate
    if (elements.meshAutoRotate) {
        elements.meshAutoRotate.addEventListener('change', (e) => {
            viewer.setMeshAutoRotate(e.target.checked);
        });
    }
    
    // Mesh rim light intensity
    if (elements.meshRimLight) {
        elements.meshRimLight.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            const normalizedValue = value / 50;  // Convert 0-100 to 0-2
            viewer.setMeshRimLightIntensity(normalizedValue);
            if (elements.meshRimLightValue) {
                elements.meshRimLightValue.textContent = `${value}%`;
            }
        });
    }
    
    // Mesh stats button
    if (elements.meshStatsBtn) {
        elements.meshStatsBtn.addEventListener('click', () => {
            showMeshStats();
        });
    }
    
    // Close mesh stats
    if (elements.closeMeshStats) {
        elements.closeMeshStats.addEventListener('click', () => {
            if (elements.meshStatsPanel) {
                elements.meshStatsPanel.classList.add('hidden');
            }
        });
    }
}

/**
 * Show mesh statistics panel
 */
function showMeshStats() {
    const stats = viewer.getMeshStats();
    
    if (!stats || !elements.meshStatsPanel || !elements.meshStatsContent) {
        return;
    }
    
    elements.meshStatsContent.innerHTML = `
        <div class="stat-row">
            <span class="stat-label">Mesh Count</span>
            <span class="stat-value">${stats.meshCount}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Vertices</span>
            <span class="stat-value">${stats.vertexCount.toLocaleString()}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Triangles</span>
            <span class="stat-value">${stats.triangleCount.toLocaleString()}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Animations</span>
            <span class="stat-value">${stats.hasAnimations ? stats.animationCount : 'None'}</span>
        </div>
        ${stats.bounds ? `
        <div class="stat-row">
            <span class="stat-label">Bounds</span>
            <span class="stat-value" style="font-size: 10px;">
                Min: [${stats.bounds.min.map(v => v.toFixed(2)).join(', ')}]<br>
                Max: [${stats.bounds.max.map(v => v.toFixed(2)).join(', ')}]
            </span>
        </div>
        ` : ''}
    `;
    
    elements.meshStatsPanel.classList.remove('hidden');
}

/**
 * Update instructions based on mode
 * @param {boolean} isMeshMode - Whether mesh mode is active
 */
function updateInstructions(isMeshMode) {
    const instructionsEl = document.getElementById('instructions');
    if (instructionsEl) {
        if (isMeshMode) {
            instructionsEl.innerHTML = '<p>Drag to rotate • Scroll to zoom • Right-click to pan • Arrow keys to navigate</p>';
        } else {
            instructionsEl.innerHTML = '<p>Move mouse to look around • Scroll to zoom • Arrow keys to navigate</p>';
        }
    }
}

/**
 * Show depth map diagnostics panel
 */
function showDiagnostics() {
    const diagnostics = viewer.getDiagnostics();
    
    if (!diagnostics) {
        elements.diagnosticsContent.innerHTML = '<p>No image loaded</p>';
        elements.diagnosticsPanel.classList.remove('hidden');
        return;
    }
    
    // Build histogram visualization (simplified to 32 bins)
    const binSize = 8;
    const binnedHistogram = [];
    for (let i = 0; i < 256; i += binSize) {
        let sum = 0;
        for (let j = 0; j < binSize; j++) {
            sum += diagnostics.histogram[i + j] || 0;
        }
        binnedHistogram.push(sum);
    }
    const maxBin = Math.max(...binnedHistogram);
    
    const histogramHTML = binnedHistogram.map(count => {
        const height = maxBin > 0 ? (count / maxBin * 100) : 0;
        return `<div class="histogram-bar" style="height: ${height}%"></div>`;
    }).join('');
    
    // Build recommendations HTML
    const recommendationsHTML = diagnostics.recommendation.map(tip => 
        `<li>${tip}</li>`
    ).join('');
    
    elements.diagnosticsContent.innerHTML = `
        <div class="stat-row">
            <span class="stat-label">Resolution</span>
            <span class="stat-value">${diagnostics.width} × ${diagnostics.height}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Depth Range</span>
            <span class="stat-value">${diagnostics.minDepth} - ${diagnostics.maxDepth} (${diagnostics.depthRange})</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Average Depth</span>
            <span class="stat-value">${diagnostics.avgDepth.toFixed(1)}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Sharp Edges</span>
            <span class="stat-value">${diagnostics.sharpEdgePercentage}%</span>
        </div>
        
        <div class="histogram">
            ${histogramHTML}
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 10px; color: var(--text-muted); margin-top: 4px;">
            <span>Far (0)</span>
            <span>Depth Distribution</span>
            <span>Close (255)</span>
        </div>
        
        <div class="recommendations">
            <h4>Recommendations</h4>
            <ul>${recommendationsHTML}</ul>
        </div>
    `;
    
    elements.diagnosticsPanel.classList.remove('hidden');
}

/**
 * Set up viewer event listeners
 */
function setupViewerEvents() {
    // Images loaded event
    window.addEventListener('imagesLoaded', (e) => {
        const { count, images } = e.detail;
        
        elements.totalImages.textContent = count;
        elements.currentIndex.textContent = '1';
        
        // Create thumbnails
        createThumbnails(images);
        
        updateNavigationState();
    });
    
    // Image changed event
    window.addEventListener('imageChanged', (e) => {
        const { index } = e.detail;
        
        elements.currentIndex.textContent = index + 1;
        
        // Update thumbnail active state
        updateThumbnailState(index);
        
        updateNavigationState();
    });
    
    // Depth info update
    window.addEventListener('depthUpdate', (e) => {
        const { depth } = e.detail;
        elements.depthInfo.textContent = `Depth: ${depth}`;
    });
}

/**
 * Create thumbnail elements
 * @param {Array} images - Array of image data
 */
function createThumbnails(images) {
    elements.thumbnails.innerHTML = '';
    
    images.forEach((img, index) => {
        const thumb = document.createElement('div');
        thumb.className = 'thumbnail' + (index === 0 ? ' active' : '');
        thumb.dataset.index = index;
        
        const imgEl = document.createElement('img');
        imgEl.src = img.image;
        imgEl.alt = img.name;
        
        thumb.appendChild(imgEl);
        
        thumb.addEventListener('click', () => {
            viewer.goToImage(index);
        });
        
        elements.thumbnails.appendChild(thumb);
    });
}

/**
 * Update thumbnail active state
 * @param {number} activeIndex - Current active index
 */
function updateThumbnailState(activeIndex) {
    const thumbs = elements.thumbnails.querySelectorAll('.thumbnail');
    thumbs.forEach((thumb, index) => {
        thumb.classList.toggle('active', index === activeIndex);
    });
}

/**
 * Update navigation button states
 */
function updateNavigationState() {
    const currentIndex = viewer.getCurrentIndex();
    const totalImages = viewer.getImageCount();
    
    // Disable prev on first image
    elements.prevBtn.disabled = totalImages <= 1;
    elements.nextBtn.disabled = totalImages <= 1;
}

/**
 * Load sample images from assets folder
 */
async function loadSampleImages() {
    // Check if first sample image exists
    const testResponse = await fetch(SAMPLE_IMAGES[0].image, { method: 'HEAD' });
    
    if (!testResponse.ok) {
        throw new Error('Sample images not found');
    }
    
    await viewer.loadImages(SAMPLE_IMAGES);
}

/**
 * Generate and load placeholder images
 * Creates procedural images and depth maps for testing
 */
async function loadPlaceholderImages() {
    console.log('Generating placeholder images...');
    
    const placeholders = [
        generatePlaceholderImage(800, 600, 'Gradient Mountain', '#1a1a2e', '#4a148c', 'radial'),
        generatePlaceholderImage(800, 600, 'Ocean Depth', '#001f3f', '#39cccc', 'linear'),
        generatePlaceholderImage(800, 600, 'Forest Layers', '#1b4332', '#95d5b2', 'complex'),
    ];
    
    await viewer.loadImages(placeholders);
}

/**
 * Generate a placeholder image and depth map
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @param {string} name - Image name
 * @param {string} color1 - Primary color
 * @param {string} color2 - Secondary color
 * @param {string} pattern - Depth pattern type
 * @returns {Object} Image data object with data URLs
 */
function generatePlaceholderImage(width, height, name, color1, color2, pattern) {
    // Create image canvas
    const imageCanvas = document.createElement('canvas');
    imageCanvas.width = width;
    imageCanvas.height = height;
    const imageCtx = imageCanvas.getContext('2d');
    
    // Create depth canvas
    const depthCanvas = document.createElement('canvas');
    depthCanvas.width = width;
    depthCanvas.height = height;
    const depthCtx = depthCanvas.getContext('2d');
    
    // Generate image (gradient background with shapes)
    const imageGradient = imageCtx.createLinearGradient(0, 0, width, height);
    imageGradient.addColorStop(0, color1);
    imageGradient.addColorStop(1, color2);
    imageCtx.fillStyle = imageGradient;
    imageCtx.fillRect(0, 0, width, height);
    
    // Add some geometric shapes to the image
    imageCtx.globalAlpha = 0.3;
    for (let i = 0; i < 5; i++) {
        const x = Math.random() * width;
        const y = Math.random() * height;
        const radius = 50 + Math.random() * 100;
        
        imageCtx.beginPath();
        imageCtx.arc(x, y, radius, 0, Math.PI * 2);
        imageCtx.fillStyle = i % 2 === 0 ? color1 : color2;
        imageCtx.fill();
    }
    imageCtx.globalAlpha = 1;
    
    // Add title text
    imageCtx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    imageCtx.font = 'bold 32px Arial';
    imageCtx.textAlign = 'center';
    imageCtx.fillText(name, width / 2, height / 2);
    imageCtx.font = '16px Arial';
    imageCtx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    imageCtx.fillText('Placeholder Image', width / 2, height / 2 + 30);
    
    // Generate depth map based on pattern
    switch (pattern) {
        case 'radial':
            // Radial gradient depth (center closer)
            const radialGrad = depthCtx.createRadialGradient(
                width / 2, height / 2, 0,
                width / 2, height / 2, Math.max(width, height) / 2
            );
            radialGrad.addColorStop(0, '#ffffff');
            radialGrad.addColorStop(1, '#000000');
            depthCtx.fillStyle = radialGrad;
            depthCtx.fillRect(0, 0, width, height);
            break;
            
        case 'linear':
            // Linear gradient (top closer)
            const linearGrad = depthCtx.createLinearGradient(0, 0, 0, height);
            linearGrad.addColorStop(0, '#ffffff');
            linearGrad.addColorStop(1, '#333333');
            depthCtx.fillStyle = linearGrad;
            depthCtx.fillRect(0, 0, width, height);
            break;
            
        case 'complex':
        default:
            // Complex depth with multiple layers
            // Background (far)
            depthCtx.fillStyle = '#333333';
            depthCtx.fillRect(0, 0, width, height);
            
            // Middle ground
            depthCtx.fillStyle = '#888888';
            depthCtx.beginPath();
            depthCtx.moveTo(0, height * 0.6);
            depthCtx.lineTo(width * 0.3, height * 0.4);
            depthCtx.lineTo(width * 0.5, height * 0.5);
            depthCtx.lineTo(width * 0.7, height * 0.35);
            depthCtx.lineTo(width, height * 0.55);
            depthCtx.lineTo(width, height);
            depthCtx.lineTo(0, height);
            depthCtx.closePath();
            depthCtx.fill();
            
            // Foreground (close)
            depthCtx.fillStyle = '#ffffff';
            depthCtx.beginPath();
            depthCtx.ellipse(width * 0.3, height * 0.7, 100, 80, 0, 0, Math.PI * 2);
            depthCtx.fill();
            
            depthCtx.beginPath();
            depthCtx.ellipse(width * 0.7, height * 0.75, 120, 90, 0, 0, Math.PI * 2);
            depthCtx.fill();
            break;
    }
    
    return {
        image: imageCanvas.toDataURL('image/png'),
        depth: depthCanvas.toDataURL('image/png'),
        name: name
    };
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    elements.loadingOverlay.classList.add('hidden');
}

/**
 * Show loading overlay
 */
function showLoading() {
    elements.loadingOverlay.classList.remove('hidden');
}

// =============================================================================
// Start Application
// =============================================================================

// Wait for DOM to be ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Export for console debugging
window.RVRViewer = {
    getViewer: () => viewer,
    loadImages: (images) => viewer?.loadImages(images),
};
