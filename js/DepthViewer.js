/**
 * DepthViewer.js
 * Core 3D viewer class for rendering depth-displaced images with parallax effect.
 * 
 * Responsibilities:
 * - Three.js scene, camera, and renderer setup
 * - Mouse-driven parallax camera movement
 * - Stereoscopic dual-camera rendering
 * - Scene transition management
 */

import * as THREE from 'three';
import { ImageScene, RenderMode } from './ImageScene.js';

export class DepthViewer {
    constructor(container) {
        this.container = container;
        this.scenes = [];
        this.currentIndex = 0;
        this.isTransitioning = false;
        
        // Multi-layer support
        this.multiLayerGroups = [];  // Stores layer groups when in multi-layer mode
        this.isMultiLayerMode = false;
        this.currentRenderMode = RenderMode.BASIC;
        
        // Configuration
        this.config = {
            depthIntensity: 0.3,      // 0-1, controls Z displacement
            parallaxStrength: 0.5,    // 0-1, controls camera parallax (higher = more movement)
            meshResolution: 128,      // Subdivision count for mesh (lower = faster)
            stereoEnabled: false,     // Side-by-side stereo mode
            stereoSeparation: 0.064,  // Eye separation in meters
            autoDolly: false,         // Auto camera movement
            wireframe: false,         // Debug wireframe
            fov: 50,                  // Field of view
            nearPlane: 0.1,
            farPlane: 100,
            layerCount: 2,            // Number of layers for multi-layer mode
        };
        
        // Mouse tracking
        this.mouse = new THREE.Vector2(0, 0);
        this.targetMouse = new THREE.Vector2(0, 0);
        this.mouseSmoothing = 0.15;  // Higher = more responsive (was 0.08)
        
        // Camera base position (for parallax offset)
        this.cameraBasePosition = new THREE.Vector3(0, 0, 3);
        this.cameraTarget = new THREE.Vector3(0, 0, 0);
        
        // Raycaster for depth info
        this.raycaster = new THREE.Raycaster();
        
        // Auto dolly
        this.dollyTime = 0;
        this.dollySpeed = 0.3;
        this.dollyAmount = 0.15;
        
        // Clock for animations
        this.clock = new THREE.Clock();
        
        // Initialize
        this._initRenderer();
        this._initScene();
        this._initLights();
        this._initCamera();
        this._initEventListeners();
        
        // Start render loop
        this._animate();
    }
    
    /**
     * Initialize WebGL renderer
     */
    _initRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: false,
            powerPreference: 'high-performance',
        });
        
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        
        this.container.appendChild(this.renderer.domElement);
    }
    
    /**
     * Initialize Three.js scene
     */
    _initScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0f);
        
        // Optional fog for depth atmosphere
        this.scene.fog = new THREE.Fog(0x0a0a0f, 5, 15);
    }
    
    /**
     * Initialize lighting
     */
    _initLights() {
        // Ambient light for base illumination
        this.ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(this.ambientLight);
        
        // Main directional light
        this.mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
        this.mainLight.position.set(2, 3, 4);
        this.scene.add(this.mainLight);
        
        // Subtle fill light from opposite side
        this.fillLight = new THREE.DirectionalLight(0x8888ff, 0.3);
        this.fillLight.position.set(-2, -1, 2);
        this.scene.add(this.fillLight);
    }
    
    /**
     * Initialize camera(s)
     */
    _initCamera() {
        const aspect = window.innerWidth / window.innerHeight;
        
        // Main camera
        this.camera = new THREE.PerspectiveCamera(
            this.config.fov,
            aspect,
            this.config.nearPlane,
            this.config.farPlane
        );
        this.camera.position.copy(this.cameraBasePosition);
        this.camera.lookAt(this.cameraTarget);
        
        // Stereo cameras for side-by-side rendering
        this.leftCamera = new THREE.PerspectiveCamera(
            this.config.fov,
            aspect / 2,
            this.config.nearPlane,
            this.config.farPlane
        );
        
        this.rightCamera = new THREE.PerspectiveCamera(
            this.config.fov,
            aspect / 2,
            this.config.nearPlane,
            this.config.farPlane
        );
    }
    
    /**
     * Set up event listeners
     */
    _initEventListeners() {
        // Mouse movement for parallax
        window.addEventListener('mousemove', (e) => this._onMouseMove(e));
        
        // Window resize
        window.addEventListener('resize', () => this._onResize());
        
        // Mouse wheel for zoom
        window.addEventListener('wheel', (e) => this._onWheel(e), { passive: false });
        
        // Keyboard navigation
        window.addEventListener('keydown', (e) => this._onKeyDown(e));
    }
    
    /**
     * Handle mouse movement
     */
    _onMouseMove(event) {
        // Normalize mouse position to -1 to 1
        this.targetMouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        this.targetMouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    }
    
    /**
     * Handle window resize
     */
    _onResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        const aspect = width / height;
        
        // Update main camera
        this.camera.aspect = aspect;
        this.camera.updateProjectionMatrix();
        
        // Update stereo cameras
        this.leftCamera.aspect = aspect / 2;
        this.leftCamera.updateProjectionMatrix();
        this.rightCamera.aspect = aspect / 2;
        this.rightCamera.updateProjectionMatrix();
        
        // Update renderer
        this.renderer.setSize(width, height);
    }
    
    /**
     * Handle mouse wheel for zoom
     */
    _onWheel(event) {
        event.preventDefault();
        
        const zoomSpeed = 0.001;
        const delta = event.deltaY * zoomSpeed;
        
        // Adjust camera base position Z
        this.cameraBasePosition.z = THREE.MathUtils.clamp(
            this.cameraBasePosition.z + delta,
            1.5, // Min zoom (close)
            6    // Max zoom (far)
        );
    }
    
    /**
     * Handle keyboard input
     */
    _onKeyDown(event) {
        switch (event.key) {
            case 'ArrowLeft':
                this.previousImage();
                break;
            case 'ArrowRight':
                this.nextImage();
                break;
            case 'r':
            case 'R':
                this.resetCamera();
                break;
        }
    }
    
    /**
     * Update camera based on mouse position (parallax effect)
     */
    _updateCamera(deltaTime) {
        // Smooth mouse interpolation
        this.mouse.x += (this.targetMouse.x - this.mouse.x) * this.mouseSmoothing;
        this.mouse.y += (this.targetMouse.y - this.mouse.y) * this.mouseSmoothing;
        
        // Calculate parallax offset (multiply by 1.5 for more dramatic effect)
        const parallaxX = this.mouse.x * this.config.parallaxStrength * 1.5;
        const parallaxY = this.mouse.y * this.config.parallaxStrength * 0.8;
        
        // Auto dolly movement
        let dollyOffset = 0;
        if (this.config.autoDolly) {
            this.dollyTime += deltaTime * this.dollySpeed;
            dollyOffset = Math.sin(this.dollyTime) * this.dollyAmount;
        }
        
        // Update camera position
        this.camera.position.x = this.cameraBasePosition.x + parallaxX + dollyOffset;
        this.camera.position.y = this.cameraBasePosition.y + parallaxY;
        this.camera.position.z = this.cameraBasePosition.z;
        
        // Camera always looks at center
        this.camera.lookAt(this.cameraTarget);
        
        // Update stereo cameras if enabled
        if (this.config.stereoEnabled) {
            const halfSep = this.config.stereoSeparation / 2;
            
            this.leftCamera.position.copy(this.camera.position);
            this.leftCamera.position.x -= halfSep;
            this.leftCamera.lookAt(this.cameraTarget);
            
            this.rightCamera.position.copy(this.camera.position);
            this.rightCamera.position.x += halfSep;
            this.rightCamera.lookAt(this.cameraTarget);
        }
    }
    
    /**
     * Update raycaster for depth info display
     */
    _updateDepthInfo() {
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        const currentScene = this.scenes[this.currentIndex];
        if (currentScene && currentScene.mesh) {
            const intersects = this.raycaster.intersectObject(currentScene.mesh);
            
            if (intersects.length > 0) {
                const point = intersects[0].point;
                const depthValue = point.z.toFixed(3);
                
                // Dispatch event for UI update
                window.dispatchEvent(new CustomEvent('depthUpdate', {
                    detail: { depth: depthValue }
                }));
            }
        }
    }
    
    /**
     * Main render loop
     */
    _animate() {
        requestAnimationFrame(() => this._animate());
        
        const deltaTime = this.clock.getDelta();
        
        // Update camera parallax
        this._updateCamera(deltaTime);
        
        // Update depth info
        this._updateDepthInfo();
        
        // Update light direction to follow mouse (for "lit" mode)
        this.updateLightFromMouse();
        
        // Update active scenes
        this.scenes.forEach((scene, index) => {
            if (scene.mesh) {
                scene.update(deltaTime);
            }
        });
        
        // Render
        if (this.config.stereoEnabled) {
            this._renderStereo();
        } else {
            this.renderer.render(this.scene, this.camera);
        }
    }
    
    /**
     * Render stereoscopic side-by-side view
     */
    _renderStereo() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        const halfWidth = width / 2;
        
        this.renderer.setScissorTest(true);
        
        // Left eye
        this.renderer.setViewport(0, 0, halfWidth, height);
        this.renderer.setScissor(0, 0, halfWidth, height);
        this.renderer.render(this.scene, this.leftCamera);
        
        // Right eye
        this.renderer.setViewport(halfWidth, 0, halfWidth, height);
        this.renderer.setScissor(halfWidth, 0, halfWidth, height);
        this.renderer.render(this.scene, this.rightCamera);
        
        this.renderer.setScissorTest(false);
    }
    
    /**
     * Load images with depth maps
     * @param {Array} imageData - Array of {image: url, depth: url, mask?: url, name: string}
     */
    async loadImages(imageData) {
        // Store image data for multi-layer mode
        this.imageData = imageData;
        
        const loadPromises = imageData.map(async (data, index) => {
            const imageScene = new ImageScene(this.scene, this.config);
            await imageScene.load(data.image, data.depth, data.name);
            
            // Store mask URL if provided
            imageScene.maskUrl = data.mask || null;
            
            // Initially hide all but first
            if (index !== 0) {
                imageScene.setVisible(false);
            }
            
            return imageScene;
        });
        
        this.scenes = await Promise.all(loadPromises);
        
        // Dispatch loaded event
        window.dispatchEvent(new CustomEvent('imagesLoaded', {
            detail: { 
                count: this.scenes.length,
                images: imageData
            }
        }));
        
        return this.scenes;
    }
    
    /**
     * Navigate to next image
     */
    nextImage() {
        if (this.isTransitioning || this.scenes.length === 0) return;
        
        const nextIndex = (this.currentIndex + 1) % this.scenes.length;
        this.goToImage(nextIndex);
    }
    
    /**
     * Navigate to previous image
     */
    previousImage() {
        if (this.isTransitioning || this.scenes.length === 0) return;
        
        const prevIndex = (this.currentIndex - 1 + this.scenes.length) % this.scenes.length;
        this.goToImage(prevIndex);
    }
    
    /**
     * Navigate to specific image index
     * @param {number} index - Target image index
     */
    goToImage(index) {
        if (this.isTransitioning || index === this.currentIndex || index < 0 || index >= this.scenes.length) {
            return;
        }
        
        this.isTransitioning = true;
        
        if (this.isMultiLayerMode) {
            // Handle multi-layer transition
            this._setLayerGroupVisible(this.currentIndex, false);
            this._setLayerGroupVisible(index, true);
            
            this.currentIndex = index;
            this.isTransitioning = false;
            
            window.dispatchEvent(new CustomEvent('imageChanged', {
                detail: { index: this.currentIndex }
            }));
        } else {
            // Handle single-mesh transition
            const currentScene = this.scenes[this.currentIndex];
            const nextScene = this.scenes[index];
            
            // Start transition
            currentScene.fadeOut(500).then(() => {
                currentScene.setVisible(false);
            });
            
            nextScene.setVisible(true);
            nextScene.fadeIn(500).then(() => {
                this.currentIndex = index;
                this.isTransitioning = false;
                
                // Dispatch navigation event
                window.dispatchEvent(new CustomEvent('imageChanged', {
                    detail: { index: this.currentIndex }
                }));
            });
        }
    }
    
    /**
     * Reset camera to default position
     */
    resetCamera() {
        this.cameraBasePosition.set(0, 0, 3);
        this.targetMouse.set(0, 0);
        this.mouse.set(0, 0);
        this.dollyTime = 0;
    }
    
    /**
     * Update depth intensity
     * @param {number} value - Intensity value 0-1
     */
    setDepthIntensity(value) {
        this.config.depthIntensity = value;
        
        // Update single-mesh scenes
        this.scenes.forEach(scene => {
            scene.setDepthIntensity(value);
        });
        
        // Update multi-layer groups if in that mode
        if (this.isMultiLayerMode) {
            this._updateLayerDepths();
        }
    }
    
    /**
     * Update parallax strength
     * @param {number} value - Strength value 0-1
     */
    setParallaxStrength(value) {
        this.config.parallaxStrength = value;
    }
    
    /**
     * Update mesh resolution
     * @param {number} resolution - Subdivision count
     */
    setMeshResolution(resolution) {
        this.config.meshResolution = resolution;
        this.scenes.forEach(scene => {
            scene.rebuildMesh(resolution);
        });
    }
    
    /**
     * Toggle stereo mode
     * @param {boolean} enabled - Enable stereo rendering
     */
    setStereoMode(enabled) {
        this.config.stereoEnabled = enabled;
        document.body.classList.toggle('stereo-mode', enabled);
        this._onResize();
    }
    
    /**
     * Toggle auto dolly
     * @param {boolean} enabled - Enable auto camera movement
     */
    setAutoDolly(enabled) {
        this.config.autoDolly = enabled;
        if (!enabled) {
            this.dollyTime = 0;
        }
    }
    
    /**
     * Toggle wireframe mode
     * @param {boolean} enabled - Show wireframe
     */
    setWireframe(enabled) {
        this.config.wireframe = enabled;
        this.scenes.forEach(scene => {
            scene.setWireframe(enabled);
        });
    }
    
    /**
     * Get current configuration
     */
    getConfig() {
        return { ...this.config };
    }
    
    /**
     * Get current image index
     */
    getCurrentIndex() {
        return this.currentIndex;
    }
    
    /**
     * Get total image count
     */
    getImageCount() {
        return this.scenes.length;
    }
    
    /**
     * Set render mode for all scenes
     * @param {string} mode - 'basic', 'edgeAware', 'multiLayer', or 'parallax'
     */
    async setRenderMode(mode) {
        this.currentRenderMode = mode;
        
        if (mode === 'multiLayer') {
            // Switch to multi-layer rendering
            await this._enableMultiLayerMode();
        } else {
            // Switch back to single-mesh rendering
            this._disableMultiLayerMode();
            this.scenes.forEach(scene => {
                scene.setRenderMode(mode);
            });
        }
    }
    
    /**
     * Set number of layers for multi-layer mode
     * @param {number} count - 2, 3, or 4
     */
    async setLayerCount(count) {
        this.config.layerCount = count;
        
        if (this.isMultiLayerMode) {
            // Rebuild multi-layer groups with new count
            await this._rebuildMultiLayers();
        }
    }
    
    /**
     * Enable multi-layer rendering mode
     * Creates separate meshes for each depth layer to eliminate edge artifacts
     */
    async _enableMultiLayerMode() {
        if (this.isMultiLayerMode) return;
        
        console.log(`[MultiLayer] Enabling with ${this.config.layerCount} layers`);
        this.isMultiLayerMode = true;
        
        // Hide all single-mesh scenes
        this.scenes.forEach(scene => {
            scene.setVisible(false);
        });
        
        // Create multi-layer groups for each image
        await this._rebuildMultiLayers();
    }
    
    /**
     * Disable multi-layer mode, return to single-mesh rendering
     */
    _disableMultiLayerMode() {
        if (!this.isMultiLayerMode) return;
        
        console.log('[MultiLayer] Disabling');
        this.isMultiLayerMode = false;
        
        // Remove all layer meshes
        this.multiLayerGroups.forEach(group => {
            group.layers.forEach(layer => {
                this.scene.remove(layer.mesh);
                layer.mesh.geometry.dispose();
                layer.mesh.material.dispose();
            });
        });
        this.multiLayerGroups = [];
        
        // Show the appropriate single-mesh scene
        this.scenes.forEach((scene, index) => {
            scene.setVisible(index === this.currentIndex);
        });
    }
    
    /**
     * Rebuild multi-layer meshes for all images
     */
    async _rebuildMultiLayers() {
        // Clean up existing
        this.multiLayerGroups.forEach(group => {
            group.layers.forEach(layer => {
                this.scene.remove(layer.mesh);
                layer.mesh.geometry.dispose();
                layer.mesh.material.dispose();
            });
        });
        this.multiLayerGroups = [];
        
        // Create new layer groups (async for external masks)
        for (let sceneIndex = 0; sceneIndex < this.scenes.length; sceneIndex++) {
            const scene = this.scenes[sceneIndex];
            const isVisible = sceneIndex === this.currentIndex;
            const group = await this._createLayerGroup(scene, isVisible);
            this.multiLayerGroups.push(group);
        }
    }
    
    /**
     * Create a group of layer meshes from a single ImageScene
     * @param {ImageScene} scene - Source scene
     * @param {boolean} visible - Whether layers should be visible
     */
    async _createLayerGroup(scene, visible) {
        const layers = [];
        const numLayers = this.config.layerCount;
        
        // Get image dimensions
        const width = scene.imageTexture.image.width;
        const height = scene.imageTexture.image.height;
        const aspectRatio = width / height;
        
        let maskTexture;
        let useExternalMask = false;
        
        // Check if external SAM mask is available
        let uniqueMaskValues = null;
        if (scene.maskUrl) {
            try {
                console.log(`[MultiLayer] Loading external mask: ${scene.maskUrl}`);
                const textureLoader = new THREE.TextureLoader();
                maskTexture = await new Promise((resolve, reject) => {
                    textureLoader.load(scene.maskUrl, resolve, undefined, reject);
                });
                maskTexture.minFilter = THREE.NearestFilter;
                maskTexture.magFilter = THREE.NearestFilter;
                
                // Analyze mask to find unique layer values
                const maskCanvas = document.createElement('canvas');
                maskCanvas.width = maskTexture.image.width;
                maskCanvas.height = maskTexture.image.height;
                const maskCtx = maskCanvas.getContext('2d');
                maskCtx.drawImage(maskTexture.image, 0, 0);
                const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;
                
                // Find unique values in the mask
                const valueSet = new Set();
                for (let i = 0; i < maskData.length; i += 4) {
                    valueSet.add(maskData[i]); // Red channel
                }
                uniqueMaskValues = [...valueSet].sort((a, b) => a - b);
                
                useExternalMask = true;
                console.log(`[MultiLayer] Using SAM mask for ${scene.name} - found ${uniqueMaskValues.length} segments:`, uniqueMaskValues);
            } catch (e) {
                console.warn(`[MultiLayer] Failed to load mask, generating from depth`, e);
            }
        }
        
        // If no external mask, generate from depth
        if (!useExternalMask) {
            console.log(`[MultiLayer] Generating mask from depth for ${scene.name}`);
            
            // Create canvas to analyze depth
            const depthCanvas = document.createElement('canvas');
            depthCanvas.width = width;
            depthCanvas.height = height;
            const ctx = depthCanvas.getContext('2d');
            ctx.drawImage(scene.depthTexture.image, 0, 0);
            const depthData = ctx.getImageData(0, 0, width, height).data;
            
            // Create mask texture for layer separation
            const maskCanvas = document.createElement('canvas');
            maskCanvas.width = width;
            maskCanvas.height = height;
            const maskCtx = maskCanvas.getContext('2d');
            const maskImageData = maskCtx.createImageData(width, height);
            
            // Assign each pixel to a layer based on depth
            const edgeThreshold = 0.08;
            
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const idx = (y * width + x) * 4;
                    const depth = depthData[idx] / 255.0;
                    
                    let layerId = Math.floor(depth * numLayers);
                    layerId = Math.min(layerId, numLayers - 1);
                    
                    // Edge detection for better boundaries
                    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
                        const idxL = (y * width + (x - 1)) * 4;
                        const idxR = (y * width + (x + 1)) * 4;
                        const idxU = ((y - 1) * width + x) * 4;
                        const idxD = ((y + 1) * width + x) * 4;
                        
                        const depthL = depthData[idxL] / 255.0;
                        const depthR = depthData[idxR] / 255.0;
                        const depthU = depthData[idxU] / 255.0;
                        const depthD = depthData[idxD] / 255.0;
                        
                        const edgeH = Math.abs(depthL - depthR);
                        const edgeV = Math.abs(depthU - depthD);
                        
                        if (Math.max(edgeH, edgeV) > edgeThreshold) {
                            const maxNeighborDepth = Math.max(depthL, depthR, depthU, depthD);
                            layerId = Math.floor(maxNeighborDepth * numLayers);
                            layerId = Math.min(layerId, numLayers - 1);
                        }
                    }
                    
                    const maskValue = (layerId / (numLayers - 1)) * 255;
                    maskImageData.data[idx] = maskValue;
                    maskImageData.data[idx + 1] = maskValue;
                    maskImageData.data[idx + 2] = maskValue;
                    maskImageData.data[idx + 3] = 255;
                }
            }
            
            maskCtx.putImageData(maskImageData, 0, 0);
            
            maskTexture = new THREE.CanvasTexture(maskCanvas);
            maskTexture.minFilter = THREE.NearestFilter;
            maskTexture.magFilter = THREE.NearestFilter;
        }
        
        // Create mesh for each layer
        const planeHeight = 2;
        const planeWidth = planeHeight * aspectRatio;
        const resolution = this.config.meshResolution;
        
        // Determine layer values to create
        // If external mask: use actual unique values from the mask
        // If generated: use evenly spaced values based on numLayers
        const layerValues = useExternalMask && uniqueMaskValues 
            ? uniqueMaskValues.map(v => v / 255.0)  // Normalize to 0-1
            : Array.from({ length: numLayers }, (_, i) => i / Math.max(1, numLayers - 1));
        
        const actualNumLayers = layerValues.length;
        
        // Calculate tolerance based on actual gaps between layer values
        // For SAM masks with sequential integers (0, 1, 2...), gaps are 1/255 ≈ 0.0039
        // Use half the minimum gap to avoid overlap
        let tolerance;
        if (useExternalMask && layerValues.length > 1) {
            let minGap = 1.0;
            for (let i = 1; i < layerValues.length; i++) {
                const gap = layerValues[i] - layerValues[i-1];
                if (gap > 0 && gap < minGap) minGap = gap;
            }
            tolerance = minGap * 0.4;  // 40% of minimum gap
            console.log(`[MultiLayer] Mask gap tolerance: ${tolerance.toFixed(4)} (min gap: ${minGap.toFixed(4)})`);
        } else {
            tolerance = 0.5 / numLayers;
        }
        
        console.log(`[MultiLayer] Creating ${actualNumLayers} layer meshes with values:`, layerValues.map(v => v.toFixed(4)));
        console.log(`[MultiLayer] Using tolerance: ${tolerance.toFixed(4)}`);
        
        for (let i = 0; i < actualNumLayers; i++) {
            const maskTargetValue = layerValues[i];
            const avgDepth = maskTargetValue;  // Use mask value as depth ordering
            console.log(`[MultiLayer] Layer ${i}: targetValue=${maskTargetValue.toFixed(4)}, depth=${avgDepth.toFixed(4)}`);
            
            const geometry = new THREE.PlaneGeometry(
                planeWidth, planeHeight,
                resolution, Math.floor(resolution / aspectRatio)
            );
            
            // Shader that only shows pixels belonging to this layer
            const material = new THREE.ShaderMaterial({
                uniforms: {
                    imageMap: { value: scene.imageTexture },
                    depthMap: { value: scene.depthTexture },
                    maskMap: { value: maskTexture },
                    layerId: { value: maskTargetValue },
                    depthIntensity: { value: this.config.depthIntensity },
                    opacity: { value: visible ? 1.0 : 0.0 },
                    tolerance: { value: tolerance },
                },
                vertexShader: `
                    uniform sampler2D depthMap;
                    uniform float depthIntensity;
                    
                    varying vec2 vUv;
                    
                    void main() {
                        vUv = uv;
                        
                        float depth = texture2D(depthMap, uv).r;
                        
                        vec3 displaced = position;
                        displaced.z += (depth - 0.5) * depthIntensity;
                        
                        gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
                    }
                `,
                fragmentShader: `
                    uniform sampler2D imageMap;
                    uniform sampler2D maskMap;
                    uniform float layerId;
                    uniform float opacity;
                    uniform float tolerance;
                    
                    varying vec2 vUv;
                    
                    void main() {
                        float maskValue = texture2D(maskMap, vUv).r;
                        
                        // Only show pixels belonging to this layer
                        if (abs(maskValue - layerId) > tolerance) {
                            discard;
                        }
                        
                        vec4 texColor = texture2D(imageMap, vUv);
                        gl_FragColor = vec4(texColor.rgb, texColor.a * opacity);
                    }
                `,
                transparent: true,
                side: THREE.DoubleSide,
                depthWrite: true,
                depthTest: true,
            });
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.name = `${scene.name}_layer_${i}`;
            mesh.visible = visible;
            
            // Render order: background first, foreground last
            mesh.renderOrder = i;
            
            this.scene.add(mesh);
            
            layers.push({
                mesh,
                material,
                layerId: i,
                avgDepth,
                maskTexture
            });
        }
        
        return {
            sourceName: scene.name,
            layers,
            maskTexture
        };
    }
    
    /**
     * Update layer depth intensity
     */
    _updateLayerDepths() {
        this.multiLayerGroups.forEach(group => {
            group.layers.forEach(layer => {
                layer.material.uniforms.depthIntensity.value = this.config.depthIntensity;
            });
        });
    }
    
    /**
     * Set visibility for a multi-layer group
     */
    _setLayerGroupVisible(groupIndex, visible) {
        if (groupIndex >= 0 && groupIndex < this.multiLayerGroups.length) {
            const group = this.multiLayerGroups[groupIndex];
            group.layers.forEach(layer => {
                layer.mesh.visible = visible;
                layer.material.uniforms.opacity.value = visible ? 1.0 : 0.0;
            });
        }
    }
    
    /**
     * Set debug visualization mode
     * @param {string} mode - 'none', 'depth', 'heatmap', 'contours', 'edges', 'displacement', 'wireframe'
     */
    setDebugMode(mode) {
        this.scenes.forEach(scene => {
            scene.setDebugMode(mode);
        });
    }
    
    /**
     * Set edge softness for edge-aware mode
     * @param {number} value - 0-1
     */
    setEdgeSoftness(value) {
        this.scenes.forEach(scene => {
            scene.setEdgeSoftness(value);
        });
    }
    
    /**
     * Get diagnostics for current image
     * @returns {Object|null} Diagnostic information
     */
    getDiagnostics() {
        const currentScene = this.scenes[this.currentIndex];
        if (currentScene) {
            return currentScene.getDiagnostics();
        }
        return null;
    }
    
    /**
     * Set light direction for all scenes (used in "lit" mode)
     * Light follows mouse for dynamic effect
     * @param {number} x - X direction
     * @param {number} y - Y direction
     */
    setLightDirection(x, y) {
        this.scenes.forEach(scene => {
            scene.setLightDirection(x, y, 1.0);
        });
    }
    
    /**
     * Update light to follow mouse (call in render loop)
     */
    updateLightFromMouse() {
        // Light direction follows mouse for dynamic shading
        this.scenes.forEach(scene => {
            scene.setLightDirection(
                this.mouse.x * 0.5 + 0.5,
                this.mouse.y * 0.5 + 0.5,
                1.0
            );
        });
    }
    
    /**
     * Cleanup and dispose
     */
    dispose() {
        this.scenes.forEach(scene => scene.dispose());
        this.renderer.dispose();
        this.container.removeChild(this.renderer.domElement);
    }
}
