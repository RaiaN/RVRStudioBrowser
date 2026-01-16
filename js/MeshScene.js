/**
 * MeshScene.js
 * Manages 3D mesh loading and rendering for BLADE-derived meshes.
 * 
 * Features:
 * - glTF/GLB mesh loading with texture support
 * - Interactive camera controls (orbit, pan, zoom)
 * - Enhanced lighting for 3D perception
 * - Micro-parallax effects for enhanced depth
 * - Post-processing options (SSAO, rim lighting)
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/**
 * Mesh display modes
 */
export const MeshDisplayMode = {
    TEXTURED: 'textured',       // Full texture rendering
    WIREFRAME: 'wireframe',     // Wireframe overlay
    NORMAL: 'normal',           // Normal map visualization
    DEPTH: 'depth',             // Depth visualization
    VERTEX_COLORS: 'vertexColors', // Vertex color display
};

/**
 * Lighting presets
 */
export const LightingPreset = {
    STUDIO: 'studio',           // 3-point studio lighting
    NATURAL: 'natural',         // Soft natural light
    DRAMATIC: 'dramatic',       // High contrast dramatic
    RIM: 'rim',                 // Strong rim/backlight
};

export class MeshScene {
    constructor(parentScene, camera, config = {}) {
        this.parentScene = parentScene;
        this.camera = camera;
        
        // Configuration
        this.config = {
            autoRotate: false,
            autoRotateSpeed: 0.5,
            enableShadows: true,
            enableSSAO: false,
            rimLightIntensity: 0.3,
            ambientOcclusion: 0.5,
            ...config
        };
        
        // State
        this.mesh = null;
        this.meshGroup = null;
        this.boundingBox = null;
        this.originalMaterials = [];
        this.displayMode = MeshDisplayMode.TEXTURED;
        this.lightingPreset = LightingPreset.STUDIO;
        
        // Animation
        this.animationMixer = null;
        this.animations = [];
        this.currentAnimation = null;
        
        // Lights (managed by this scene)
        this.lights = {
            ambient: null,
            key: null,
            fill: null,
            back: null,
            rim: null,
        };
        
        // Loaders
        this._initLoaders();
        
        // Create light rig
        this._createLightRig();
    }
    
    /**
     * Initialize asset loaders
     */
    _initLoaders() {
        // GLTF Loader with DRACO compression support
        this.gltfLoader = new GLTFLoader();
        
        // DRACO loader for compressed meshes
        const dracoLoader = new DRACOLoader();
        dracoLoader.setDecoderPath('https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/libs/draco/');
        this.gltfLoader.setDRACOLoader(dracoLoader);
        
        // Texture loader
        this.textureLoader = new THREE.TextureLoader();
    }
    
    /**
     * Create professional lighting rig
     */
    _createLightRig() {
        // Ambient light
        this.lights.ambient = new THREE.AmbientLight(0xffffff, 0.4);
        this.parentScene.add(this.lights.ambient);
        
        // Key light (main light)
        this.lights.key = new THREE.DirectionalLight(0xffffff, 1.0);
        this.lights.key.position.set(3, 4, 2);
        this.lights.key.castShadow = this.config.enableShadows;
        this.lights.key.shadow.mapSize.width = 2048;
        this.lights.key.shadow.mapSize.height = 2048;
        this.lights.key.shadow.camera.near = 0.1;
        this.lights.key.shadow.camera.far = 20;
        this.lights.key.shadow.bias = -0.0001;
        this.parentScene.add(this.lights.key);
        
        // Fill light (softer, opposite side)
        this.lights.fill = new THREE.DirectionalLight(0x8899ff, 0.4);
        this.lights.fill.position.set(-2, 1, 2);
        this.parentScene.add(this.lights.fill);
        
        // Back light
        this.lights.back = new THREE.DirectionalLight(0xffffee, 0.3);
        this.lights.back.position.set(0, 2, -3);
        this.parentScene.add(this.lights.back);
        
        // Rim light (for edge highlighting)
        this.lights.rim = new THREE.SpotLight(0xffffff, this.config.rimLightIntensity);
        this.lights.rim.position.set(-3, 3, -2);
        this.lights.rim.angle = Math.PI / 6;
        this.lights.rim.penumbra = 0.5;
        this.parentScene.add(this.lights.rim);
    }
    
    /**
     * Apply lighting preset
     * @param {string} preset - Lighting preset name
     */
    setLightingPreset(preset) {
        this.lightingPreset = preset;
        
        switch (preset) {
            case LightingPreset.STUDIO:
                this.lights.ambient.intensity = 0.4;
                this.lights.key.intensity = 1.0;
                this.lights.key.position.set(3, 4, 2);
                this.lights.fill.intensity = 0.4;
                this.lights.fill.color.setHex(0x8899ff);
                this.lights.back.intensity = 0.3;
                this.lights.rim.intensity = 0.3;
                break;
                
            case LightingPreset.NATURAL:
                this.lights.ambient.intensity = 0.6;
                this.lights.key.intensity = 0.8;
                this.lights.key.position.set(2, 5, 3);
                this.lights.fill.intensity = 0.3;
                this.lights.fill.color.setHex(0xaabbff);
                this.lights.back.intensity = 0.2;
                this.lights.rim.intensity = 0.1;
                break;
                
            case LightingPreset.DRAMATIC:
                this.lights.ambient.intensity = 0.15;
                this.lights.key.intensity = 1.5;
                this.lights.key.position.set(4, 3, 1);
                this.lights.fill.intensity = 0.1;
                this.lights.fill.color.setHex(0x445566);
                this.lights.back.intensity = 0.0;
                this.lights.rim.intensity = 0.8;
                break;
                
            case LightingPreset.RIM:
                this.lights.ambient.intensity = 0.2;
                this.lights.key.intensity = 0.6;
                this.lights.key.position.set(0, 4, 3);
                this.lights.fill.intensity = 0.2;
                this.lights.fill.color.setHex(0x667788);
                this.lights.back.intensity = 0.8;
                this.lights.rim.intensity = 1.2;
                break;
        }
        
        console.log(`[MeshScene] Lighting preset: ${preset}`);
    }
    
    /**
     * Load glTF/GLB mesh
     * @param {string} url - URL to mesh file
     * @param {string} name - Display name
     * @returns {Promise<THREE.Group>}
     */
    async load(url, name = '') {
        console.log(`[MeshScene] Loading mesh: ${url}`);
        
        return new Promise((resolve, reject) => {
            this.gltfLoader.load(
                url,
                (gltf) => {
                    // Store the entire scene
                    this.meshGroup = gltf.scene;
                    this.meshGroup.name = name || url.split('/').pop();
                    
                    // Process meshes
                    this._processMeshes(this.meshGroup);
                    
                    // Calculate bounding box and center
                    this.boundingBox = new THREE.Box3().setFromObject(this.meshGroup);
                    const center = this.boundingBox.getCenter(new THREE.Vector3());
                    const size = this.boundingBox.getSize(new THREE.Vector3());
                    
                    // Center the mesh
                    this.meshGroup.position.sub(center);
                    
                    // Scale to fit in view (max dimension = 2 units)
                    const maxDim = Math.max(size.x, size.y, size.z);
                    if (maxDim > 0) {
                        const scale = 2.0 / maxDim;
                        this.meshGroup.scale.setScalar(scale);
                    }
                    
                    // Add to scene
                    this.parentScene.add(this.meshGroup);
                    
                    // Handle animations
                    if (gltf.animations && gltf.animations.length > 0) {
                        this.animations = gltf.animations;
                        this.animationMixer = new THREE.AnimationMixer(this.meshGroup);
                        console.log(`[MeshScene] Found ${this.animations.length} animations`);
                    }
                    
                    // Update light positions based on mesh size
                    this._updateLightPositions(size);
                    
                    console.log(`[MeshScene] ✓ Loaded: ${name}`);
                    console.log(`  Bounds: ${size.x.toFixed(2)} x ${size.y.toFixed(2)} x ${size.z.toFixed(2)}`);
                    
                    resolve(this.meshGroup);
                },
                (progress) => {
                    if (progress.total > 0) {
                        const percent = (progress.loaded / progress.total * 100).toFixed(0);
                        console.log(`[MeshScene] Loading: ${percent}%`);
                    }
                },
                (error) => {
                    console.error('[MeshScene] Load error:', error);
                    reject(error);
                }
            );
        });
    }
    
    /**
     * Process all meshes in the group
     * @param {THREE.Object3D} object
     */
    _processMeshes(object) {
        object.traverse((child) => {
            if (child.isMesh) {
                // Store original material
                this.originalMaterials.push({
                    mesh: child,
                    material: child.material.clone()
                });
                
                // Enable shadows
                child.castShadow = this.config.enableShadows;
                child.receiveShadow = this.config.enableShadows;
                
                // Enhance material for better visuals
                if (child.material) {
                    this._enhanceMaterial(child.material);
                }
            }
        });
    }
    
    /**
     * Enhance material properties
     * @param {THREE.Material} material
     */
    _enhanceMaterial(material) {
        if (material.isMeshStandardMaterial || material.isMeshPhysicalMaterial) {
            // Ensure proper settings for PBR
            material.needsUpdate = true;
            
            // Adjust roughness/metalness if too extreme
            if (material.roughness === undefined) material.roughness = 0.7;
            if (material.metalness === undefined) material.metalness = 0.0;
            
            // Enable environment mapping if available
            if (this.parentScene.environment) {
                material.envMapIntensity = 0.5;
            }
        }
    }
    
    /**
     * Update light positions based on mesh size
     * @param {THREE.Vector3} size
     */
    _updateLightPositions(size) {
        const maxDim = Math.max(size.x, size.y, size.z);
        const lightDistance = maxDim * 2;
        
        this.lights.key.position.set(lightDistance, lightDistance * 1.3, lightDistance * 0.7);
        this.lights.fill.position.set(-lightDistance * 0.7, lightDistance * 0.3, lightDistance * 0.7);
        this.lights.back.position.set(0, lightDistance * 0.7, -lightDistance);
        this.lights.rim.position.set(-lightDistance, lightDistance, -lightDistance * 0.7);
        
        // Update shadow camera
        if (this.lights.key.shadow) {
            const shadowSize = maxDim * 2;
            this.lights.key.shadow.camera.left = -shadowSize;
            this.lights.key.shadow.camera.right = shadowSize;
            this.lights.key.shadow.camera.top = shadowSize;
            this.lights.key.shadow.camera.bottom = -shadowSize;
            this.lights.key.shadow.camera.updateProjectionMatrix();
        }
    }
    
    /**
     * Set display mode
     * @param {string} mode - Display mode
     */
    setDisplayMode(mode) {
        this.displayMode = mode;
        
        if (!this.meshGroup) return;
        
        this.meshGroup.traverse((child) => {
            if (!child.isMesh) return;
            
            // Find original material
            const original = this.originalMaterials.find(m => m.mesh === child);
            
            switch (mode) {
                case MeshDisplayMode.TEXTURED:
                    if (original) {
                        child.material = original.material.clone();
                    }
                    child.material.wireframe = false;
                    break;
                    
                case MeshDisplayMode.WIREFRAME:
                    child.material = new THREE.MeshBasicMaterial({
                        color: 0x00ffff,
                        wireframe: true,
                        transparent: true,
                        opacity: 0.8
                    });
                    break;
                    
                case MeshDisplayMode.NORMAL:
                    child.material = new THREE.MeshNormalMaterial({
                        flatShading: false
                    });
                    break;
                    
                case MeshDisplayMode.DEPTH:
                    child.material = new THREE.MeshDepthMaterial({
                        depthPacking: THREE.BasicDepthPacking
                    });
                    break;
                    
                case MeshDisplayMode.VERTEX_COLORS:
                    child.material = new THREE.MeshBasicMaterial({
                        vertexColors: true
                    });
                    break;
            }
        });
        
        console.log(`[MeshScene] Display mode: ${mode}`);
    }
    
    /**
     * Play animation by index or name
     * @param {number|string} animation - Animation index or name
     */
    playAnimation(animation) {
        if (!this.animationMixer || this.animations.length === 0) {
            console.warn('[MeshScene] No animations available');
            return;
        }
        
        // Stop current animation
        if (this.currentAnimation) {
            this.currentAnimation.stop();
        }
        
        // Find animation
        let clip;
        if (typeof animation === 'number') {
            clip = this.animations[animation];
        } else {
            clip = this.animations.find(a => a.name === animation);
        }
        
        if (!clip) {
            console.warn(`[MeshScene] Animation not found: ${animation}`);
            return;
        }
        
        // Play animation
        this.currentAnimation = this.animationMixer.clipAction(clip);
        this.currentAnimation.play();
        
        console.log(`[MeshScene] Playing animation: ${clip.name}`);
    }
    
    /**
     * Stop current animation
     */
    stopAnimation() {
        if (this.currentAnimation) {
            this.currentAnimation.stop();
            this.currentAnimation = null;
        }
    }
    
    /**
     * Set auto-rotation
     * @param {boolean} enabled
     * @param {number} speed - Rotation speed
     */
    setAutoRotate(enabled, speed = 0.5) {
        this.config.autoRotate = enabled;
        this.config.autoRotateSpeed = speed;
    }
    
    /**
     * Set rim light intensity
     * @param {number} intensity - 0-2
     */
    setRimLightIntensity(intensity) {
        this.config.rimLightIntensity = intensity;
        if (this.lights.rim) {
            this.lights.rim.intensity = intensity;
        }
    }
    
    /**
     * Set visibility
     * @param {boolean} visible
     */
    setVisible(visible) {
        if (this.meshGroup) {
            this.meshGroup.visible = visible;
        }
        
        // Also toggle lights
        Object.values(this.lights).forEach(light => {
            if (light) light.visible = visible;
        });
    }
    
    /**
     * Get mesh statistics
     * @returns {Object}
     */
    getStats() {
        if (!this.meshGroup) return null;
        
        let vertexCount = 0;
        let triangleCount = 0;
        let meshCount = 0;
        
        this.meshGroup.traverse((child) => {
            if (child.isMesh && child.geometry) {
                meshCount++;
                
                const geo = child.geometry;
                if (geo.index) {
                    triangleCount += geo.index.count / 3;
                } else if (geo.attributes.position) {
                    triangleCount += geo.attributes.position.count / 3;
                }
                
                if (geo.attributes.position) {
                    vertexCount += geo.attributes.position.count;
                }
            }
        });
        
        return {
            meshCount,
            vertexCount,
            triangleCount,
            hasAnimations: this.animations.length > 0,
            animationCount: this.animations.length,
            bounds: this.boundingBox ? {
                min: this.boundingBox.min.toArray(),
                max: this.boundingBox.max.toArray()
            } : null
        };
    }
    
    /**
     * Update (called each frame)
     * @param {number} deltaTime
     */
    update(deltaTime) {
        // Update animation mixer
        if (this.animationMixer) {
            this.animationMixer.update(deltaTime);
        }
        
        // Auto-rotate
        if (this.config.autoRotate && this.meshGroup) {
            this.meshGroup.rotation.y += deltaTime * this.config.autoRotateSpeed;
        }
    }
    
    /**
     * Cleanup
     */
    dispose() {
        // Remove mesh from scene
        if (this.meshGroup) {
            this.parentScene.remove(this.meshGroup);
            
            // Dispose geometries and materials
            this.meshGroup.traverse((child) => {
                if (child.isMesh) {
                    if (child.geometry) child.geometry.dispose();
                    if (child.material) {
                        if (Array.isArray(child.material)) {
                            child.material.forEach(m => m.dispose());
                        } else {
                            child.material.dispose();
                        }
                    }
                }
            });
        }
        
        // Remove lights
        Object.values(this.lights).forEach(light => {
            if (light) {
                this.parentScene.remove(light);
                if (light.dispose) light.dispose();
            }
        });
        
        // Clear original materials
        this.originalMaterials = [];
        
        // Stop animation
        if (this.animationMixer) {
            this.animationMixer.stopAllAction();
        }
    }
}


/**
 * MeshViewer - Standalone viewer for 3D meshes
 * Provides complete setup including scene, camera, controls
 */
export class MeshViewer {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            backgroundColor: 0x0a0a0f,
            enableControls: true,
            enableGrid: false,
            enableAxes: false,
            ...options
        };
        
        this.meshScenes = [];
        this.currentIndex = 0;
        
        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        
        // Animation
        this.clock = new THREE.Clock();
        this.isRunning = false;
        
        // Initialize
        this._init();
    }
    
    /**
     * Initialize Three.js scene
     */
    _init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.options.backgroundColor);
        
        // Camera
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 100);
        this.camera.position.set(0, 0, 4);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            alpha: false,
            powerPreference: 'high-performance'
        });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        this.container.appendChild(this.renderer.domElement);
        
        // Controls
        if (this.options.enableControls) {
            this.controls = new OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true;
            this.controls.dampingFactor = 0.05;
            this.controls.screenSpacePanning = true;
            this.controls.minDistance = 1;
            this.controls.maxDistance = 20;
            this.controls.maxPolarAngle = Math.PI * 0.9;
        }
        
        // Optional helpers
        if (this.options.enableGrid) {
            const grid = new THREE.GridHelper(10, 20, 0x444444, 0x222222);
            this.scene.add(grid);
        }
        
        if (this.options.enableAxes) {
            const axes = new THREE.AxesHelper(2);
            this.scene.add(axes);
        }
        
        // Event listeners
        window.addEventListener('resize', () => this._onResize());
        
        // Start render loop
        this._animate();
    }
    
    /**
     * Handle window resize
     */
    _onResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
    }
    
    /**
     * Animation loop
     */
    _animate() {
        this.isRunning = true;
        
        const animate = () => {
            if (!this.isRunning) return;
            
            requestAnimationFrame(animate);
            
            const deltaTime = this.clock.getDelta();
            
            // Update controls
            if (this.controls) {
                this.controls.update();
            }
            
            // Update mesh scenes
            this.meshScenes.forEach(scene => scene.update(deltaTime));
            
            // Render
            this.renderer.render(this.scene, this.camera);
        };
        
        animate();
    }
    
    /**
     * Load mesh(es) from URLs
     * @param {Array<{url: string, name: string}>} meshData
     */
    async loadMeshes(meshData) {
        // Clear existing
        this.meshScenes.forEach(scene => scene.dispose());
        this.meshScenes = [];
        
        // Load each mesh
        for (let i = 0; i < meshData.length; i++) {
            const data = meshData[i];
            const meshScene = new MeshScene(this.scene, this.camera);
            
            try {
                await meshScene.load(data.url, data.name);
                
                // Hide all but first
                if (i !== 0) {
                    meshScene.setVisible(false);
                }
                
                this.meshScenes.push(meshScene);
            } catch (error) {
                console.error(`Failed to load mesh: ${data.url}`, error);
            }
        }
        
        // Dispatch loaded event
        window.dispatchEvent(new CustomEvent('meshesLoaded', {
            detail: {
                count: this.meshScenes.length,
                meshes: meshData
            }
        }));
        
        return this.meshScenes;
    }
    
    /**
     * Navigate to mesh by index
     * @param {number} index
     */
    goToMesh(index) {
        if (index < 0 || index >= this.meshScenes.length) return;
        
        // Hide current
        if (this.meshScenes[this.currentIndex]) {
            this.meshScenes[this.currentIndex].setVisible(false);
        }
        
        // Show new
        this.currentIndex = index;
        if (this.meshScenes[this.currentIndex]) {
            this.meshScenes[this.currentIndex].setVisible(true);
        }
        
        // Reset camera
        this.resetCamera();
        
        // Dispatch event
        window.dispatchEvent(new CustomEvent('meshChanged', {
            detail: { index }
        }));
    }
    
    /**
     * Reset camera to default position
     */
    resetCamera() {
        this.camera.position.set(0, 0, 4);
        this.camera.lookAt(0, 0, 0);
        
        if (this.controls) {
            this.controls.reset();
        }
    }
    
    /**
     * Set lighting preset for current mesh
     * @param {string} preset
     */
    setLightingPreset(preset) {
        if (this.meshScenes[this.currentIndex]) {
            this.meshScenes[this.currentIndex].setLightingPreset(preset);
        }
    }
    
    /**
     * Set display mode for current mesh
     * @param {string} mode
     */
    setDisplayMode(mode) {
        if (this.meshScenes[this.currentIndex]) {
            this.meshScenes[this.currentIndex].setDisplayMode(mode);
        }
    }
    
    /**
     * Get stats for current mesh
     */
    getStats() {
        if (this.meshScenes[this.currentIndex]) {
            return this.meshScenes[this.currentIndex].getStats();
        }
        return null;
    }
    
    /**
     * Cleanup
     */
    dispose() {
        this.isRunning = false;
        
        this.meshScenes.forEach(scene => scene.dispose());
        this.meshScenes = [];
        
        if (this.controls) {
            this.controls.dispose();
        }
        
        this.renderer.dispose();
        this.container.removeChild(this.renderer.domElement);
    }
}
