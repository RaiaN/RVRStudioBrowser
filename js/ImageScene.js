/**
 * ImageScene.js
 * Manages individual image + depth map pairs as depth-displaced 3D meshes.
 * 
 * Features:
 * - Basic vertex displacement mode
 * - Debug visualization tools (depth map, edges, wireframe)
 */

import * as THREE from 'three';
import { Shaders } from './shaders/DepthDisplacement.glsl.js';

// Rendering modes
export const RenderMode = {
    MULTI_LAYER: 'multiLayer', // Separates into depth layers
    HYBRID: 'hybrid',          // ⭐ Depth-only parallax - zero artifacts!
};

// Debug visualization modes
export const DebugMode = {
    NONE: 'none',
    DEPTH: 'depth',           // Show depth map
    DEPTH_HEATMAP: 'heatmap', // Depth as heatmap
    DEPTH_CONTOURS: 'contours', // Depth with contour lines
    EDGES: 'edges',           // Highlight depth discontinuities (artifact sources)
    DISPLACEMENT: 'displacement', // Show displacement magnitude
    WIREFRAME: 'wireframe',   // Wireframe with depth coloring
};

export class ImageScene {
    constructor(parentScene, config) {
        this.parentScene = parentScene;
        this.config = config;
        
        this.mesh = null;
        this.material = null;
        this.geometry = null;
        
        this.imageTexture = null;
        this.depthTexture = null;
        this.maskTexture = null;        // Foreground mask (white = subject)
        this.inpaintedTexture = null;   // Inpainted version for disoccluded regions
        
        this.name = '';
        this.aspectRatio = 1;
        
        // Animation state
        this.opacity = 1;
        this.targetOpacity = 1;
        this.isVisible = true;
        
        // Shader uniforms
        this.uniforms = null;
        
        // Current modes
        this.renderMode = RenderMode.HYBRID;  // Default to hybrid (depth parallax)
        this.debugMode = DebugMode.NONE;
        
        // Point cloud settings
        this.pointCloudDensity = 1;  // 1 = full resolution, 2 = half, etc.
    }
    
    /**
     * Load image and depth map
     */
    async load(imageUrl, depthUrl, name = '') {
        this.name = name;
        
        const textureLoader = new THREE.TextureLoader();
        
        // Load both textures in parallel
        const [imageTexture, depthTexture] = await Promise.all([
            this._loadTexture(textureLoader, imageUrl),
            this._loadTexture(textureLoader, depthUrl)
        ]);
        
        this.imageTexture = imageTexture;
        this.depthTexture = depthTexture;
        
        // Configure texture settings
        this.imageTexture.colorSpace = THREE.SRGBColorSpace;
        this.imageTexture.minFilter = THREE.LinearFilter;
        this.imageTexture.magFilter = THREE.LinearFilter;
        
        // Critical: Linear filtering for smooth depth sampling
        this.depthTexture.minFilter = THREE.LinearFilter;
        this.depthTexture.magFilter = THREE.LinearFilter;
        this.depthTexture.wrapS = THREE.ClampToEdgeWrapping;
        this.depthTexture.wrapT = THREE.ClampToEdgeWrapping;
        
        // Calculate aspect ratio
        this.aspectRatio = this.imageTexture.image.width / this.imageTexture.image.height;
        
        // Create the mesh
        this._createMesh();
        
        return this;
    }
    
    /**
     * Load a texture with promise wrapper
     */
    _loadTexture(loader, url) {
        return new Promise((resolve, reject) => {
            loader.load(
                url,
                (texture) => resolve(texture),
                undefined,
                (error) => reject(error)
            );
        });
    }
    
    /**
     * Create all shader uniforms
     */
    _createUniforms() {
        const texWidth = this.depthTexture.image.width;
        const texHeight = this.depthTexture.image.height;
        
        return {
            imageMap: { value: this.imageTexture },
            depthMap: { value: this.depthTexture },
            inpaintedMap: { value: this.inpaintedTexture || this.imageTexture },  // Fallback to original
            depthIntensity: { value: this.config.depthIntensity },
            depthBias: { value: 0.5 },
            opacity: { value: 1.0 },
            
            // For edge-aware shader
            texelSize: { value: new THREE.Vector2(1.0 / texWidth, 1.0 / texHeight) },
            edgeSoftness: { value: 0.15 },
            
            // For edge fading (artifact hiding)
            edgeThreshold: { value: 0.06 },    // Depth difference that triggers fade
            edgeFadeWidth: { value: 0.12 },    // How gradual the fade is
            softEdges: { value: 1.0 },         // Enable soft edges (1.0 = on, 0.0 = off)
            
            // For depth debug
            colorMode: { value: 0 },
            
            // For parallax
            numLayers: { value: 32 },
            
            // Resolution
            resolution: { value: new THREE.Vector2(texWidth, texHeight) },
            
            // For Point Cloud mode
            pointSize: { value: this.config.pointSize || 2.0 },
            pointSizeAttenuation: { value: this.config.pointSizeAttenuation || 0.5 },
            pointSoftness: { value: this.config.pointSoftness || 0.3 },
            useInpainted: { value: this.inpaintedTexture ? 1.0 : 0.0 },
            debugMode: { value: 0 }
        };
    }
    
    /**
     * Get shader config for current mode
     */
    _getShaderConfig() {
        if (this.debugMode !== DebugMode.NONE) {
            switch (this.debugMode) {
                case DebugMode.DEPTH:
                    return { ...Shaders.debug.depth, colorMode: 0 };
                case DebugMode.DEPTH_HEATMAP:
                    return { ...Shaders.debug.depth, colorMode: 1 };
                case DebugMode.DEPTH_CONTOURS:
                    return { ...Shaders.debug.depth, colorMode: 2 };
                case DebugMode.EDGES:
                    return Shaders.debug.edges;
                case DebugMode.DISPLACEMENT:
                    return Shaders.debug.displacement;
                case DebugMode.WIREFRAME:
                    return Shaders.debug.wireframe;
            }
        }
        
        return Shaders[this.renderMode] || Shaders.multiLayer;
    }
    
    /**
     * Create the mesh with current shader mode
     */
    _createMesh() {
        // Check for special render modes
        if (this.renderMode === RenderMode.HYBRID) {
            this._createHybridMesh();
            return;
        }
        
        const resolution = this.config.meshResolution;
        
        // Create subdivided plane
        const planeHeight = 2;
        const planeWidth = planeHeight * this.aspectRatio;
        
        // Use resolution as-is (don't multiply - causes slowdown)
        const segmentsX = resolution;
        const segmentsY = Math.floor(resolution / this.aspectRatio);
        
        this.geometry = new THREE.PlaneGeometry(
            planeWidth,
            planeHeight,
            segmentsX,
            segmentsY
        );
        
        // Create uniforms
        this.uniforms = this._createUniforms();
        
        // Get shader for current mode
        const shaderConfig = this._getShaderConfig();
        
        // Set colorMode if applicable
        if (shaderConfig.colorMode !== undefined) {
            this.uniforms.colorMode.value = shaderConfig.colorMode;
        }
        
        this.material = new THREE.ShaderMaterial({
            uniforms: this.uniforms,
            vertexShader: shaderConfig.vertex,
            fragmentShader: shaderConfig.fragment,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: true,
            wireframe: this.debugMode === DebugMode.WIREFRAME,
        });
        
        // Create mesh
        this.mesh = new THREE.Mesh(this.geometry, this.material);
        this.mesh.name = this.name;
        
        // Add to parent scene
        this.parentScene.add(this.mesh);
    }
    
    /**
     * Create point cloud geometry - grid of points that sample image/depth
     * Simple approach: create a grid, each vertex has UV to sample textures
     */
    _createPointCloud() {
        // Grid resolution based on density setting
        // density 1 = full (400 base), 2 = half (200), 3 = third (~133), 4 = quarter (100)
        const density = this.pointCloudDensity;
        const baseResolution = 400;  // High resolution for quality
        const gridResX = Math.floor(baseResolution / density);
        const gridResY = Math.floor(gridResX / this.aspectRatio);
        const totalPoints = gridResX * gridResY;
        
        console.log(`[ImageScene] Creating point cloud: ${totalPoints.toLocaleString()} points (${gridResX}x${gridResY} grid)`);
        
        // Grid size in world units (matches plane size)
        const planeHeight = 2.0;
        const planeWidth = planeHeight * this.aspectRatio;
        
        // Create arrays
        const positions = new Float32Array(totalPoints * 3);
        const uvs = new Float32Array(totalPoints * 2);
        
        let idx = 0;
        for (let iy = 0; iy < gridResY; iy++) {
            for (let ix = 0; ix < gridResX; ix++) {
                // UV coordinates [0, 1]
                const u = ix / (gridResX - 1);
                const v = iy / (gridResY - 1);
                
                // Grid position centered at origin
                const x = (u - 0.5) * planeWidth;
                const y = (0.5 - v) * planeHeight;  // Flip Y
                
                positions[idx * 3 + 0] = x;
                positions[idx * 3 + 1] = y;
                positions[idx * 3 + 2] = 0;  // Z will be displaced in shader
                
                uvs[idx * 2 + 0] = u;
                uvs[idx * 2 + 1] = 1.0 - v;  // Flip V for texture sampling
                
                idx++;
            }
        }
        
        // Create geometry
        this.geometry = new THREE.BufferGeometry();
        this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        this.geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
        
        // Create uniforms
        this.uniforms = this._createUniforms();
        
        // Calculate point size to ensure coverage:
        // Grid cell size in pixels ≈ (canvas height / gridResY) * overlap factor
        // For a ~600px canvas with 200 grid points, each cell is ~3px
        // We need points slightly larger than cell size for overlap
        // pointSize is in screen pixels
        const cellSizeEstimate = 600 / gridResY;  // Rough estimate
        const overlapFactor = 1.5;  // Ensure overlap between points
        const calculatedPointSize = cellSizeEstimate * overlapFactor;
        
        // Use config value or calculated, whichever gives better coverage
        const userPointSize = this.config.pointSize || 3.0;
        this.uniforms.pointSize.value = Math.max(userPointSize, calculatedPointSize);
        
        console.log(`[ImageScene] Point size: ${this.uniforms.pointSize.value.toFixed(1)}px (grid cell ~${cellSizeEstimate.toFixed(1)}px)`);
        
        // Get shader
        const shaderConfig = Shaders.pointCloud;
        
        this.material = new THREE.ShaderMaterial({
            uniforms: this.uniforms,
            vertexShader: shaderConfig.vertex,
            fragmentShader: shaderConfig.fragment,
            transparent: true,
            depthWrite: true,
            depthTest: true,
        });
        
        // Create Points object
        this.mesh = new THREE.Points(this.geometry, this.material);
        this.mesh.name = this.name + '_pointcloud';
        
        this.parentScene.add(this.mesh);
        
        console.log(`[ImageScene] Point cloud created successfully`);
    }
    
    /**
     * Create DEPTH-ONLY parallax mesh
     * Single flat plane with UV-based parallax - no mask needed!
     * Uses only: image + depth map
     */
    _createHybridMesh() {
        const planeHeight = 2;
        const planeWidth = planeHeight * this.aspectRatio;
        
        // Create uniforms
        this.uniforms = this._createUniforms();
        
        // Add parallax-specific uniforms
        this.uniforms.parallaxAmount = { value: 0.05 };  // UV shift amount
        this.uniforms.viewOffset = { value: new THREE.Vector2(0, 0) };  // Updated from camera
        this.uniforms.debugLayer = { value: 0.0 };
        
        // Simple flat quad - no subdivision needed!
        this.geometry = new THREE.PlaneGeometry(planeWidth, planeHeight, 1, 1);
        
        const shaderConfig = Shaders.hybrid;
        
        this.material = new THREE.ShaderMaterial({
            uniforms: this.uniforms,
            vertexShader: shaderConfig.vertex,
            fragmentShader: shaderConfig.fragment,
            transparent: true,
            side: THREE.DoubleSide,
        });
        
        this.mesh = new THREE.Mesh(this.geometry, this.material);
        this.mesh.name = this.name + '_parallax';
        this.parentScene.add(this.mesh);
        
        console.log(`[ImageScene] Depth-only parallax created (no mask needed)`);
    }
    
    /**
     * Create edge points for hybrid mode
     * Points are generated where depth gradient is high
     */
    _createEdgePoints(planeWidth, planeHeight) {
        // We need to compute edge strength from the depth texture
        // Since we can't easily read GPU textures, we'll create a dense grid
        // and let the shader discard non-edge points
        
        const density = this.pointCloudDensity || 1;
        const baseResolution = 400;  // Higher resolution for better edge coverage
        const gridResX = Math.floor(baseResolution / density);
        const gridResY = Math.floor(gridResX / this.aspectRatio);
        const totalPoints = gridResX * gridResY;
        
        console.log(`[ImageScene] Creating edge points: ${totalPoints.toLocaleString()} candidate points (${gridResX}x${gridResY})`);
        
        // Create arrays
        const positions = new Float32Array(totalPoints * 3);
        const uvs = new Float32Array(totalPoints * 2);
        
        // Create grid of point positions and UVs
        // Edge detection happens in vertex shader from depth gradient
        let idx = 0;
        for (let iy = 0; iy < gridResY; iy++) {
            for (let ix = 0; ix < gridResX; ix++) {
                const u = ix / (gridResX - 1);
                const v = iy / (gridResY - 1);
                
                const x = (u - 0.5) * planeWidth;
                const y = (0.5 - v) * planeHeight;
                
                positions[idx * 3 + 0] = x;
                positions[idx * 3 + 1] = y;
                positions[idx * 3 + 2] = 0;
                
                uvs[idx * 2 + 0] = u;
                uvs[idx * 2 + 1] = 1.0 - v;
                
                idx++;
            }
        }
        
        // Create geometry
        const pointGeometry = new THREE.BufferGeometry();
        pointGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        pointGeometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
        
        // Create point material with edge shader
        const pointShaderConfig = Shaders.hybrid.points;
        
        // Point size needs to be LARGE enough to cover mesh gaps at edges
        // Bigger points = better coverage of stretching artifacts
        const cellSize = 600 / gridResY;
        const pointSize = Math.max(this.config.pointSize || 6, cellSize * 2.0);
        
        const pointMaterial = new THREE.ShaderMaterial({
            uniforms: {
                ...this.uniforms,
                pointSize: { value: pointSize },
                pointSoftness: { value: this.config.pointSoftness || 0.4 },
                debugEdges: { value: 0.0 },  // 0 = normal, 1 = visualize edges
            },
            vertexShader: pointShaderConfig.vertex,
            fragmentShader: pointShaderConfig.fragment,
            transparent: true,
            depthWrite: false,  // Don't write depth (render on top of mesh)
            depthTest: true,
        });
        
        // Create Points object
        this.edgePoints = new THREE.Points(pointGeometry, pointMaterial);
        this.edgePoints.name = this.name + '_edge_points';
        this.edgePoints.renderOrder = 1;  // Render after mesh
        this.parentScene.add(this.edgePoints);
        
        this.edgePointCount = totalPoints;
        this.edgePointGeometry = pointGeometry;
        this.edgePointMaterial = pointMaterial;
    }
    
    /**
     * Rebuild mesh (resolution change or mode change)
     */
    rebuildMesh(resolution = null) {
        if (!this.imageTexture || !this.depthTexture) return;
        
        // Store current state
        const wasVisible = this.isVisible;
        const currentOpacity = this.opacity;
        
        // Remove old mesh
        if (this.mesh) {
            this.parentScene.remove(this.mesh);
            this.geometry.dispose();
            this.material.dispose();
        }
        
        // Remove edge points if they exist (from old hybrid mode)
        if (this.edgePoints) {
            this.parentScene.remove(this.edgePoints);
            this.edgePointGeometry?.dispose();
            this.edgePointMaterial?.dispose();
            this.edgePoints = null;
            this.edgePointGeometry = null;
            this.edgePointMaterial = null;
        }
        
        // Remove background mesh if exists (from layered hybrid mode)
        if (this.bgMesh) {
            this.parentScene.remove(this.bgMesh);
            this.bgGeometry?.dispose();
            this.bgMaterial?.dispose();
            this.bgMesh = null;
            this.bgGeometry = null;
            this.bgMaterial = null;
        }
        
        // Update config if new resolution provided
        if (resolution !== null) {
            this.config.meshResolution = resolution;
        }
        
        // Recreate mesh
        this._createMesh();
        
        // Restore state
        this.setVisible(wasVisible);
        this.uniforms.opacity.value = currentOpacity;
        this.opacity = currentOpacity;
    }
    
    /**
     * Set render mode (affects artifact handling)
     */
    setRenderMode(mode) {
        if (this.renderMode === mode) return;
        this.renderMode = mode;
        this.rebuildMesh();
        
        console.log(`[ImageScene] Render mode: ${mode}`);
    }
    
    /**
     * Set debug visualization mode
     */
    setDebugMode(mode) {
        if (this.debugMode === mode) return;
        this.debugMode = mode;
        this.rebuildMesh();
        
        console.log(`[ImageScene] Debug mode: ${mode}`);
    }
    
    /**
     * Update depth intensity
     */
    setDepthIntensity(intensity) {
        this.config.depthIntensity = intensity;
        if (this.uniforms) {
            this.uniforms.depthIntensity.value = intensity;
        }
    }
    
    /**
     * Set edge softness (for edge-aware mode)
     */
    setEdgeSoftness(value) {
        if (this.uniforms) {
            this.uniforms.edgeSoftness.value = value;
        }
    }
    
    /**
     * Set edge threshold for artifact fading / hybrid mode edge detection
     * @param {number} value - Depth difference threshold (0.02-0.2)
     */
    setEdgeThreshold(value) {
        if (this.uniforms) {
            this.uniforms.edgeThreshold.value = value;
        }
        // Also update edge point material (hybrid mode)
        if (this.edgePointMaterial && this.edgePointMaterial.uniforms) {
            this.edgePointMaterial.uniforms.edgeThreshold.value = value;
        }
    }
    
    /**
     * Toggle debug visualization of edge points (hybrid mode)
     * @param {boolean} enabled - Show edge points as cyan color
     */
    setDebugEdges(enabled) {
        if (this.edgePointMaterial && this.edgePointMaterial.uniforms) {
            this.edgePointMaterial.uniforms.debugEdges.value = enabled ? 1.0 : 0.0;
        }
    }
    
    /**
     * Toggle debug visualization of layers (hybrid mode)
     * Shows foreground as GREEN tint, background as MAGENTA tint
     * @param {boolean} enabled - Show layer tints
     */
    setDebugLayers(enabled) {
        if (this.uniforms && this.uniforms.debugLayer) {
            this.uniforms.debugLayer.value = enabled ? 1.0 : 0.0;
        }
    }
    
    /**
     * Set view offset for parallax effect (hybrid mode)
     * @param {number} x - Horizontal offset (-1 to 1)
     * @param {number} y - Vertical offset (-1 to 1)
     */
    setViewOffset(x, y) {
        if (this.uniforms && this.uniforms.viewOffset) {
            this.uniforms.viewOffset.value.set(x, y);
        }
    }
    
    /**
     * Set parallax amount for hybrid mode
     * @param {number} amount - UV shift multiplier (0.01-0.1)
     */
    setParallaxAmount(amount) {
        if (this.uniforms && this.uniforms.parallaxAmount) {
            this.uniforms.parallaxAmount.value = amount;
        }
    }
    
    /**
     * Show only background layer (hide foreground)
     * @param {boolean} bgOnly - true = show BG only
     */
    setShowBackgroundOnly(bgOnly) {
        if (this.mesh) {
            this.mesh.visible = !bgOnly && this.isVisible;
        }
    }
    
    /**
     * Show only foreground layer (hide background)
     * @param {boolean} fgOnly - true = show FG only
     */
    setShowForegroundOnly(fgOnly) {
        if (this.bgMesh) {
            this.bgMesh.visible = !fgOnly && this.isVisible;
        }
    }
    
    /**
     * Set edge fade width for artifact hiding
     * @param {number} value - How gradual the fade is (0.05-0.3)
     */
    setEdgeFadeWidth(value) {
        if (this.uniforms) {
            this.uniforms.edgeFadeWidth.value = value;
        }
    }
    
    /**
     * Enable/disable soft edges (reduces displacement + alpha at depth edges)
     * @param {boolean} enabled - true to enable
     */
    setSoftEdges(enabled) {
        if (this.uniforms) {
            this.uniforms.softEdges.value = enabled ? 1.0 : 0.0;
        }
    }
    
    /**
     * Set parallax quality (number of layers)
     */
    setParallaxQuality(layers) {
        if (this.uniforms) {
            this.uniforms.numLayers.value = layers;
        }
    }
    
    // =========================================================================
    // POINT CLOUD METHODS
    // =========================================================================
    
    /**
     * Load inpainted texture for point cloud mode
     * This texture fills in regions hidden in the original view
     */
    async loadInpaintedTexture(url) {
        const textureLoader = new THREE.TextureLoader();
        
        try {
            this.inpaintedTexture = await this._loadTexture(textureLoader, url);
            this.inpaintedTexture.colorSpace = THREE.SRGBColorSpace;
            this.inpaintedTexture.minFilter = THREE.LinearFilter;
            this.inpaintedTexture.magFilter = THREE.LinearFilter;
            
            if (this.uniforms) {
                this.uniforms.inpaintedMap.value = this.inpaintedTexture;
                this.uniforms.useInpainted.value = 1.0;
            }
            
            console.log(`[ImageScene] Loaded inpainted texture: ${url}`);
            return true;
        } catch (error) {
            console.warn(`[ImageScene] Failed to load inpainted texture: ${error}`);
            return false;
        }
    }
    
    /**
     * Load mask texture for hybrid mode (foreground/background separation)
     * White pixels = foreground subject, Black pixels = background
     */
    async loadMaskTexture(url) {
        const textureLoader = new THREE.TextureLoader();
        
        try {
            this.maskTexture = await this._loadTexture(textureLoader, url);
            this.maskTexture.minFilter = THREE.LinearFilter;
            this.maskTexture.magFilter = THREE.LinearFilter;
            
            if (this.uniforms && this.uniforms.maskMap) {
                this.uniforms.maskMap.value = this.maskTexture;
            }
            
            console.log(`[ImageScene] Loaded mask texture: ${url}`);
            return true;
        } catch (error) {
            console.warn(`[ImageScene] Failed to load mask texture: ${error}`);
            return false;
        }
    }
    
    /**
     * Set point cloud density (affects performance vs quality)
     * @param {number} density - 1 = full resolution, 2 = half, 3 = third, etc.
     */
    setPointCloudDensity(density) {
        this.pointCloudDensity = Math.max(1, Math.floor(density));
    }
    
    /**
     * Set point size for point cloud mode
     * @param {number} size - Point size in screen pixels
     */
    setPointSize(size) {
        this.config.pointSize = size;
        if (this.uniforms) {
            // Direct pixel size - no density scaling needed
            this.uniforms.pointSize.value = size;
        }
    }
    
    /**
     * Set how much depth affects point size
     * @param {number} attenuation - 0 = no effect, 1 = near points 2x size
     */
    setPointSizeAttenuation(attenuation) {
        this.config.pointSizeAttenuation = attenuation;
        if (this.uniforms) {
            this.uniforms.pointSizeAttenuation.value = attenuation;
        }
    }
    
    /**
     * Set point softness (edge falloff)
     * @param {number} softness - 0 = hard circles, 1 = very soft
     */
    setPointSoftness(softness) {
        this.config.pointSoftness = softness;
        if (this.uniforms) {
            this.uniforms.pointSoftness.value = softness;
        }
    }
    
    /**
     * Get point cloud statistics
     */
    getPointCloudStats() {
        if (this.renderMode !== RenderMode.POINT_CLOUD || !this.depthTexture) {
            return null;
        }
        
        const density = this.pointCloudDensity;
        const baseResolution = 400;
        const gridResX = Math.floor(baseResolution / density);
        const gridResY = Math.floor(gridResX / this.aspectRatio);
        const pointsRendered = gridResX * gridResY;
        
        return {
            gridResolution: `${gridResX}×${gridResY}`,
            pointsRendered: pointsRendered,
            density: density,
            pointSize: this.uniforms?.pointSize?.value || this.config.pointSize || 5,
            hasInpaintedTexture: !!this.inpaintedTexture
        };
    }
    
    /**
     * Set visibility
     */
    setVisible(visible) {
        this.isVisible = visible;
        if (this.mesh) {
            this.mesh.visible = visible;
        }
        if (this.bgMesh) {
            this.bgMesh.visible = visible;
        }
        if (this.edgePoints) {
            this.edgePoints.visible = visible;
        }
    }
    
    /**
     * Set wireframe mode
     */
    setWireframe(enabled) {
        if (this.material) {
            this.material.wireframe = enabled;
        }
    }
    
    /**
     * Get diagnostic info about the depth map
     */
    getDiagnostics() {
        if (!this.depthTexture) return null;
        
        const img = this.depthTexture.image;
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        const data = imageData.data;
        
        let min = 255, max = 0, sum = 0;
        let edgeCount = 0;
        const histogram = new Array(256).fill(0);
        
        for (let y = 0; y < img.height; y++) {
            for (let x = 0; x < img.width; x++) {
                const i = (y * img.width + x) * 4;
                const depth = data[i];
                
                min = Math.min(min, depth);
                max = Math.max(max, depth);
                sum += depth;
                histogram[depth]++;
                
                // Count sharp edges
                if (x > 0 && y > 0) {
                    const left = data[i - 4];
                    const top = data[i - img.width * 4];
                    if (Math.abs(depth - left) > 30 || Math.abs(depth - top) > 30) {
                        edgeCount++;
                    }
                }
            }
        }
        
        const totalPixels = img.width * img.height;
        
        return {
            width: img.width,
            height: img.height,
            minDepth: min,
            maxDepth: max,
            avgDepth: sum / totalPixels,
            depthRange: max - min,
            sharpEdgePercentage: (edgeCount / totalPixels * 100).toFixed(2),
            histogram: histogram,
            recommendation: this._getRecommendation(max - min, edgeCount / totalPixels)
        };
    }
    
    /**
     * Get recommendation based on diagnostics
     */
    _getRecommendation(depthRange, edgeRatio) {
        const tips = [];
        
        if (depthRange < 50) {
            tips.push('Depth map has low contrast - increase depth intensity for more effect');
        }
        
        if (edgeRatio > 0.1) {
            tips.push('High edge count detected - use Edge-Aware mode to reduce artifacts');
            tips.push('Consider blurring the depth map slightly');
        }
        
        if (edgeRatio > 0.2) {
            tips.push('Very sharp depth transitions - try Parallax mode for smoothest results');
        }
        
        return tips.length > 0 ? tips : ['Depth map looks good!'];
    }
    
    /**
     * Fade in animation
     */
    fadeIn(duration = 500) {
        return new Promise((resolve) => {
            if (this.uniforms) this.uniforms.opacity.value = 0;
            this.opacity = 0;
            this.targetOpacity = 1;
            
            const startTime = performance.now();
            
            const animate = () => {
                const elapsed = performance.now() - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                
                this.opacity = eased;
                if (this.uniforms) this.uniforms.opacity.value = eased;
                
                if (progress < 1) {
                    requestAnimationFrame(animate);
                } else {
                    resolve();
                }
            };
            
            animate();
        });
    }
    
    /**
     * Fade out animation
     */
    fadeOut(duration = 500) {
        return new Promise((resolve) => {
            this.targetOpacity = 0;
            const startOpacity = this.opacity;
            const startTime = performance.now();
            
            const animate = () => {
                const elapsed = performance.now() - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const eased = Math.pow(progress, 3);
                
                this.opacity = startOpacity * (1 - eased);
                if (this.uniforms) this.uniforms.opacity.value = this.opacity;
                
                if (progress < 1) {
                    requestAnimationFrame(animate);
                } else {
                    resolve();
                }
            };
            
            animate();
        });
    }
    
    /**
     * Update called each frame
     */
    update(deltaTime) {
        // Per-frame updates here
    }
    
    /**
     * Cleanup
     */
    dispose() {
        if (this.mesh) {
            this.parentScene.remove(this.mesh);
        }
        if (this.bgMesh) {
            this.parentScene.remove(this.bgMesh);
        }
        if (this.edgePoints) {
            this.parentScene.remove(this.edgePoints);
        }
        if (this.geometry) this.geometry.dispose();
        if (this.material) this.material.dispose();
        if (this.bgGeometry) this.bgGeometry.dispose();
        if (this.bgMaterial) this.bgMaterial.dispose();
        if (this.edgePointGeometry) this.edgePointGeometry.dispose();
        if (this.edgePointMaterial) this.edgePointMaterial.dispose();
        if (this.imageTexture) this.imageTexture.dispose();
        if (this.depthTexture) this.depthTexture.dispose();
        if (this.maskTexture) this.maskTexture.dispose();
        if (this.inpaintedTexture) this.inpaintedTexture.dispose();
    }
}
