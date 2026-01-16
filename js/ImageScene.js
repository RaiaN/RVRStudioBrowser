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
    BASIC: 'basic',            // Simple vertex displacement
    EDGE_AWARE: 'edgeAware',   // Basic + edge fade (hides artifacts) ⭐
    MULTI_LAYER: 'multiLayer', // Separates into layers
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
        
        this.name = '';
        this.aspectRatio = 1;
        
        // Animation state
        this.opacity = 1;
        this.targetOpacity = 1;
        this.isVisible = true;
        
        // Shader uniforms
        this.uniforms = null;
        
        // Current modes
        this.renderMode = RenderMode.BASIC;  // Default to basic for best performance
        this.debugMode = DebugMode.NONE;
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
            resolution: { value: new THREE.Vector2(texWidth, texHeight) }
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
        
        return Shaders[this.renderMode] || Shaders.basic;
    }
    
    /**
     * Create the mesh with current shader mode
     */
    _createMesh() {
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
     * Set edge threshold for artifact fading
     * @param {number} value - Depth difference that triggers fade (0.02-0.2)
     */
    setEdgeThreshold(value) {
        if (this.uniforms) {
            this.uniforms.edgeThreshold.value = value;
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
    
    /**
     * Set visibility
     */
    setVisible(visible) {
        this.isVisible = visible;
        if (this.mesh) {
            this.mesh.visible = visible;
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
        if (this.geometry) this.geometry.dispose();
        if (this.material) this.material.dispose();
        if (this.imageTexture) this.imageTexture.dispose();
        if (this.depthTexture) this.depthTexture.dispose();
    }
}
