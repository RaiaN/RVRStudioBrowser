/**
 * MultiLayerScene.js
 * Renders images as multiple depth-separated layers.
 * 
 * This solves the artifact problem by:
 * - Separating foreground and background into different meshes
 * - No triangles span depth discontinuities
 * - Each layer is smooth and artifact-free
 * - Full 3D depth effect is preserved
 */

import * as THREE from 'three';
import { Shaders } from './shaders/DepthDisplacement.glsl.js';

export class MultiLayerScene {
    constructor(parentScene, config) {
        this.parentScene = parentScene;
        this.config = config;
        
        this.layers = [];         // Array of layer meshes
        this.layerData = null;    // Layer metadata from JSON
        
        this.name = '';
        this.aspectRatio = 1;
        
        // Animation state
        this.opacity = 1;
        this.isVisible = true;
        
        // Group to hold all layers
        this.group = new THREE.Group();
        this.parentScene.add(this.group);
    }
    
    /**
     * Load multi-layer scene from directory
     * Expected structure:
     *   layers/
     *     layers.json         - Metadata
     *     layer_00.png        - Background layer (with alpha)
     *     layer_01.png        - Foreground layer (with alpha)
     *     ...
     * 
     * @param {string} layerDir - Path to layers directory
     * @param {string} name - Display name
     */
    async loadFromDirectory(layerDir, name = '') {
        this.name = name;
        
        // Load metadata
        const metaResponse = await fetch(`${layerDir}/layers.json`);
        this.layerData = await metaResponse.json();
        
        console.log(`[MultiLayer] Loading ${this.layerData.num_layers} layers`);
        
        const textureLoader = new THREE.TextureLoader();
        
        // Load all layer textures
        for (const layerInfo of this.layerData.layers) {
            const texturePath = `${layerDir}/${layerInfo.file}`;
            
            const texture = await this._loadTexture(textureLoader, texturePath);
            texture.colorSpace = THREE.SRGBColorSpace;
            texture.minFilter = THREE.LinearFilter;
            texture.magFilter = THREE.LinearFilter;
            
            // Calculate aspect ratio from first layer
            if (this.layers.length === 0) {
                this.aspectRatio = texture.image.width / texture.image.height;
            }
            
            // Create mesh for this layer
            const mesh = this._createLayerMesh(texture, layerInfo.depth, layerInfo.id);
            this.layers.push({
                mesh: mesh,
                texture: texture,
                depth: layerInfo.depth,
                id: layerInfo.id
            });
            
            this.group.add(mesh);
        }
        
        return this;
    }
    
    /**
     * Load from single image + depth + segmentation mask
     * 
     * @param {string} imagePath - Path to source image
     * @param {string} depthPath - Path to depth map
     * @param {string} maskPath - Path to segmentation mask (grayscale, each value = layer ID)
     * @param {string} name - Display name
     */
    async loadFromMask(imagePath, depthPath, maskPath, name = '') {
        this.name = name;
        
        const textureLoader = new THREE.TextureLoader();
        
        // Load all textures
        const [imageTexture, depthTexture, maskTexture] = await Promise.all([
            this._loadTexture(textureLoader, imagePath),
            this._loadTexture(textureLoader, depthPath),
            this._loadTexture(textureLoader, maskPath)
        ]);
        
        this.aspectRatio = imageTexture.image.width / imageTexture.image.height;
        
        // Extract mask data to determine layer count
        const maskData = this._extractMaskData(maskTexture);
        const uniqueLayers = [...new Set(maskData.data)].sort((a, b) => a - b);
        
        console.log(`[MultiLayer] Found ${uniqueLayers.length} layers from mask`);
        
        // Create a mesh for each layer
        for (const layerId of uniqueLayers) {
            // Calculate average depth for this layer
            let depthSum = 0;
            let count = 0;
            const depthData = this._extractDepthData(depthTexture);
            
            for (let i = 0; i < maskData.data.length; i++) {
                if (maskData.data[i] === layerId) {
                    depthSum += depthData.data[i];
                    count++;
                }
            }
            
            const avgDepth = count > 0 ? depthSum / count : layerId / uniqueLayers.length;
            
            // Create layer mesh with custom shader that uses mask
            const mesh = this._createMaskedLayerMesh(
                imageTexture, 
                depthTexture, 
                maskTexture, 
                layerId, 
                avgDepth
            );
            
            this.layers.push({
                mesh: mesh,
                texture: imageTexture,
                depth: avgDepth,
                id: layerId
            });
            
            this.group.add(mesh);
        }
        
        return this;
    }
    
    /**
     * Load texture with promise wrapper
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
     * Extract mask data to array
     */
    _extractMaskData(texture) {
        const image = texture.image;
        const canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, image.width, image.height);
        const data = new Uint8Array(image.width * image.height);
        
        for (let i = 0; i < data.length; i++) {
            // Use red channel as layer ID
            data[i] = imageData.data[i * 4];
        }
        
        return { data, width: image.width, height: image.height };
    }
    
    /**
     * Extract depth data to array
     */
    _extractDepthData(texture) {
        const image = texture.image;
        const canvas = document.createElement('canvas');
        canvas.width = image.width;
        canvas.height = image.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, image.width, image.height);
        const data = new Float32Array(image.width * image.height);
        
        for (let i = 0; i < data.length; i++) {
            data[i] = imageData.data[i * 4] / 255.0;
        }
        
        return { data, width: image.width, height: image.height };
    }
    
    /**
     * Create mesh for a single layer (from pre-separated layer image)
     */
    _createLayerMesh(texture, depth, layerId) {
        const resolution = this.config.meshResolution;
        const planeHeight = 2;
        const planeWidth = planeHeight * this.aspectRatio;
        
        const geometry = new THREE.PlaneGeometry(
            planeWidth,
            planeHeight,
            resolution,
            Math.floor(resolution / this.aspectRatio)
        );
        
        // Simple material with alpha
        const material = new THREE.MeshBasicMaterial({
            map: texture,
            transparent: true,
            side: THREE.DoubleSide,
            depthWrite: true,
            depthTest: true,
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        
        // Position layer at its depth
        // depth 0 = far (background), depth 1 = close (foreground)
        const zPosition = (depth - 0.5) * this.config.depthIntensity;
        mesh.position.z = zPosition;
        
        mesh.name = `${this.name}_layer_${layerId}`;
        mesh.userData.layerId = layerId;
        mesh.userData.depth = depth;
        
        return mesh;
    }
    
    /**
     * Create mesh for layer using mask (single image, mask-based separation)
     */
    _createMaskedLayerMesh(imageTexture, depthTexture, maskTexture, layerId, avgDepth) {
        const resolution = this.config.meshResolution;
        const planeHeight = 2;
        const planeWidth = planeHeight * this.aspectRatio;
        
        const geometry = new THREE.PlaneGeometry(
            planeWidth,
            planeHeight,
            resolution,
            Math.floor(resolution / this.aspectRatio)
        );
        
        // Custom shader that only shows pixels belonging to this layer
        const uniforms = {
            imageMap: { value: imageTexture },
            depthMap: { value: depthTexture },
            maskMap: { value: maskTexture },
            layerId: { value: layerId / 255.0 },  // Normalized layer ID
            depthIntensity: { value: this.config.depthIntensity },
            opacity: { value: 1.0 },
        };
        
        const material = new THREE.ShaderMaterial({
            uniforms: uniforms,
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
                
                varying vec2 vUv;
                
                void main() {
                    float maskValue = texture2D(maskMap, vUv).r;
                    
                    // Only show pixels that belong to this layer
                    // Allow small tolerance for anti-aliasing
                    float tolerance = 0.02;
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
        });
        
        const mesh = new THREE.Mesh(geometry, material);
        
        // Position at average depth of this layer
        // (displacement is also applied in shader, so this is just base position)
        mesh.position.z = 0;
        
        mesh.name = `${this.name}_layer_${layerId}`;
        mesh.userData.layerId = layerId;
        mesh.userData.depth = avgDepth;
        mesh.userData.uniforms = uniforms;
        
        return mesh;
    }
    
    /**
     * Update depth intensity for all layers
     */
    setDepthIntensity(intensity) {
        this.config.depthIntensity = intensity;
        
        for (const layer of this.layers) {
            if (layer.mesh.userData.uniforms) {
                layer.mesh.userData.uniforms.depthIntensity.value = intensity;
            } else {
                // For simple layers, update Z position
                const zPosition = (layer.depth - 0.5) * intensity;
                layer.mesh.position.z = zPosition;
            }
        }
    }
    
    /**
     * Set visibility
     */
    setVisible(visible) {
        this.isVisible = visible;
        this.group.visible = visible;
    }
    
    /**
     * Set opacity for all layers
     */
    setOpacity(opacity) {
        this.opacity = opacity;
        
        for (const layer of this.layers) {
            if (layer.mesh.material.uniforms) {
                layer.mesh.material.uniforms.opacity.value = opacity;
            } else if (layer.mesh.material.opacity !== undefined) {
                layer.mesh.material.opacity = opacity;
            }
        }
    }
    
    /**
     * Fade in animation
     */
    fadeIn(duration = 500) {
        return new Promise((resolve) => {
            this.setOpacity(0);
            this.opacity = 0;
            
            const startTime = performance.now();
            
            const animate = () => {
                const elapsed = performance.now() - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const eased = 1 - Math.pow(1 - progress, 3);
                
                this.setOpacity(eased);
                this.opacity = eased;
                
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
            const startOpacity = this.opacity;
            const startTime = performance.now();
            
            const animate = () => {
                const elapsed = performance.now() - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const eased = Math.pow(progress, 3);
                
                const newOpacity = startOpacity * (1 - eased);
                this.setOpacity(newOpacity);
                this.opacity = newOpacity;
                
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
     * Update (called each frame)
     */
    update(deltaTime) {
        // Per-frame updates here
    }
    
    /**
     * Cleanup
     */
    dispose() {
        for (const layer of this.layers) {
            this.group.remove(layer.mesh);
            layer.mesh.geometry.dispose();
            layer.mesh.material.dispose();
            if (layer.texture) {
                layer.texture.dispose();
            }
        }
        
        this.parentScene.remove(this.group);
        this.layers = [];
    }
}
