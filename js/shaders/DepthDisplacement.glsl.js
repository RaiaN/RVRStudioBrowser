/**
 * DepthDisplacement.glsl.js
 * GPU shaders for depth-displaced image rendering.
 * 
 * Techniques included:
 * - Basic vertex displacement
 * - Edge-aware smoothing
 * - Soft depth transitions
 */

// =============================================================================
// VERTEX SHADERS
// =============================================================================

/**
 * Basic depth displacement - displaces vertices based on depth map
 */
export const BasicDisplacementVertex = `
    uniform sampler2D depthMap;
    uniform float depthIntensity;
    uniform float depthBias;
    
    varying vec2 vUv;
    varying float vDepth;
    varying vec3 vNormal;
    varying vec3 vViewPosition;
    
    void main() {
        vUv = uv;
        
        // Sample depth from texture with linear filtering
        float depth = texture2D(depthMap, uv).r;
        vDepth = depth;
        
        // Displace vertex along Z axis
        vec3 displaced = position;
        displaced.z += (depth - depthBias) * depthIntensity;
        
        // Transform normal
        vNormal = normalize(normalMatrix * normal);
        
        // Calculate view position for lighting
        vec4 mvPosition = modelViewMatrix * vec4(displaced, 1.0);
        vViewPosition = -mvPosition.xyz;
        
        gl_Position = projectionMatrix * mvPosition;
    }
`;

/**
 * Edge-aware displacement - smooths depth at edges to reduce artifacts
 */
export const EdgeAwareDisplacementVertex = `
    uniform sampler2D depthMap;
    uniform float depthIntensity;
    uniform float depthBias;
    uniform vec2 texelSize;  // 1.0 / texture resolution
    uniform float edgeSoftness;
    
    varying vec2 vUv;
    varying float vDepth;
    varying vec3 vNormal;
    varying vec3 vViewPosition;
    
    // Sample depth with edge-aware bilateral-like filtering
    float sampleDepthSmooth(vec2 uv) {
        float center = texture2D(depthMap, uv).r;
        
        // Sample neighbors
        float left   = texture2D(depthMap, uv + vec2(-texelSize.x, 0.0)).r;
        float right  = texture2D(depthMap, uv + vec2( texelSize.x, 0.0)).r;
        float top    = texture2D(depthMap, uv + vec2(0.0,  texelSize.y)).r;
        float bottom = texture2D(depthMap, uv + vec2(0.0, -texelSize.y)).r;
        
        // Calculate edge strength (depth discontinuity)
        float edgeH = abs(left - right);
        float edgeV = abs(top - bottom);
        float edge = max(edgeH, edgeV);
        
        // Blend towards average at edges
        float avg = (left + right + top + bottom) * 0.25;
        float blendFactor = smoothstep(0.0, edgeSoftness, edge);
        
        return mix(center, avg, blendFactor * 0.5);
    }
    
    void main() {
        vUv = uv;
        
        // Use edge-aware depth sampling
        float depth = sampleDepthSmooth(uv);
        vDepth = depth;
        
        // Displace vertex
        vec3 displaced = position;
        displaced.z += (depth - depthBias) * depthIntensity;
        
        vNormal = normalize(normalMatrix * normal);
        
        vec4 mvPosition = modelViewMatrix * vec4(displaced, 1.0);
        vViewPosition = -mvPosition.xyz;
        
        gl_Position = projectionMatrix * mvPosition;
    }
`;

// =============================================================================
// FRAGMENT SHADERS
// =============================================================================

/**
 * Basic textured output
 */
export const BasicFragment = `
    uniform sampler2D imageMap;
    uniform float opacity;
    
    varying vec2 vUv;
    varying float vDepth;
    
    void main() {
        vec4 texColor = texture2D(imageMap, vUv);
        gl_FragColor = vec4(texColor.rgb, texColor.a * opacity);
    }
`;

/**
 * Lit fragment shader - uses depth-derived normals for lighting
 * This adds subtle 3D shading to enhance the depth effect
 */
export const LitFragment = `
    uniform sampler2D imageMap;
    uniform sampler2D depthMap;
    uniform vec2 texelSize;
    uniform float opacity;
    uniform float lightIntensity;
    uniform vec3 lightDirection;
    
    varying vec2 vUv;
    varying float vDepth;
    varying vec3 vNormal;
    varying vec3 vViewPosition;
    
    // Calculate normal from depth map gradient
    vec3 calculateNormalFromDepth(vec2 uv) {
        // Sample neighboring depth values
        float left  = texture2D(depthMap, uv + vec2(-texelSize.x, 0.0)).r;
        float right = texture2D(depthMap, uv + vec2( texelSize.x, 0.0)).r;
        float up    = texture2D(depthMap, uv + vec2(0.0,  texelSize.y)).r;
        float down  = texture2D(depthMap, uv + vec2(0.0, -texelSize.y)).r;
        
        // Calculate gradient (slope)
        float dx = (right - left) * 2.0;
        float dy = (up - down) * 2.0;
        
        // Convert gradient to normal vector
        // The Z component controls how "bumpy" the surface appears
        vec3 normal = normalize(vec3(-dx, -dy, 0.1));
        
        return normal;
    }
    
    void main() {
        vec4 texColor = texture2D(imageMap, vUv);
        
        // Calculate surface normal from depth gradient
        vec3 normal = calculateNormalFromDepth(vUv);
        
        // Simple directional lighting
        vec3 lightDir = normalize(lightDirection);
        float NdotL = max(dot(normal, lightDir), 0.0);
        
        // Ambient + diffuse lighting
        float ambient = 0.7;
        float diffuse = NdotL * lightIntensity;
        float lighting = ambient + diffuse * 0.3;
        
        // Apply lighting to texture
        vec3 litColor = texColor.rgb * lighting;
        
        gl_FragColor = vec4(litColor, texColor.a * opacity);
    }
`;

/**
 * Debug: Visualize depth map
 */
export const DebugDepthFragment = `
    uniform sampler2D depthMap;
    uniform float opacity;
    uniform int colorMode;  // 0=grayscale, 1=heatmap, 2=contours
    
    varying vec2 vUv;
    varying float vDepth;
    
    // Heatmap color palette
    vec3 heatmap(float t) {
        // Blue -> Cyan -> Green -> Yellow -> Red
        vec3 c;
        t = clamp(t, 0.0, 1.0);
        if (t < 0.25) {
            c = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t * 4.0);
        } else if (t < 0.5) {
            c = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.25) * 4.0);
        } else if (t < 0.75) {
            c = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.5) * 4.0);
        } else {
            c = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.75) * 4.0);
        }
        return c;
    }
    
    void main() {
        float depth = texture2D(depthMap, vUv).r;
        vec3 color;
        
        if (colorMode == 0) {
            // Grayscale
            color = vec3(depth);
        } else if (colorMode == 1) {
            // Heatmap
            color = heatmap(depth);
        } else {
            // Contour lines
            float contours = fract(depth * 20.0);
            contours = smoothstep(0.0, 0.1, contours) * smoothstep(1.0, 0.9, contours);
            color = mix(vec3(0.1), heatmap(depth), contours);
        }
        
        gl_FragColor = vec4(color, opacity);
    }
`;

/**
 * Debug: Visualize depth edges (where artifacts occur)
 */
export const DebugEdgesFragment = `
    uniform sampler2D depthMap;
    uniform sampler2D imageMap;
    uniform vec2 texelSize;
    uniform float opacity;
    uniform float edgeThreshold;
    
    varying vec2 vUv;
    
    void main() {
        // Sobel edge detection on depth
        float tl = texture2D(depthMap, vUv + vec2(-texelSize.x,  texelSize.y)).r;
        float t  = texture2D(depthMap, vUv + vec2( 0.0,          texelSize.y)).r;
        float tr = texture2D(depthMap, vUv + vec2( texelSize.x,  texelSize.y)).r;
        float l  = texture2D(depthMap, vUv + vec2(-texelSize.x,  0.0)).r;
        float r  = texture2D(depthMap, vUv + vec2( texelSize.x,  0.0)).r;
        float bl = texture2D(depthMap, vUv + vec2(-texelSize.x, -texelSize.y)).r;
        float b  = texture2D(depthMap, vUv + vec2( 0.0,         -texelSize.y)).r;
        float br = texture2D(depthMap, vUv + vec2( texelSize.x, -texelSize.y)).r;
        
        // Sobel operators
        float gx = (tr + 2.0*r + br) - (tl + 2.0*l + bl);
        float gy = (bl + 2.0*b + br) - (tl + 2.0*t + tr);
        float edge = sqrt(gx*gx + gy*gy);
        
        // Original image
        vec4 texColor = texture2D(imageMap, vUv);
        
        // Highlight edges in red (these are problem areas)
        float edgeMask = smoothstep(edgeThreshold * 0.5, edgeThreshold, edge);
        vec3 finalColor = mix(texColor.rgb, vec3(1.0, 0.0, 0.0), edgeMask * 0.8);
        
        gl_FragColor = vec4(finalColor, opacity);
    }
`;

/**
 * Debug: Visualize displacement magnitude
 */
export const DebugDisplacementFragment = `
    uniform sampler2D depthMap;
    uniform float depthIntensity;
    uniform float depthBias;
    uniform float opacity;
    
    varying vec2 vUv;
    varying float vDepth;
    
    vec3 heatmap(float t) {
        t = clamp(t, 0.0, 1.0);
        return vec3(
            smoothstep(0.5, 0.8, t),
            smoothstep(0.0, 0.5, t) - smoothstep(0.5, 1.0, t),
            1.0 - smoothstep(0.2, 0.5, t)
        );
    }
    
    void main() {
        float depth = texture2D(depthMap, vUv).r;
        float displacement = abs(depth - depthBias) * depthIntensity;
        
        // Normalize to visible range
        float normalizedDisp = displacement / (depthIntensity * 0.5);
        
        gl_FragColor = vec4(heatmap(normalizedDisp), opacity);
    }
`;

/**
 * Debug: Wireframe with depth coloring
 */
export const DebugWireframeFragment = `
    uniform float opacity;
    varying float vDepth;
    
    void main() {
        // Color by depth: blue=far, red=close
        vec3 color = mix(vec3(0.2, 0.4, 1.0), vec3(1.0, 0.3, 0.2), vDepth);
        gl_FragColor = vec4(color, opacity);
    }
`;

// =============================================================================
// ADVANCED TECHNIQUES
// =============================================================================

/**
 * Parallax Occlusion Mapping - per-pixel depth without vertex displacement
 * Smoother but more GPU intensive
 */
export const ParallaxOcclusionVertex = `
    varying vec2 vUv;
    varying vec3 vViewDir;
    varying vec3 vTangentViewDir;
    
    void main() {
        vUv = uv;
        
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        
        // Calculate view direction in tangent space
        vec3 viewDir = normalize(-mvPosition.xyz);
        
        // Simple tangent space (assuming flat plane)
        vec3 T = normalize(vec3(modelViewMatrix * vec4(1.0, 0.0, 0.0, 0.0)));
        vec3 B = normalize(vec3(modelViewMatrix * vec4(0.0, 1.0, 0.0, 0.0)));
        vec3 N = normalize(vec3(modelViewMatrix * vec4(0.0, 0.0, 1.0, 0.0)));
        
        mat3 TBN = mat3(T, B, N);
        vTangentViewDir = normalize(TBN * viewDir);
        vViewDir = viewDir;
        
        gl_Position = projectionMatrix * mvPosition;
    }
`;

export const ParallaxOcclusionFragment = `
    uniform sampler2D imageMap;
    uniform sampler2D depthMap;
    uniform float depthIntensity;
    uniform float opacity;
    uniform int numLayers;  // Quality: 8-64
    
    varying vec2 vUv;
    varying vec3 vTangentViewDir;
    
    vec2 parallaxMapping(vec2 uv, vec3 viewDir) {
        float layerDepth = 1.0 / float(numLayers);
        float currentLayerDepth = 0.0;
        
        // Parallax offset direction and amount
        vec2 P = viewDir.xy * depthIntensity * 0.1;
        vec2 deltaUV = P / float(numLayers);
        
        vec2 currentUV = uv;
        float depthMapValue = 1.0 - texture2D(depthMap, currentUV).r;
        
        // Ray march through depth layers
        for (int i = 0; i < 64; i++) {
            if (i >= numLayers) break;
            if (currentLayerDepth >= depthMapValue) break;
            
            currentUV -= deltaUV;
            depthMapValue = 1.0 - texture2D(depthMap, currentUV).r;
            currentLayerDepth += layerDepth;
        }
        
        // Interpolation for smoother result
        vec2 prevUV = currentUV + deltaUV;
        float afterDepth = depthMapValue - currentLayerDepth;
        float beforeDepth = (1.0 - texture2D(depthMap, prevUV).r) - currentLayerDepth + layerDepth;
        float weight = afterDepth / (afterDepth - beforeDepth);
        
        return mix(currentUV, prevUV, weight);
    }
    
    void main() {
        vec2 parallaxUV = parallaxMapping(vUv, vTangentViewDir);
        
        // Discard if UV out of bounds
        if (parallaxUV.x < 0.0 || parallaxUV.x > 1.0 || 
            parallaxUV.y < 0.0 || parallaxUV.y > 1.0) {
            discard;
        }
        
        vec4 texColor = texture2D(imageMap, parallaxUV);
        gl_FragColor = vec4(texColor.rgb, texColor.a * opacity);
    }
`;

// =============================================================================
// SHADER COLLECTIONS
// =============================================================================

export const Shaders = {
    // Basic displacement (default)
    basic: {
        vertex: BasicDisplacementVertex,
        fragment: BasicFragment,
        uniforms: ['depthMap', 'depthIntensity', 'depthBias', 'imageMap', 'opacity']
    },
    
    // Debug modes
    debug: {
        depth: {
            vertex: BasicDisplacementVertex,
            fragment: DebugDepthFragment,
            uniforms: ['depthMap', 'depthIntensity', 'depthBias', 'opacity', 'colorMode']
        },
        edges: {
            vertex: BasicDisplacementVertex,
            fragment: DebugEdgesFragment,
            uniforms: ['depthMap', 'imageMap', 'texelSize', 'opacity', 'edgeThreshold', 'depthIntensity', 'depthBias']
        },
        displacement: {
            vertex: BasicDisplacementVertex,
            fragment: DebugDisplacementFragment,
            uniforms: ['depthMap', 'depthIntensity', 'depthBias', 'opacity']
        },
        wireframe: {
            vertex: BasicDisplacementVertex,
            fragment: DebugWireframeFragment,
            uniforms: ['depthMap', 'depthIntensity', 'depthBias', 'opacity']
        }
    }
};

export default Shaders;
