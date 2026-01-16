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
 * Now with optional edge-aware displacement reduction to prevent stretching
 */
export const BasicDisplacementVertex = `
    uniform sampler2D depthMap;
    uniform float depthIntensity;
    uniform float depthBias;
    uniform vec2 texelSize;
    uniform float edgeThreshold;
    uniform float edgeFadeWidth;
    uniform float softEdges;  // 0 or 1 - enable edge-aware displacement
    
    varying vec2 vUv;
    varying float vDepth;
    varying float vEdgeFactor;
    varying vec3 vNormal;
    varying vec3 vViewPosition;
    
    // Detect depth discontinuities
    float getEdgeStrength(vec2 uv) {
        if (softEdges < 0.5) return 0.0;
        
        float center = texture2D(depthMap, uv).r;
        
        // Sample 8 neighbors
        float l1 = texture2D(depthMap, uv + vec2(-texelSize.x, 0.0)).r;
        float r1 = texture2D(depthMap, uv + vec2( texelSize.x, 0.0)).r;
        float t1 = texture2D(depthMap, uv + vec2(0.0,  texelSize.y)).r;
        float b1 = texture2D(depthMap, uv + vec2(0.0, -texelSize.y)).r;
        float tl = texture2D(depthMap, uv + vec2(-texelSize.x,  texelSize.y)).r;
        float tr = texture2D(depthMap, uv + vec2( texelSize.x,  texelSize.y)).r;
        float bl = texture2D(depthMap, uv + vec2(-texelSize.x, -texelSize.y)).r;
        float br = texture2D(depthMap, uv + vec2( texelSize.x, -texelSize.y)).r;
        
        // Max depth difference
        float maxDiff = max(
            max(max(abs(center - l1), abs(center - r1)),
                max(abs(center - t1), abs(center - b1))),
            max(max(abs(center - tl), abs(center - tr)),
                max(abs(center - bl), abs(center - br)))
        );
        
        return smoothstep(edgeThreshold, edgeThreshold + edgeFadeWidth, maxDiff);
    }
    
    void main() {
        vUv = uv;
        
        // Sample depth from texture with linear filtering
        float depth = texture2D(depthMap, uv).r;
        vDepth = depth;
        
        // Detect if this vertex is at a depth edge
        float edgeStrength = getEdgeStrength(uv);
        vEdgeFactor = edgeStrength;
        
        // REDUCE displacement at edges to prevent triangle stretching!
        float effectiveIntensity = depthIntensity * (1.0 - edgeStrength * 0.85);
        
        // Displace vertex along Z axis
        vec3 displaced = position;
        displaced.z += (depth - depthBias) * effectiveIntensity;
        
        // Transform normal
        vNormal = normalize(normalMatrix * normal);
        
        // Calculate view position for lighting
        vec4 mvPosition = modelViewMatrix * vec4(displaced, 1.0);
        vViewPosition = -mvPosition.xyz;
        
        gl_Position = projectionMatrix * mvPosition;
    }
`;

/**
 * Edge-aware displacement - REDUCES displacement at depth edges to prevent stretching
 * This is the KEY fix for edge artifacts!
 */
export const EdgeAwareDisplacementVertex = `
    uniform sampler2D depthMap;
    uniform float depthIntensity;
    uniform float depthBias;
    uniform vec2 texelSize;  // 1.0 / texture resolution
    uniform float edgeSoftness;
    uniform float edgeThreshold;
    uniform float edgeFadeWidth;
    
    varying vec2 vUv;
    varying float vDepth;
    varying float vEdgeFactor;  // Pass to fragment for additional fading
    varying vec3 vNormal;
    varying vec3 vViewPosition;
    
    // Detect depth edges and return edge strength (0 = no edge, 1 = strong edge)
    float getEdgeStrength(vec2 uv) {
        float center = texture2D(depthMap, uv).r;
        
        // Sample neighbors at multiple scales for robust edge detection
        float l1 = texture2D(depthMap, uv + vec2(-texelSize.x, 0.0)).r;
        float r1 = texture2D(depthMap, uv + vec2( texelSize.x, 0.0)).r;
        float t1 = texture2D(depthMap, uv + vec2(0.0,  texelSize.y)).r;
        float b1 = texture2D(depthMap, uv + vec2(0.0, -texelSize.y)).r;
        
        // Diagonal neighbors
        float tl = texture2D(depthMap, uv + vec2(-texelSize.x,  texelSize.y)).r;
        float tr = texture2D(depthMap, uv + vec2( texelSize.x,  texelSize.y)).r;
        float bl = texture2D(depthMap, uv + vec2(-texelSize.x, -texelSize.y)).r;
        float br = texture2D(depthMap, uv + vec2( texelSize.x, -texelSize.y)).r;
        
        // 2-pixel radius for larger discontinuities
        float l2 = texture2D(depthMap, uv + vec2(-texelSize.x * 2.0, 0.0)).r;
        float r2 = texture2D(depthMap, uv + vec2( texelSize.x * 2.0, 0.0)).r;
        float t2 = texture2D(depthMap, uv + vec2(0.0,  texelSize.y * 2.0)).r;
        float b2 = texture2D(depthMap, uv + vec2(0.0, -texelSize.y * 2.0)).r;
        
        // Maximum depth difference from center
        float maxDiff = max(
            max(max(abs(center - l1), abs(center - r1)),
                max(abs(center - t1), abs(center - b1))),
            max(max(abs(center - tl), abs(center - tr)),
                max(abs(center - bl), abs(center - br)))
        );
        
        // Include larger radius
        float maxDiff2 = max(
            max(abs(center - l2), abs(center - r2)),
            max(abs(center - t2), abs(center - b2))
        );
        maxDiff = max(maxDiff, maxDiff2 * 0.7);
        
        return smoothstep(edgeThreshold, edgeThreshold + edgeFadeWidth, maxDiff);
    }
    
    void main() {
        vUv = uv;
        
        float centerDepth = texture2D(depthMap, uv).r;
        vDepth = centerDepth;
        
        // Detect if this vertex is at a depth edge
        float edgeStrength = getEdgeStrength(uv);
        vEdgeFactor = edgeStrength;
        
        // REDUCE displacement at edges - this prevents triangle stretching!
        // At strong edges, pull displacement toward neutral (depthBias)
        float effectiveIntensity = depthIntensity * (1.0 - edgeStrength * 0.8);
        
        // Displace vertex with reduced intensity at edges
        vec3 displaced = position;
        displaced.z += (centerDepth - depthBias) * effectiveIntensity;
        
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
 * Basic textured output - with optional edge fading from vertex shader
 */
export const BasicFragment = `
    uniform sampler2D imageMap;
    uniform float opacity;
    uniform float softEdges;
    
    varying vec2 vUv;
    varying float vDepth;
    varying float vEdgeFactor;
    
    void main() {
        vec4 texColor = texture2D(imageMap, vUv);
        
        // Additional alpha fade at edges (from vertex-calculated edge factor)
        float edgeAlpha = softEdges > 0.5 ? (1.0 - vEdgeFactor * 0.7) : 1.0;
        
        gl_FragColor = vec4(texColor.rgb, texColor.a * opacity * edgeAlpha);
    }
`;

/**
 * Edge-aware fragment - fades out pixels at depth discontinuities
 * This is the KEY to hiding stretching artifacts!
 */
export const EdgeAwareFragment = `
    uniform sampler2D imageMap;
    uniform sampler2D depthMap;
    uniform vec2 texelSize;
    uniform float opacity;
    uniform float edgeThreshold;    // Depth difference that triggers fade (0.02-0.15)
    uniform float edgeFadeWidth;    // How gradual the fade is (0.05-0.3)
    
    varying vec2 vUv;
    varying float vDepth;
    
    void main() {
        vec4 texColor = texture2D(imageMap, vUv);
        float centerDepth = texture2D(depthMap, vUv).r;
        
        // Sample depth in a larger neighborhood for robust edge detection
        // Use 2-pixel radius for better coverage of stretched triangles
        float l1  = texture2D(depthMap, vUv + vec2(-texelSize.x, 0.0)).r;
        float r1  = texture2D(depthMap, vUv + vec2( texelSize.x, 0.0)).r;
        float t1  = texture2D(depthMap, vUv + vec2(0.0,  texelSize.y)).r;
        float b1  = texture2D(depthMap, vUv + vec2(0.0, -texelSize.y)).r;
        
        float l2  = texture2D(depthMap, vUv + vec2(-texelSize.x * 2.0, 0.0)).r;
        float r2  = texture2D(depthMap, vUv + vec2( texelSize.x * 2.0, 0.0)).r;
        float t2  = texture2D(depthMap, vUv + vec2(0.0,  texelSize.y * 2.0)).r;
        float b2  = texture2D(depthMap, vUv + vec2(0.0, -texelSize.y * 2.0)).r;
        
        // Diagonal samples
        float tl = texture2D(depthMap, vUv + vec2(-texelSize.x,  texelSize.y)).r;
        float tr = texture2D(depthMap, vUv + vec2( texelSize.x,  texelSize.y)).r;
        float bl = texture2D(depthMap, vUv + vec2(-texelSize.x, -texelSize.y)).r;
        float br = texture2D(depthMap, vUv + vec2( texelSize.x, -texelSize.y)).r;
        
        // Calculate maximum depth difference from center (edges)
        float maxDiff = max(
            max(max(abs(centerDepth - l1), abs(centerDepth - r1)),
                max(abs(centerDepth - t1), abs(centerDepth - b1))),
            max(max(abs(centerDepth - tl), abs(centerDepth - tr)),
                max(abs(centerDepth - bl), abs(centerDepth - br)))
        );
        
        // Also check 2-pixel neighbors for larger discontinuities
        float maxDiff2 = max(
            max(abs(centerDepth - l2), abs(centerDepth - r2)),
            max(abs(centerDepth - t2), abs(centerDepth - b2))
        );
        maxDiff = max(maxDiff, maxDiff2 * 0.7);
        
        // Smooth fade based on edge strength
        float edgeFactor = smoothstep(edgeThreshold, edgeThreshold + edgeFadeWidth, maxDiff);
        float alpha = (1.0 - edgeFactor) * opacity;
        
        gl_FragColor = vec4(texColor.rgb, texColor.a * alpha);
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
    
    // Edge-aware mode - fades out depth discontinuities to hide artifacts
    edgeAware: {
        vertex: BasicDisplacementVertex,
        fragment: EdgeAwareFragment,
        uniforms: ['depthMap', 'depthIntensity', 'depthBias', 'imageMap', 'opacity', 
                   'texelSize', 'edgeThreshold', 'edgeFadeWidth']
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
