Shader "ARCore/DepthVisualizer"
{
    Properties
    {
        _MainTex ("Depth Texture", 2D) = "black" {}
        _MinDistance ("Min Distance (Meters)", Float) = 0.2
        _MaxDistance ("Max Distance (Meters)", Float) = 4.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Transparent" }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float _MinDistance;
            float _MaxDistance;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            // Simple Turbo-style color map for depth values [0, 1]
            fixed4 DepthColorMap(float t)
            {
                t = clamp(t, 0.0, 1.0);
                fixed3 c;
                c.r = saturate(1.5 - abs(t * 3.0 - 2.0));
                c.g = saturate(1.5 - abs(t * 3.0 - 1.0));
                c.b = saturate(1.5 - abs(t * 3.0 - 0.0));
                return fixed4(c, 1.0);
            }

            fixed4 frag (v2f i) : SV_Target
            {
                // Sample metric depth from ARCore environment depth texture
                float depthMeters = tex2D(_MainTex, i.uv).r;

                if (depthMeters <= 0.001)
                {
                    // Invalid/unknown depth
                    return fixed4(0.05, 0.05, 0.05, 1.0);
                }

                // Normalize depth based on min/max distances
                float normalized = saturate((depthMeters - _MinDistance) / max(0.001, _MaxDistance - _MinDistance));
                return DepthColorMap(normalized);
            }
            ENDCG
        }
    }
}
