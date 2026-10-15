Shader "SciFi/HolographicHand"
{
    Properties
    {
        [HDR] _HoloColor ("Hologram Color", Color) = (0.1, 0.85, 1.0, 0.8)
        [HDR] _RimColor ("Rim Glow Color", Color) = (0.4, 0.95, 1.0, 1.0)
        _RimPower ("Rim Power", Range(0.5, 8.0)) = 2.5
        _BaseAlpha ("Base Alpha", Range(0.0, 1.0)) = 0.25
        _ScanlineSpeed ("Scanline Speed", Float) = 4.0
        _ScanlineTiling ("Scanline Tiling", Float) = 80.0
        _PulseSpeed ("Pulse Speed", Float) = 2.5
        _PulseIntensity ("Pulse Intensity", Range(0.0, 0.5)) = 0.15
    }

    SubShader
    {
        Tags 
        { 
            "Queue"="Transparent+100" 
            "RenderType"="Transparent" 
            "IgnoreProjector"="True" 
        }

        LOD 200
        Blend SrcAlpha One // Additive hologram glow
        ZWrite Off
        Cull Back

        Pass
        {
            Name "HologramPass"

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float3 worldNormal : TEXCOORD0;
                float3 viewDir : TEXCOORD1;
                float4 worldPos : TEXCOORD2;
            };

            fixed4 _HoloColor;
            fixed4 _RimColor;
            float _RimPower;
            float _BaseAlpha;
            float _ScanlineSpeed;
            float _ScanlineTiling;
            float _PulseSpeed;
            float _PulseIntensity;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.worldPos = mul(unity_ObjectToWorld, v.vertex);
                o.worldNormal = UnityObjectToWorldNormal(v.normal);
                o.viewDir = normalize(WorldSpaceViewDir(v.vertex));
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float3 normal = normalize(i.worldNormal);
                float3 viewDir = normalize(i.viewDir);

                // Fresnel / Rim glow (edges are brighter)
                float NdotV = max(0.0, dot(normal, viewDir));
                float rim = pow(1.0 - NdotV, _RimPower);

                // Animated scanline wave
                float scanline = sin((i.worldPos.y * _ScanlineTiling) + (_Time.y * _ScanlineSpeed));
                scanline = clamp(scanline * 0.5 + 0.5, 0.3, 1.0);

                // Subtle organic energy pulse
                float pulse = 1.0 + sin(_Time.y * _PulseSpeed) * _PulseIntensity;

                // Combine base color and edge rim
                fixed3 finalColor = (_HoloColor.rgb * _BaseAlpha + _RimColor.rgb * rim * 1.5) * scanline * pulse;
                float finalAlpha = clamp(_BaseAlpha + rim * 0.8, 0.0, 1.0);

                return fixed4(finalColor, finalAlpha);
            }
            ENDCG
        }
    }

    FallBack "Unlit/Transparent"
}
