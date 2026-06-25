"""GradingAgent package — three-tier grading pipeline.

OWNERS:
    Agent A2 (Stage S-02) — Tier 1 local pre-filter (Pillow resolution, OpenCV
        Laplacian blur, imagehash dedup) + Tier 2 platform-fit scoring from
        config/platform_weights.yml + dating face-gate + LinkedIn suitability flag.
    Agent A3 (Stage S-03) — Tier 3 semantic grading (Groq Llama 4 Scout vision),
        BLIP-2 scene description, JSON parse + retry + HF Serverless 429 fallback,
        and composite aggregation.

Stub only at S-00.
"""
