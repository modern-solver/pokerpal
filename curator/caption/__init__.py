"""CaptionAgent package — platform-specific caption/bio/prompt/post generation.

OWNER: Agent A4 (Stage S-04). Loads templates from config/prompts.yml, switches
output_mode by platform (post_caption / dating_bio / prompt_answer / linkedin_post),
enforces banned-phrase rejection, the Hinge prompt library, length caps, and tone
selection via Groq Llama 3.3 70B. Stub only at S-00.
"""
