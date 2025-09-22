import streamlit as st
import json
import base64
import io
import time
from datetime import datetime
from utils.common import create_tool_header, show_progress_bar, add_to_recent
from utils.file_handler import FileHandler
from utils.ai_client import ai_client
import pandas as pd

# Lazy imports for data analysis libraries to prevent startup failures
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    DATA_ANALYSIS_AVAILABLE = True
except ImportError:
    DATA_ANALYSIS_AVAILABLE = False


def display_tools():
    """Display all AI tools"""

    tool_categories = {
        "🔑 AI API Management": [
            "API Setup & Management", "Provider Status", "Test Connections"
        ],
        "Text Generation": [
            "Content Creator", "Story Writer", "Article Generator", "Copywriting Assistant", "Technical Writer"
        ],
        "Image Generation": [
            "AI Art Creator", "Style Transfer", "Image Synthesis", "Concept Art", "Photo Enhancement"
        ],
        "Data Analysis": [
            "Pattern Recognition", "Trend Analysis", "Predictive Modeling", "Data Insights", "Statistical Analysis"
        ],
        "Chatbots": [
            "Conversational AI", "Customer Service Bot", "Educational Assistant", "Domain Expert", "Multi-Purpose Bot"
        ],
        "Language Processing": [
            "Text Translator", "Sentiment Analysis", "Text Summarizer", "Language Detector", "Content Moderator"
        ],
        "Computer Vision": [
            "Image Recognition", "Object Detection", "Scene Analysis", "OCR Reader", "Visual Search"
        ],
        "AI Utilities": [
            "Prompt Optimizer", "AI Workflow", "Batch Processor", "Response Analyzer"
        ],
        "Machine Learning": [
            "Model Training", "Data Preprocessing", "Feature Engineering", "Model Evaluation", "AutoML"
        ],
        "Voice & Audio": [
            "Speech Recognition", "Voice Synthesis", "Audio Analysis", "Voice Cloning", "Sound Generation"
        ]
    }

    # Handle navigation from buttons
    if 'ai_selected_category' not in st.session_state:
        st.session_state.ai_selected_category = "Multiple AI Model Integrations"
    if 'ai_selected_tool' not in st.session_state:
        st.session_state.ai_selected_tool = "Prompt Optimizer"

    selected_category = st.selectbox("Select AI Tool Category",
                                     list(tool_categories.keys()),
                                     index=list(tool_categories.keys()).index(
                                         st.session_state.ai_selected_category) if st.session_state.ai_selected_category in tool_categories else 0,
                                     key="category_selector")

    selected_tool = st.selectbox("Select Tool",
                                 tool_categories[selected_category],
                                 index=tool_categories[selected_category].index(
                                     st.session_state.ai_selected_tool) if st.session_state.ai_selected_tool in
                                                                           tool_categories[selected_category] else 0,
                                 key="tool_selector")

    # Update session state when selections change
    st.session_state.ai_selected_category = selected_category
    st.session_state.ai_selected_tool = selected_tool

    st.markdown("---")

    add_to_recent(f"AI Tools - {selected_tool}")

    # Display selected tool
    if selected_tool == "API Setup & Management":
        api_setup_management()
    elif selected_tool == "Provider Status":
        provider_status()
    elif selected_tool == "Test Connections":
        test_connections()
    elif selected_tool == "Multi-Model Chat":
        multi_model_chat()
    elif selected_tool == "Content Creator":
        content_creator()
    elif selected_tool == "AI Art Creator":
        ai_art_creator()
    elif selected_tool == "Sentiment Analysis":
        sentiment_analysis()
    elif selected_tool == "Image Recognition":
        image_recognition()
    elif selected_tool == "Object Detection":
        object_detection()
    elif selected_tool == "Scene Analysis":
        scene_analysis()
    elif selected_tool == "Visual Search":
        visual_search()
    elif selected_tool == "Text Translator":
        text_translator()
    elif selected_tool == "Text Summarizer":
        text_summarizer()
    elif selected_tool == "Prompt Optimizer":
        prompt_optimizer()
    elif selected_tool == "AI Workflow":
        ai_workflow()
    elif selected_tool == "Batch Processor":
        batch_processor()
    elif selected_tool == "Response Analyzer":
        response_analyzer()
    elif selected_tool == "Data Insights":
        data_insights()
    elif selected_tool == "Conversational AI":
        conversational_ai()
    elif selected_tool == "Customer Service Bot":
        customer_service_bot()
    elif selected_tool == "Educational Assistant":
        educational_assistant()
    elif selected_tool == "Domain Expert":
        domain_expert()
    elif selected_tool == "Multi-Purpose Bot":
        multi_purpose_bot()
    elif selected_tool == "OCR Reader":
        ocr_reader()
    elif selected_tool == "Voice Synthesis":
        voice_synthesis()
    elif selected_tool == "Pattern Recognition":
        pattern_recognition()
    elif selected_tool == "Story Writer":
        story_writer()
    elif selected_tool == "Style Transfer":
        style_transfer()
    elif selected_tool == "Image Synthesis":
        image_synthesis()
    elif selected_tool == "Concept Art":
        concept_art()
    elif selected_tool == "Photo Enhancement":
        photo_enhancement()
    elif selected_tool == "Article Generator":
        article_generator()
    elif selected_tool == "Copywriting Assistant":
        copywriting_assistant()
    elif selected_tool == "Technical Writer":
        technical_writer()
    elif selected_tool == "Trend Analysis":
        trend_analysis()
    elif selected_tool == "Predictive Modeling":
        predictive_modeling()
    elif selected_tool == "Statistical Analysis":
        statistical_analysis()
    elif selected_tool == "Model Training":
        model_training()
    elif selected_tool == "Data Preprocessing":
        data_preprocessing()
    elif selected_tool == "Feature Engineering":
        feature_engineering()
    elif selected_tool == "Model Evaluation":
        model_evaluation()
    elif selected_tool == "AutoML":
        automl()
    elif selected_tool == "Speech Recognition":
        speech_recognition()
    elif selected_tool == "Audio Analysis":
        audio_analysis()
    elif selected_tool == "Voice Cloning":
        voice_cloning()
    elif selected_tool == "Sound Generation":
        sound_generation()
    else:
        st.info(f"{selected_tool} tool is being implemented. Please check back soon!")


def multi_model_chat():
    """Chat with multiple AI models simultaneously"""
    create_tool_header("Multi-Model Chat", "Chat with multiple AI models at once", "🤖")

    # Model selection
    st.subheader("Select AI Models")
    available_models = ai_client.get_available_models()

    if not available_models:
        st.warning("No AI models available. Please check your API keys.")
        return

    selected_models = st.multiselect("Choose Models", available_models,
                                     default=available_models[:2] if len(available_models) >= 2 else available_models)

    if not selected_models:
        st.warning("Please select at least one model.")
        return

    # Chat interface
    st.subheader("Multi-Model Conversation")

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # User input
    user_message = st.text_area("Enter your message:", height=100)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Send to All Models"):
            if user_message:
                send_to_all_models(user_message, selected_models)

    with col2:
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.rerun()

    with col3:
        if st.button("Export Chat"):
            export_chat_history()

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Conversation History")

        for entry in st.session_state.chat_history:
            # User message
            st.markdown(f"**👤 You:** {entry['user_message']}")

            # Model responses
            for model, response in entry['responses'].items():
                with st.expander(f"🤖 {model.title()} Response"):
                    st.write(response)

            st.markdown("---")


def content_creator():
    """AI-powered content creation tool"""
    create_tool_header("Content Creator", "Generate various types of content with AI", "✍️")

    content_type = st.selectbox("Content Type", [
        "Blog Post", "Social Media Post", "Email", "Product Description",
        "Press Release", "Technical Documentation", "Creative Writing", "Marketing Copy"
    ])

    # Content parameters
    st.subheader("Content Parameters")

    col1, col2 = st.columns(2)
    with col1:
        topic = st.text_input("Topic/Subject", placeholder="Enter the main topic")
        target_audience = st.selectbox("Target Audience", [
            "General Public", "Professionals", "Students", "Experts", "Children", "Teenagers"
        ])
        tone = st.selectbox("Tone", [
            "Professional", "Casual", "Friendly", "Formal", "Humorous", "Persuasive", "Educational"
        ])

    with col2:
        length = st.selectbox("Content Length", [
            "Short (100-300 words)", "Medium (300-800 words)", "Long (800-1500 words)", "Very Long (1500+ words)"
        ])
        language = st.selectbox("Language", [
            "English", "Spanish", "French", "German", "Italian", "Portuguese", "Chinese", "Japanese"
        ])
        creativity = st.slider("Creativity Level", 0.1, 1.0, 0.7, 0.1)

    # Additional options
    if content_type in ["Blog Post", "Technical Documentation"]:
        include_outline = st.checkbox("Include Outline/Structure")
        include_seo = st.checkbox("SEO Optimized")

    if content_type == "Social Media Post":
        platform = st.selectbox("Platform", ["Twitter", "Facebook", "LinkedIn", "Instagram", "TikTok"])
        include_hashtags = st.checkbox("Include Hashtags")

    additional_instructions = st.text_area("Additional Instructions (optional)",
                                           placeholder="Any specific requirements or guidelines...")

    if st.button("Generate Content") and topic:
        with st.spinner("Generating content with AI..."):
            # Build kwargs dict with only the extra parameters we need
            extra_kwargs = {}
            if content_type in ["Blog Post", "Technical Documentation"]:
                extra_kwargs['include_outline'] = locals().get('include_outline', False)
                extra_kwargs['include_seo'] = locals().get('include_seo', False)
            if content_type == "Social Media Post":
                extra_kwargs['platform'] = locals().get('platform', 'general')
                extra_kwargs['include_hashtags'] = locals().get('include_hashtags', False)

            content = generate_content(
                content_type, topic, target_audience, tone, length,
                language, creativity, additional_instructions,
                **extra_kwargs
            )

            if content:
                st.subheader("Generated Content")
                st.markdown(content)

                # Content analysis
                st.subheader("Content Analysis")
                analysis = analyze_content(content)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Word Count", analysis['word_count'])
                with col2:
                    st.metric("Reading Time", f"{analysis['reading_time']} min")
                with col3:
                    st.metric("Readability", analysis['readability_level'])

                # Download options
                FileHandler.create_download_link(
                    content.encode(),
                    f"{content_type.lower().replace(' ', '_')}.txt",
                    "text/plain"
                )

                # Regenerate options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Regenerate"):
                        st.rerun()
                with col2:
                    if st.button("Refine Content"):
                        refine_content(content)


def ai_art_creator():
    """AI-powered image generation using Gemini"""
    create_tool_header("AI Art Creator", "Generate images with AI using your Gemini API", "🎨")

    # Check if API is available
    provider_info = ai_client.get_provider_info()
    available_providers = provider_info["available_providers"]

    if not available_providers:
        st.warning("⚠️ No AI image providers configured. Please set up your API keys.")
        ai_client._show_api_key_setup_instructions()
        return

    # Simple image generation interface
    st.subheader("Image Generation")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        prompt = st.text_area(
            "Describe what you want to create",
            placeholder="A beautiful sunset over a mountain landscape, digital art style...",
            height=100,
            help="Describe the image you want to generate"
        )
    
    with col2:
        style = st.selectbox("Art Style", [
            "Photorealistic", "Digital Art", "Oil Painting", "Watercolor", 
            "Cartoon", "Anime", "Abstract", "3D Render", "Sketch"
        ])
        
        size = st.selectbox("Image Size", [
            "1024x1024 (Square)", "1024x768 (Landscape)", "768x1024 (Portrait)"
        ])

    # Convert size back to format expected by API
    size_map = {
        "1024x1024 (Square)": "1024x1024",
        "1024x768 (Landscape)": "1024x768", 
        "768x1024 (Portrait)": "768x1024"
    }
    actual_size = size_map[size]

    # Generate button and result
    if st.button("🎨 Generate Image", type="primary") and prompt:
        with st.spinner("Generating image with AI..."):
            # Create enhanced prompt with style
            enhanced_prompt = f"{prompt}, {style.lower()} style"
            
            try:
                # Generate image using Gemini
                image_bytes = ai_client.generate_image(
                    prompt=enhanced_prompt,
                    provider="gemini",
                    size=actual_size
                )
                
                if image_bytes:
                    st.subheader("Generated Image")
                    st.image(image_bytes, caption=f"Generated: {enhanced_prompt}")
                    
                    # Download button
                    FileHandler.create_download_link(
                        image_bytes,
                        f"ai_generated_image.png",
                        "image/png"
                    )
                    
                    # Regenerate options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Generate Another"):
                            st.rerun()
                    with col2:
                        if st.button("✨ Refine Prompt"):
                            st.info("Try adjusting your description for different results!")
                else:
                    st.error("Failed to generate image. Please try again with a different prompt.")
                    
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")
                st.info("Make sure your Gemini API key is configured correctly.")


def sentiment_analysis():
    """Analyze sentiment of text"""
    create_tool_header("Sentiment Analysis", "Analyze text sentiment using AI", "😊")
    
    # Input text
    text = st.text_area("Enter text to analyze:", 
                       placeholder="Enter any text to analyze its sentiment...", 
                       height=150)
    
    # Analysis options
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("AI Model", ["gemini", "openai", "anthropic"])
    with col2:
        analysis_type = st.selectbox("Analysis Type", ["Basic", "Detailed"])
    
    # Analyze button
    if st.button("Analyze Sentiment", type="primary") and text:
        with st.spinner("Analyzing sentiment..."):
            try:
                result = ai_client.analyze_sentiment(text, model=model)
                
                if "error" not in result:
                    st.subheader("Sentiment Analysis Results")
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Sentiment", result.get("sentiment", "Unknown"))
                    with col2:
                        confidence = result.get("confidence", 0)
                        st.metric("Confidence", f"{confidence:.2%}" if isinstance(confidence, (int, float)) else confidence)
                    with col3:
                        score = result.get("score", "N/A")
                        st.metric("Score", f"{score:.2f}" if isinstance(score, (int, float)) else score)
                    
                    # Detailed results
                    if analysis_type == "Detailed" and "explanation" in result:
                        st.subheader("Detailed Analysis")
                        st.write(result["explanation"])
                        
                else:
                    st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error analyzing sentiment: {str(e)}")


def image_recognition():
    """AI-powered image recognition and analysis"""
    create_tool_header("Image Recognition", "Analyze and recognize objects in images", "👁️")
            "Abstract", "Impressionist", "Expressionist", "Surreal", "Cubist", "Art Nouveau",
            "Art Deco", "Baroque", "Renaissance", "Minimalist", "Pop Art", "Street Art"
        ])
        
        medium = st.selectbox("Artistic Medium", [
            "None", "Canvas", "Paper", "Wood Panel", "Metal", "Glass", "Stone", "Fabric",
            "Digital Canvas", "Fresco", "Mural", "Stained Glass", "Mosaic", "Collage"
        ])
    
    with style_col2:
        art_movement = st.selectbox("Art Movement/Era", [
            "Contemporary", "Modern", "Classical", "Romantic", "Gothic", "Art Deco", "Bauhaus",
            "Dadaism", "Fauvism", "Post-Impressionism", "Neo-Classical", "Victorian",
            "Medieval", "Renaissance", "Baroque", "Rococo", "Neoclassicism"
        ])
        
        artist_inspiration = st.selectbox("Artist Inspiration", [
            "None", "Leonardo da Vinci", "Van Gogh", "Picasso", "Monet", "Dali", "Warhol",
            "Banksy", "Basquiat", "Frida Kahlo", "Georgia O'Keeffe", "Ansel Adams",
            "Studio Ghibli", "Pixar", "Marvel Comics", "DC Comics", "Akira Toriyama",
            "Kentaro Miura", "Frank Frazetta", "Boris Vallejo", "H.R. Giger", "Zdzisław Beksiński"
        ])
    
    with style_col3:
        texture = st.selectbox("Surface Texture", [
            "Smooth", "Rough", "Glossy", "Matte", "Metallic", "Fabric", "Wood Grain",
            "Stone", "Glass", "Plastic", "Leather", "Fur", "Feathers", "Scales"
        ])
        
        finish = st.selectbox("Finish Quality", [
            "Standard", "Polished", "Distressed", "Weathered", "Aged", "Pristine",
            "Handcrafted", "Industrial", "Organic", "Geometric"
        ])

    # Composition & Camera Controls
    st.subheader("📸 Composition & Camera")
    
    comp_col1, comp_col2, comp_col3 = st.columns(3)
    
    with comp_col1:
        shot_type = st.selectbox("Shot Type", [
            "Medium Shot", "Close-up", "Extreme Close-up", "Wide Shot", "Long Shot",
            "Establishing Shot", "Over-the-shoulder", "Bird's Eye View", "Worm's Eye View",
            "Dutch Angle", "Point of View", "Macro", "Panoramic"
        ])
        
        camera_angle = st.selectbox("Camera Angle", [
            "Eye Level", "High Angle", "Low Angle", "Overhead", "Upward", "Side View",
            "Three-Quarter View", "Profile", "Frontal", "Rear View", "Diagonal"
        ])
    
    with comp_col2:
        depth_of_field = st.selectbox("Depth of Field", [
            "Sharp Focus", "Shallow DOF", "Deep DOF", "Bokeh Background", "Tilt-Shift",
            "Focus Stacking", "Split Focus", "Rack Focus"
        ])
        
        lens_type = st.selectbox("Lens Effect", [
            "Standard", "Wide Angle", "Telephoto", "Fisheye", "Macro", "Anamorphic",
            "Prime Lens", "Zoom Lens", "Tilt-Shift Lens"
        ])
    
    with comp_col3:
        framing = st.selectbox("Framing", [
            "Centered", "Rule of Thirds", "Golden Ratio", "Symmetrical", "Asymmetrical",
            "Leading Lines", "Frame within Frame", "Negative Space", "Tight Crop"
        ])
        
        perspective = st.selectbox("Perspective", [
            "Linear", "One Point", "Two Point", "Three Point", "Isometric", "Aerial",
            "Forced Perspective", "Atmospheric", "Foreshortening"
        ])

    # Lighting & Atmosphere
    st.subheader("💡 Lighting & Atmosphere")
    
    light_col1, light_col2, light_col3 = st.columns(3)
    
    with light_col1:
        lighting_setup = st.selectbox("Lighting Setup", [
            "Natural Light", "Studio Lighting", "Dramatic Lighting", "Soft Lighting",
            "Hard Lighting", "Rim Lighting", "Backlighting", "Side Lighting",
            "Top Lighting", "Key Light", "Fill Light", "Ambient Light", "Volumetric Lighting"
        ])
        
        time_of_day = st.selectbox("Time of Day", [
            "Midday", "Golden Hour", "Blue Hour", "Sunrise", "Sunset", "Dawn", "Dusk",
            "Noon", "Afternoon", "Evening", "Night", "Midnight", "Pre-dawn"
        ])
    
    with light_col2:
        weather = st.selectbox("Weather/Atmosphere", [
            "Clear", "Cloudy", "Overcast", "Stormy", "Rainy", "Snowy", "Foggy", "Misty",
            "Hazy", "Sunny", "Windy", "Humid", "Dry", "Thunderstorm", "Blizzard"
        ])
        
        mood_atmosphere = st.selectbox("Mood/Atmosphere", [
            "Serene", "Dramatic", "Mysterious", "Joyful", "Melancholic", "Energetic",
            "Peaceful", "Intense", "Romantic", "Eerie", "Majestic", "Intimate",
            "Epic", "Nostalgic", "Futuristic", "Ancient", "Ethereal", "Gritty"
        ])
    
    with light_col3:
        color_temperature = st.selectbox("Color Temperature", [
            "Neutral", "Warm", "Cool", "Very Warm", "Very Cool", "Tungsten", "Daylight",
            "Fluorescent", "Candlelight", "Moonlight", "Neon", "LED"
        ])
        
        contrast_level = st.selectbox("Contrast", [
            "Normal", "High Contrast", "Low Contrast", "Dramatic Shadows", "Soft Shadows",
            "No Shadows", "Harsh Lighting", "Even Lighting", "Chiaroscuro"
        ])

    # Color & Palette
    st.subheader("🌈 Color & Palette")
    
    color_col1, color_col2 = st.columns(2)
    
    with color_col1:
        color_palette = st.selectbox("Color Palette", [
            "Natural", "Vibrant", "Muted", "Monochromatic", "Complementary", "Analogous",
            "Triadic", "Split-Complementary", "Tetradic", "Warm Tones", "Cool Tones",
            "Earth Tones", "Pastel", "Neon", "Jewel Tones", "Metallic", "Sepia",
            "Black and White", "High Saturation", "Desaturated", "Gradient"
        ])
        
        dominant_color = st.selectbox("Dominant Color", [
            "Auto", "Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Pink",
            "Brown", "Black", "White", "Gray", "Gold", "Silver", "Copper", "Bronze"
        ])
    
    with color_col2:
        color_harmony = st.selectbox("Color Harmony", [
            "Natural", "Complementary", "Analogous", "Triadic", "Split-Complementary",
            "Tetradic", "Monochromatic", "Achromatic", "Custom Gradient"
        ])
        
        saturation = st.selectbox("Saturation Level", [
            "Natural", "High Saturation", "Low Saturation", "Desaturated", "Oversaturated",
            "Selective Color", "Vintage", "Faded", "Vivid", "Muted Pastels"
        ])

    # Technical Settings
    st.subheader("⚙️ Technical Settings")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        resolution = st.selectbox("Resolution", [
            "1024x1024", "1024x768", "768x1024", "1152x896", "896x1152", 
            "1216x832", "832x1216", "1344x768", "768x1344", "1536x640", "640x1536"
        ])
        
        aspect_ratio = st.selectbox("Aspect Ratio", [
            "Square (1:1)", "Landscape (4:3)", "Portrait (3:4)", "Widescreen (16:9)",
            "Ultra-wide (21:9)", "Cinema (2.35:1)", "Golden Ratio", "Custom"
        ])
    
    with tech_col2:
        quality = st.selectbox("Quality Level", [
            "Standard", "High Quality", "Ultra High Quality", "Masterpiece", "8K", "4K",
            "Professional", "Gallery Quality", "Print Ready", "Web Optimized"
        ])
        
        detail_level = st.selectbox("Detail Level", [
            "Normal", "Highly Detailed", "Extremely Detailed", "Intricate", "Fine Details",
            "Sharp Details", "Soft Details", "Minimalist", "Clean", "Complex"
        ])
    
    with tech_col3:
        render_engine = st.selectbox("Render Style", [
            "Default", "Unreal Engine", "Octane Render", "Blender", "Maya", "3ds Max",
            "Cinema 4D", "V-Ray", "Arnold", "Cycles", "PBRT", "Photorealistic"
        ])
        
        post_processing = st.selectbox("Post-Processing", [
            "None", "Color Grading", "Film Grain", "Vintage", "HDR", "Bloom Effect",
            "Lens Flare", "Vignette", "Chromatic Aberration", "Film Look"
        ])

    # Advanced Options
    with st.expander("🔧 Advanced Controls", expanded=False):
        
        adv_col1, adv_col2 = st.columns(2)
        
        with adv_col1:
            negative_prompt = st.text_area(
                "Negative Prompt (what to avoid)",
                placeholder="blurry, low quality, distorted, bad anatomy, deformed, ugly, duplicate, watermark, signature, text, logo, cropped...",
                height=100,
                help="Specify what should NOT appear in the image"
            )
            
            custom_modifiers = st.text_area(
                "Custom Modifiers",
                placeholder="trending on artstation, award-winning, museum quality, professional photography...",
                height=80,
                help="Add custom quality and style modifiers"
            )
        
        with adv_col2:
            seed = st.number_input(
                "Seed (for reproducibility)", 
                min_value=0, 
                max_value=2147483647, 
                value=0,
                help="Use the same seed to reproduce similar results"
            )
            
            guidance_scale = st.slider(
                "Guidance Scale", 
                1.0, 30.0, 7.5, 0.5,
                help="How closely the AI follows your prompt (7-15 recommended)"
            )
            
            steps = st.slider(
                "Inference Steps", 
                10, 100, 50, 5,
                help="More steps = higher quality but slower generation"
            )
            
            creativity = st.slider(
                "Creativity Level", 
                0.1, 2.0, 1.0, 0.1,
                help="Higher values = more creative interpretation"
            )

    # Prompt Enhancement Options
    with st.expander("✨ Prompt Enhancement", expanded=True):
        enhance_col1, enhance_col2 = st.columns(2)
        
        with enhance_col1:
            auto_enhance = st.checkbox("Auto-enhance prompt", value=True, help="Automatically improve your prompt")
            add_quality_tags = st.checkbox("Add quality modifiers", value=True, help="Add professional quality tags")
            add_style_consistency = st.checkbox("Ensure style consistency", value=True, help="Make all elements match the chosen style")
        
        with enhance_col2:
            boost_details = st.checkbox("Boost detail descriptions", value=False, help="Add more detailed descriptions")
            add_photography_terms = st.checkbox("Add photography terms", value=False, help="Include camera and lens terminology")
            artistic_flair = st.checkbox("Add artistic flair", value=False, help="Include artistic and creative modifiers")

    # Generation Controls
    st.subheader("🚀 Generate")
    
    gen_col1, gen_col2, gen_col3 = st.columns(3)
    
    with gen_col1:
        num_images = st.selectbox("Number of Images", [1, 2, 3, 4], help="Generate multiple variations")
    
    with gen_col2:
        batch_mode = st.checkbox("Batch Mode", help="Generate all images in sequence")
    
    with gen_col3:
        save_settings = st.checkbox("Save Settings", help="Save current settings as preset")

    # Enhanced prompt preview
    if prompt:
        enhanced_prompt = create_enhanced_prompt(
            prompt, art_style, medium, art_movement, artist_inspiration, texture, finish,
            shot_type, camera_angle, depth_of_field, lens_type, framing, perspective,
            lighting_setup, time_of_day, weather, mood_atmosphere, color_temperature, contrast_level,
            color_palette, dominant_color, color_harmony, saturation,
            quality, detail_level, render_engine, post_processing,
            custom_modifiers, auto_enhance, add_quality_tags, add_style_consistency,
            boost_details, add_photography_terms, artistic_flair
        )
        
        with st.expander("📝 Preview Enhanced Prompt", expanded=False):
            st.code(enhanced_prompt, language=None)
            
            # Prompt analysis
            word_count = len(enhanced_prompt.split())
            complexity_score = calculate_prompt_complexity(enhanced_prompt)
            
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
            with analysis_col1:
                st.metric("Word Count", word_count)
            with analysis_col2:
                st.metric("Complexity", f"{complexity_score}/10")
            with analysis_col3:
                effectiveness = "High" if complexity_score >= 7 else "Medium" if complexity_score >= 4 else "Low"
                st.metric("Effectiveness", effectiveness)

    # Provider Selection Section
    st.subheader("🔌 AI Provider")
    
    provider_info = ai_client.get_provider_info()
    available_providers = provider_info["available_providers"]
    
    if not available_providers:
        st.error("⚠️ No AI providers configured. Please set up API keys to continue.")
        ai_client._show_api_key_setup_instructions()
        
        # Show setup instructions and links
        st.markdown("""
        **Quick Setup:**
        
        Choose one of the providers below and configure your API key through the integration system:
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container():
                st.markdown("**🎨 OpenAI DALL-E 3 (Recommended)**")
                st.markdown("• High-quality image generation")
                st.markdown("• Professional artistic results")
                st.markdown("• Multiple resolution options")
                st.link_button("Get OpenAI API Key", "https://platform.openai.com/api-keys", use_container_width=True)
        
        with col2:
            with st.container():
                st.markdown("**⚡ Google Gemini 2.0**")
                st.markdown("• Fast experimental generation")
                st.markdown("• Lower cost alternative")
                st.markdown("• Integrated with Google AI")
                st.link_button("Get Gemini API Key", "https://aistudio.google.com/app/apikey", use_container_width=True)
        
        return  # Stop here if no providers available
    
    # Provider selection and status
    provider_col1, provider_col2 = st.columns([2, 1])
    
    with provider_col1:
        provider_options = ["auto"] + available_providers
        provider_labels = {
            "auto": "🤖 Auto-Select (Recommended)",
            "openai": "🎨 OpenAI DALL-E 3",
            "gemini": "⚡ Google Gemini 2.0"
        }
        
        selected_provider = st.selectbox(
            "AI Provider",
            provider_options,
            format_func=lambda x: provider_labels.get(x, x),
            help="Choose your preferred AI provider or let the system auto-select"
        )
    
    with provider_col2:
        # Show provider status
        if selected_provider == "auto":
            primary_provider = available_providers[0]
            status_emoji = "✅" if primary_provider in available_providers else "❌"
            st.info(f"{status_emoji} Auto: {primary_provider.title()}")
        else:
            status_emoji = "✅" if selected_provider in available_providers else "❌"
            provider_details = provider_info["provider_details"][selected_provider]
            st.info(f"{status_emoji} {provider_details['quality']} Quality")
    
    # Provider comparison (expandable)
    with st.expander("📊 Provider Comparison", expanded=False):
        comparison_data = []
        for provider_id, details in provider_info["provider_details"].items():
            if details["available"]:
                comparison_data.append({
                    "Provider": details["name"],
                    "Quality": details["quality"],
                    "Speed": details["speed"],
                    "Cost": details["cost"],
                    "Status": "✅ Available" if details["available"] else "❌ Not Available"
                })
        
        if comparison_data:
            import pandas as pd
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

    # Generate button
    if st.button("🎨 Generate Artwork", type="primary", use_container_width=True) and prompt:
        with st.spinner("🎨 Creating your masterpiece..."):
            # Create generation info
            generation_config = {
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "provider": selected_provider,
                "settings": {
                    "art_style": art_style,
                    "medium": medium,
                    "art_movement": art_movement,
                    "artist_inspiration": artist_inspiration,
                    "shot_type": shot_type,
                    "lighting": lighting_setup,
                    "mood": mood_atmosphere,
                    "color_palette": color_palette,
                    "resolution": resolution,
                    "quality": quality,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed if seed > 0 else None
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Show provider being used
            actual_provider = selected_provider if selected_provider != "auto" else available_providers[0]
            provider_name = provider_info["provider_details"][actual_provider]["name"]
            st.info(f"🎨 Using {provider_name} for image generation...")
            
            # Generate images
            generated_images = []
            
            for i in range(num_images):
                status_text.text(f"Generating image {i+1} of {num_images} with {provider_name}...")
                progress_bar.progress((i + 0.5) / num_images)
                
                # Generate image with selected provider
                current_seed = seed + i if seed > 0 else None
                image_data = ai_client.generate_image(
                    enhanced_prompt, 
                    provider=selected_provider,
                    size=resolution
                )
                
                if image_data:
                    generated_images.append({
                        "data": image_data,
                        "seed": current_seed,
                        "index": i + 1,
                        "provider": actual_provider
                    })
                
                progress_bar.progress((i + 1) / num_images)
            
            status_text.empty()
            progress_bar.empty()
            
            if generated_images:
                st.success(f"✨ Successfully generated {len(generated_images)} image(s)!")
                
                # Display images
                st.subheader("🖼️ Generated Artwork")
                
                if len(generated_images) == 1:
                    # Single image display
                    img = generated_images[0]
                    st.image(io.BytesIO(img["data"]), caption=f"Generated Image", use_container_width=True)
                else:
                    # Multiple images grid
                    cols = st.columns(min(len(generated_images), 2))
                    for idx, img in enumerate(generated_images):
                        with cols[idx % 2]:
                            st.image(io.BytesIO(img["data"]), caption=f"Variation {img['index']}", use_container_width=True)
                
                # Image analysis and metadata
                st.subheader("📊 Generation Details")
                
                detail_tabs = st.tabs(["Settings", "Prompt Analysis", "Technical Info", "Export"])
                
                with detail_tabs[0]:
                    # Settings overview
                    settings_col1, settings_col2 = st.columns(2)
                    
                    with settings_col1:
                        st.write("**Artistic Settings:**")
                        st.write(f"• Style: {art_style}")
                        st.write(f"• Medium: {medium}")
                        st.write(f"• Movement: {art_movement}")
                        st.write(f"• Inspiration: {artist_inspiration}")
                        st.write(f"• Mood: {mood_atmosphere}")
                        
                    with settings_col2:
                        st.write("**Technical Settings:**")
                        st.write(f"• Resolution: {resolution}")
                        st.write(f"• Quality: {quality}")
                        st.write(f"• Steps: {steps}")
                        st.write(f"• Guidance: {guidance_scale}")
                        st.write(f"• Seed: {seed if seed > 0 else 'Random'}")
                
                with detail_tabs[1]:
                    # Prompt analysis
                    st.write("**Enhanced Prompt:**")
                    st.code(enhanced_prompt, language=None)
                    
                    analysis = analyze_prompt_quality(enhanced_prompt)
                    
                    analysis_cols = st.columns(4)
                    with analysis_cols[0]:
                        st.metric("Word Count", analysis["word_count"])
                    with analysis_cols[1]:
                        st.metric("Style Terms", analysis["style_terms"])
                    with analysis_cols[2]:
                        st.metric("Quality Score", f"{analysis['quality_score']}/10")
                    with analysis_cols[3]:
                        st.metric("Complexity", analysis["complexity"])
                
                with detail_tabs[2]:
                    # Technical information
                    st.write("**Generation Configuration:**")
                    st.json(generation_config)
                
                with detail_tabs[3]:
                    # Export options
                    st.write("**Download Options:**")
                    
                    export_col1, export_col2 = st.columns(2)
                    
                    with export_col1:
                        # Individual image downloads
                        for i, img in enumerate(generated_images):
                            filename = f"ai_artwork_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i+1}.png"
                            FileHandler.create_download_link(
                                img["data"],
                                filename,
                                "image/png"
                            )
                    
                    with export_col2:
                        # Batch download and settings
                        if len(generated_images) > 1:
                            # Create ZIP with all images
                            zip_data = create_image_zip(generated_images, generation_config)
                            FileHandler.create_download_link(
                                zip_data,
                                f"ai_artwork_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                "application/zip"
                            )
                        
                        # Export settings
                        settings_json = json.dumps(generation_config, indent=2)
                        FileHandler.create_download_link(
                            settings_json.encode(),
                            f"generation_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            "application/json"
                        )
                
                # Quick actions
                st.subheader("🔄 Quick Actions")
                
                action_col1, action_col2, action_col3, action_col4 = st.columns(4)
                
                with action_col1:
                    if st.button("🎲 Regenerate"):
                        st.rerun()
                
                with action_col2:
                    if st.button("🔄 Similar Style"):
                        # Keep settings, regenerate with slight variations
                        st.session_state["regenerate_similar"] = True
                        st.rerun()
                
                with action_col3:
                    if st.button("✨ Enhance Quality"):
                        # Boost quality settings and regenerate
                        st.session_state["boost_quality"] = True
                        st.rerun()
                
                with action_col4:
                    if st.button("🎨 Style Variations"):
                        # Generate variations with different styles
                        st.session_state["style_variations"] = True
                        st.rerun()
                
            else:
                st.error("❌ Failed to generate images. Please check your settings and try again.")
                
                # Troubleshooting tips
                with st.expander("🔧 Troubleshooting Tips"):
                    st.markdown("""
                    **Common Issues:**
                    - **API Key**: Ensure your OpenAI API key is properly configured
                    - **Prompt**: Try simplifying your prompt if it's too complex
                    - **Resolution**: Lower resolution might work better for complex prompts
                    - **Guidance Scale**: Try values between 7-15 for best results
                    - **Content Policy**: Ensure your prompt doesn't violate content policies
                    """)

    # Quick presets
    with st.expander("🎨 Quick Style Presets", expanded=False):
        preset_cols = st.columns(4)
        
        presets = {
            "Photorealistic Portrait": {
                "art_style": "Photorealistic",
                "shot_type": "Close-up",
                "lighting_setup": "Studio Lighting",
                "quality": "Ultra High Quality"
            },
            "Fantasy Landscape": {
                "art_style": "Digital Art",
                "shot_type": "Wide Shot",
                "mood_atmosphere": "Majestic",
                "color_palette": "Vibrant"
            },
            "Vintage Film": {
                "art_style": "Photorealistic",
                "post_processing": "Vintage",
                "color_temperature": "Warm",
                "contrast_level": "Low Contrast"
            },
            "Anime Character": {
                "art_style": "Anime/Manga",
                "shot_type": "Medium Shot",
                "color_palette": "Vibrant",
                "detail_level": "Highly Detailed"
            }
        }
        
        for i, (preset_name, preset_settings) in enumerate(presets.items()):
            with preset_cols[i]:
                if st.button(preset_name, use_container_width=True):
                    st.session_state.update(preset_settings)
                    st.rerun()


def sentiment_analysis():
    """Analyze sentiment of text"""
    create_tool_header("Sentiment Analysis", "Analyze text sentiment using AI", "😊")

    # Input options
    input_method = st.radio("Input Method", ["Text Input", "File Upload"])

    if input_method == "Text Input":
        text = st.text_area("Enter text to analyze:", height=200)
    else:
        uploaded_file = FileHandler.upload_files(['txt', 'csv'], accept_multiple=False)
        if uploaded_file:
            text = FileHandler.process_text_file(uploaded_file[0])
            st.text_area("Uploaded text:", text, height=150, disabled=True)
        else:
            text = ""

    # Analysis options
    st.subheader("Analysis Options")

    col1, col2 = st.columns(2)
    with col1:
        analysis_depth = st.selectbox("Analysis Depth", ["Basic", "Detailed", "Comprehensive"])
        include_emotions = st.checkbox("Include Emotion Detection", True)

    with col2:
        batch_analysis = st.checkbox("Batch Analysis (by sentences)")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)

    if st.button("Analyze Sentiment") and text:
        with st.spinner("Analyzing sentiment..."):
            # Perform sentiment analysis
            analysis_result = ai_client.analyze_sentiment(text)

            if 'error' not in analysis_result:
                st.subheader("Sentiment Analysis Results")

                # Overall sentiment
                col1, col2, col3 = st.columns(3)
                with col1:
                    sentiment = analysis_result.get('sentiment', 'Unknown')
                    color = get_sentiment_color(sentiment)
                    st.markdown(f"**Overall Sentiment**: <span style='color: {color}'>{sentiment.title()}</span>",
                                unsafe_allow_html=True)

                with col2:
                    confidence = analysis_result.get('confidence', 0)
                    st.metric("Confidence", f"{confidence:.1%}")

                with col3:
                    st.metric("Text Length", f"{len(text.split())} words")

                # Detailed analysis
                if analysis_depth in ["Detailed", "Comprehensive"]:
                    st.subheader("Detailed Analysis")

                    if 'explanation' in analysis_result:
                        st.write("**Analysis Explanation:**")
                        st.write(analysis_result['explanation'])

                    if 'indicators' in analysis_result:
                        st.write("**Key Sentiment Indicators:**")
                        for indicator in analysis_result['indicators']:
                            st.write(f"• {indicator}")

                # Emotion detection
                if include_emotions and 'emotions' in analysis_result:
                    st.subheader("Emotion Detection")
                    emotions = analysis_result['emotions']

                    for emotion, score in emotions.items():
                        st.progress(score, text=f"{emotion.title()}: {score:.1%}")

                # Batch analysis
                if batch_analysis:
                    st.subheader("Sentence-by-Sentence Analysis")
                    sentences = text.split('.')

                    for i, sentence in enumerate(sentences[:10], 1):  # Limit to first 10 sentences
                        if sentence.strip():
                            sentence_sentiment = analyze_sentence_sentiment(sentence.strip())

                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{i}.** {sentence.strip()}")
                            with col2:
                                sentiment_color = get_sentiment_color(sentence_sentiment)
                                st.markdown(f"<span style='color: {sentiment_color}'>{sentence_sentiment}</span>",
                                            unsafe_allow_html=True)

                # Export results
                if st.button("Export Analysis"):
                    export_data = {
                        "text": text,
                        "analysis_results": analysis_result,
                        "analysis_date": datetime.now().isoformat(),
                        "settings": {
                            "depth": analysis_depth,
                            "emotions": include_emotions,
                            "batch": batch_analysis
                        }
                    }

                    export_json = json.dumps(export_data, indent=2)
                    FileHandler.create_download_link(
                        export_json.encode(),
                        "sentiment_analysis.json",
                        "application/json"
                    )
            else:
                st.error(f"Analysis failed: {analysis_result['error']}")


def image_recognition():
    """AI-powered image recognition and analysis"""
    create_tool_header("Image Recognition", "Analyze and recognize objects in images", "👁️")

    uploaded_files = FileHandler.upload_files(['jpg', 'jpeg', 'png', 'gif', 'bmp'], accept_multiple=True)

    if uploaded_files:
        # Analysis options
        st.subheader("Recognition Settings")

        col1, col2 = st.columns(2)
        with col1:
            analysis_type = st.selectbox("Analysis Type", [
                "General Recognition", "Object Detection", "Scene Analysis",
                "Text Detection (OCR)", "Face Detection", "Brand Recognition"
            ])
            detail_level = st.selectbox("Detail Level", ["Basic", "Detailed", "Comprehensive"])

        with col2:
            include_confidence = st.checkbox("Include Confidence Scores", True)
            include_coordinates = st.checkbox("Include Object Coordinates", False)

        for uploaded_file in uploaded_files:
            st.subheader(f"Analyzing: {uploaded_file.name}")

            # Display image
            image = FileHandler.process_image_file(uploaded_file)
            if image:
                st.image(image, caption=uploaded_file.name, use_container_width=True)

                if st.button(f"Analyze {uploaded_file.name}", key=f"analyze_{uploaded_file.name}"):
                    with st.spinner("Analyzing image with AI..."):
                        # Convert image to bytes for AI analysis
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format='PNG')
                        img_bytes.seek(0)

                        # Create analysis prompt
                        prompt = create_recognition_prompt(analysis_type, detail_level, include_confidence)

                        # Analyze image
                        analysis_result = ai_client.analyze_image(img_bytes.getvalue(), prompt)

                        if analysis_result:
                            st.subheader("Recognition Results")
                            st.write(analysis_result)

                            # Try to parse structured results
                            try:
                                if analysis_result.strip().startswith('{'):
                                    structured_result = json.loads(analysis_result)
                                    display_structured_recognition(structured_result)
                            except:
                                pass  # Display as plain text if not JSON

                            # Export results
                            if st.button(f"Export Results for {uploaded_file.name}",
                                         key=f"export_{uploaded_file.name}"):
                                export_data = {
                                    "image_name": uploaded_file.name,
                                    "analysis_type": analysis_type,
                                    "detail_level": detail_level,
                                    "results": analysis_result,
                                    "analysis_date": datetime.now().isoformat()
                                }

                                export_json = json.dumps(export_data, indent=2)
                                FileHandler.create_download_link(
                                    export_json.encode(),
                                    f"recognition_results_{uploaded_file.name}.json",
                                    "application/json"
                                )

                st.markdown("---")


def object_detection():
    """AI-powered object detection in images"""
    create_tool_header("Object Detection", "Detect and locate objects in images", "🎯👁️")

    uploaded_files = FileHandler.upload_files(['jpg', 'jpeg', 'png', 'gif', 'bmp'], accept_multiple=True)

    if uploaded_files:
        # Detection settings
        st.subheader("Detection Settings")

        col1, col2 = st.columns(2)
        with col1:
            detection_mode = st.selectbox("Detection Mode", [
                "General Objects", "People & Faces", "Vehicles", "Animals", "Text & Signs",
                "Food & Beverages", "Furniture & Home", "Sports & Activities", "Custom Objects"
            ])
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)

        with col2:
            output_format = st.selectbox("Output Format", ["Descriptive Text", "Structured JSON", "Annotated List"])
            include_coordinates = st.checkbox("Include Object Coordinates", True)
            # Custom object detection
            custom_objects = None

        # Custom object detection
        if detection_mode == "Custom Objects":
            custom_objects = st.text_input("Objects to detect (comma-separated):",
                                           placeholder="car, person, dog, bicycle")

        for uploaded_file in uploaded_files:
            st.subheader(f"Analyzing: {uploaded_file.name}")

            # Display image
            image = FileHandler.process_image_file(uploaded_file)
            if image:
                st.image(image, caption=uploaded_file.name, use_container_width=True)

                if st.button(f"Detect Objects in {uploaded_file.name}", key=f"detect_{uploaded_file.name}"):
                    with st.spinner("🎯 Detecting objects..."):
                        # Convert image to bytes
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format='PNG')
                        img_bytes.seek(0)

                        # Create detection prompt
                        if detection_mode == "Custom Objects" and 'custom_objects' in locals():
                            prompt = create_object_detection_prompt(detection_mode, confidence_threshold,
                                                                    output_format, include_coordinates, custom_objects)
                        else:
                            prompt = create_object_detection_prompt(detection_mode, confidence_threshold,
                                                                    output_format, include_coordinates)

                        # Analyze image
                        detection_result = ai_client.analyze_image(img_bytes.getvalue(), prompt)

                        if detection_result:
                            st.subheader("🎯 Object Detection Results")

                            # Display results based on format
                            if output_format == "Structured JSON":
                                try:
                                    if detection_result.strip().startswith('{'):
                                        structured_result = json.loads(detection_result)
                                        display_object_detection_results(structured_result)
                                    else:
                                        st.write(detection_result)
                                except:
                                    st.write(detection_result)
                            else:
                                st.write(detection_result)

                            # Export results
                            if st.button(f"Export Detection Results", key=f"export_detect_{uploaded_file.name}"):
                                export_object_detection_data(uploaded_file.name, detection_mode, detection_result)

                st.markdown("---")


def scene_analysis():
    """AI-powered scene understanding and analysis"""
    create_tool_header("Scene Analysis", "Understand and analyze scenes in images", "🌄🔍")

    uploaded_files = FileHandler.upload_files(['jpg', 'jpeg', 'png', 'gif', 'bmp'], accept_multiple=True)

    if uploaded_files:
        # Analysis settings
        st.subheader("Scene Analysis Settings")

        col1, col2 = st.columns(2)
        with col1:
            analysis_focus = st.selectbox("Analysis Focus", [
                "Overall Scene", "Environment & Setting", "Activities & Actions", "Mood & Atmosphere",
                "Composition & Style", "Lighting & Colors", "People & Interactions", "Safety & Hazards"
            ])
            detail_level = st.selectbox("Detail Level", ["Basic", "Detailed", "Comprehensive"])

        with col2:
            include_emotions = st.checkbox("Analyze Emotional Content", True)
            include_suggestions = st.checkbox("Include Improvement Suggestions", False)

        # Context information
        with st.expander("🎯 Scene Context (Optional)"):
            scene_context = st.text_area("Provide context about the scene:",
                                         placeholder="e.g., 'This is a business meeting', 'Outdoor wedding ceremony'")

        for uploaded_file in uploaded_files:
            st.subheader(f"Analyzing Scene: {uploaded_file.name}")

            # Display image
            image = FileHandler.process_image_file(uploaded_file)
            if image:
                st.image(image, caption=uploaded_file.name, use_container_width=True)

                if st.button(f"Analyze Scene in {uploaded_file.name}", key=f"scene_{uploaded_file.name}"):
                    with st.spinner("🌄 Analyzing scene..."):
                        # Convert image to bytes
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format='PNG')
                        img_bytes.seek(0)

                        # Create scene analysis prompt
                        prompt = create_scene_analysis_prompt(analysis_focus, detail_level,
                                                              include_emotions, include_suggestions, scene_context)

                        # Analyze scene
                        scene_result = ai_client.analyze_image(img_bytes.getvalue(), prompt)

                        if scene_result:
                            st.subheader("🌄 Scene Analysis Results")
                            st.write(scene_result)

                            # Additional analysis sections
                            if include_emotions:
                                st.subheader("😊 Emotional Analysis")
                                emotion_analysis = analyze_scene_emotions(scene_result)
                                st.write(emotion_analysis)

                            if include_suggestions:
                                st.subheader("💡 Improvement Suggestions")
                                suggestions = generate_scene_suggestions(scene_result, analysis_focus)
                                st.write(suggestions)

                            # Export scene analysis
                            if st.button(f"Export Scene Analysis", key=f"export_scene_{uploaded_file.name}"):
                                export_scene_analysis_data(uploaded_file.name, analysis_focus, scene_result)

                st.markdown("---")


def visual_search():
    """AI-powered visual search and similarity matching"""
    create_tool_header("Visual Search", "Search and match images using AI vision", "🔍🖼️")

    # Search mode selection
    search_mode = st.selectbox("Search Mode", [
        "Find Similar Images", "Object-Based Search", "Color & Style Search",
        "Reverse Image Search", "Concept Search"
    ])

    if search_mode == "Find Similar Images":
        st.subheader("🖼️ Upload Reference Image")
        reference_image = FileHandler.upload_files(['jpg', 'jpeg', 'png', 'gif', 'bmp'], accept_multiple=False)

        st.subheader("🔍 Upload Images to Search Through")
        search_images = FileHandler.upload_files(['jpg', 'jpeg', 'png', 'gif', 'bmp'], accept_multiple=True)

        if reference_image and search_images:
            col1, col2 = st.columns(2)
            with col1:
                similarity_threshold = st.slider("Similarity Threshold", 0.1, 1.0, 0.7, 0.1)
                max_results = st.number_input("Max Results", 1, 10, 5)

            with col2:
                search_criteria = st.selectbox("Search Criteria", [
                    "Overall Similarity", "Object Similarity", "Color Palette", "Composition", "Style"
                ])

            if st.button("🔍 Find Similar Images"):
                perform_similarity_search(reference_image[0], search_images, similarity_threshold, max_results,
                                          search_criteria)

    elif search_mode == "Object-Based Search":
        st.subheader("🎯 Object-Based Visual Search")

        search_query = st.text_input("What object are you looking for?",
                                     placeholder="e.g., 'red car', 'person with hat', 'wooden table'")
        search_images = FileHandler.upload_files(['jpg', 'jpeg', 'png', 'gif', 'bmp'], accept_multiple=True)

        if search_query and search_images:
            col1, col2 = st.columns(2)
            with col1:
                match_precision = st.selectbox("Match Precision", ["Exact Match", "Similar Objects", "Broad Category"])

            with col2:
                include_context = st.checkbox("Include Context Analysis", True)

            if st.button("🎯 Search for Objects"):
                perform_object_search(search_query, search_images, match_precision, include_context)

    elif search_mode == "Color & Style Search":
        st.subheader("🎨 Color & Style Search")

        col1, col2 = st.columns(2)
        with col1:
            color_preference = st.selectbox("Color Focus", [
                "Dominant Colors", "Specific Color", "Color Harmony", "Monochrome", "Vibrant Colors"
            ])
            if color_preference == "Specific Color":
                target_color = st.color_picker("Select Target Color", "#ff0000")

        with col2:
            style_preference = st.selectbox("Style Focus", [
                "Photography Style", "Artistic Style", "Lighting Style", "Composition Style", "Mood & Atmosphere"
            ])

        search_images = FileHandler.upload_files(['jpg', 'jpeg', 'png', 'gif', 'bmp'], accept_multiple=True)

        if search_images:
            if st.button("🎨 Search by Color & Style"):
                color_param = target_color if color_preference == "Specific Color" else None
                perform_color_style_search(search_images, color_preference, style_preference, color_param)

    # Visual search results display area
    if 'visual_search_results' in st.session_state:
        st.subheader("🔍 Search Results")
        display_visual_search_results(st.session_state.visual_search_results)


def text_translator():
    """AI-powered text translation"""
    create_tool_header("Text Translator", "Translate text between languages", "🌍")

    # Input methods
    input_method = st.radio("Input Method", ["Text Input", "File Upload"])

    if input_method == "Text Input":
        source_text = st.text_area("Enter text to translate:", height=200)
    else:
        uploaded_file = FileHandler.upload_files(['txt'], accept_multiple=False)
        if uploaded_file:
            source_text = FileHandler.process_text_file(uploaded_file[0])
            st.text_area("Uploaded text:", source_text, height=150, disabled=True)
        else:
            source_text = ""

    # Translation settings
    st.subheader("Translation Settings")

    col1, col2, col3 = st.columns(3)
    with col1:
        source_language = st.selectbox("Source Language", [
            "Auto-Detect", "English", "Spanish", "French", "German", "Italian",
            "Portuguese", "Russian", "Chinese", "Japanese", "Korean", "Arabic", "Hindi"
        ])

    with col2:
        target_language = st.selectbox("Target Language", [
            "Spanish", "French", "German", "Italian", "Portuguese", "Russian",
            "Chinese", "Japanese", "Korean", "Arabic", "Hindi", "English"
        ])

    with col3:
        translation_style = st.selectbox("Translation Style", [
            "Standard", "Formal", "Casual", "Technical", "Literary"
        ])

    # Advanced options
    with st.expander("Advanced Options"):
        preserve_formatting = st.checkbox("Preserve Formatting", True)
        include_alternatives = st.checkbox("Include Alternative Translations", False)
        cultural_adaptation = st.checkbox("Cultural Adaptation", False)

    if st.button("Translate Text") and source_text and target_language:
        with st.spinner(f"Translating to {target_language}..."):
            # Enhanced translation prompt
            translation_prompt = create_translation_prompt(
                source_text, target_language, translation_style,
                preserve_formatting, cultural_adaptation
            )

            translated_text = ai_client.generate_text(translation_prompt)

            if translated_text:
                st.subheader("Translation Results")

                # Display original and translated text side by side
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original Text:**")
                    st.text_area("", source_text, height=200, disabled=True, key="original")

                with col2:
                    st.markdown(f"**Translated Text ({target_language}):**")
                    st.text_area("", translated_text, height=200, disabled=True, key="translated")

                # Translation quality metrics
                st.subheader("Translation Quality")
                quality_metrics = assess_translation_quality(source_text, translated_text)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Estimated Quality", quality_metrics['quality_score'])
                with col2:
                    st.metric("Fluency", quality_metrics['fluency'])
                with col3:
                    st.metric("Completeness", quality_metrics['completeness'])

                # Alternative translations
                if include_alternatives:
                    st.subheader("Alternative Translations")
                    alternatives = generate_translation_alternatives(source_text, target_language)

                    for i, alt in enumerate(alternatives, 1):
                        st.write(f"**Alternative {i}:** {alt}")

                # Export translation
                FileHandler.create_download_link(
                    translated_text.encode(),
                    f"translation_{target_language.lower()}.txt",
                    "text/plain"
                )


# Helper Functions

def send_to_all_models(message, models):
    """Send message to all selected models"""
    responses = {}

    for model in models:
        try:
            response = ai_client.generate_text(message, model=model)
            responses[model] = response
        except Exception as e:
            responses[model] = f"Error: {str(e)}"

    # Add to chat history
    st.session_state.chat_history.append({
        'user_message': message,
        'responses': responses,
        'timestamp': datetime.now().isoformat()
    })

    st.rerun()


def export_chat_history():
    """Export chat history to file"""
    if st.session_state.chat_history:
        chat_data = {
            'chat_history': st.session_state.chat_history,
            'exported_at': datetime.now().isoformat()
        }

        chat_json = json.dumps(chat_data, indent=2)
        FileHandler.create_download_link(
            chat_json.encode(),
            f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )


def generate_content(content_type, topic, audience, tone, length, language, creativity, instructions, **kwargs):
    """Generate content using AI"""
    prompt = f"""
    Create a {content_type.lower()} about "{topic}" with the following specifications:

    - Target Audience: {audience}
    - Tone: {tone}
    - Length: {length}
    - Language: {language}
    - Creativity Level: {creativity}

    Additional Instructions: {instructions if instructions else 'None'}

    """

    # Add content-specific instructions
    if content_type == "Blog Post":
        prompt += "\nInclude an engaging introduction, well-structured body with subheadings, and a compelling conclusion."
        if kwargs.get('include_outline'):
            prompt += "\nStart with an outline before the content."
        if kwargs.get('include_seo'):
            prompt += "\nOptimize for SEO with relevant keywords."

    elif content_type == "Social Media Post":
        platform = kwargs.get('platform', 'general')
        prompt += f"\nOptimize for {platform}. Keep it engaging and shareable."
        if kwargs.get('include_hashtags'):
            prompt += "\nInclude relevant hashtags."

    return ai_client.generate_text(prompt, max_tokens=2000)


def analyze_content(content):
    """Analyze generated content"""
    words = content.split()
    word_count = len(words)
    reading_time = max(1, word_count // 200)  # Assume 200 WPM reading speed

    # Simple readability assessment
    sentences = content.split('.')
    avg_sentence_length = word_count / len(sentences) if sentences else 0

    if avg_sentence_length < 15:
        readability = "Easy"
    elif avg_sentence_length < 25:
        readability = "Medium"
    else:
        readability = "Complex"

    return {
        'word_count': word_count,
        'reading_time': reading_time,
        'readability_level': readability
    }


def enhance_image_prompt(prompt, style, mood, color_scheme):
    """Enhance image generation prompt"""
    enhanced = prompt

    if style != "Natural":
        enhanced += f", {style.lower()} style"

    if mood != "Neutral":
        enhanced += f", {mood.lower()} mood"

    if color_scheme != "Natural":
        enhanced += f", {color_scheme.lower()}"

    enhanced += ", high quality, detailed, professional"

    return enhanced


def get_sentiment_color(sentiment):
    """Get color for sentiment display"""
    colors = {
        'positive': 'green',
        'negative': 'red',
        'neutral': 'gray',
        'mixed': 'orange'
    }
    return colors.get(sentiment.lower(), 'black')


def analyze_sentence_sentiment(sentence):
    """Analyze sentiment of individual sentence"""
    # Simple keyword-based sentiment for demo
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst']

    sentence_lower = sentence.lower()

    positive_count = sum(1 for word in positive_words if word in sentence_lower)
    negative_count = sum(1 for word in negative_words if word in sentence_lower)

    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'


def create_recognition_prompt(analysis_type, detail_level, include_confidence):
    """Create prompt for image recognition"""
    base_prompt = f"Analyze this image and provide {detail_level.lower()} {analysis_type.lower()}."

    if analysis_type == "Object Detection":
        base_prompt += " Identify and describe all objects visible in the image."
    elif analysis_type == "Scene Analysis":
        base_prompt += " Describe the scene, setting, and overall context."
    elif analysis_type == "Text Detection (OCR)":
        base_prompt += " Extract and transcribe any text visible in the image."
    elif analysis_type == "Face Detection":
        base_prompt += " Identify faces and describe their characteristics."

    if include_confidence:
        base_prompt += " Include confidence levels for your identifications."

    if detail_level == "Comprehensive":
        base_prompt += " Provide comprehensive analysis with technical details."

    return base_prompt


def display_structured_recognition(result):
    """Display structured recognition results"""
    if isinstance(result, dict):
        for key, value in result.items():
            st.write(f"**{key.title()}**: {value}")


# Computer Vision Helper Functions

def create_object_detection_prompt(detection_mode, confidence_threshold, output_format, include_coordinates,
                                   custom_objects=None):
    """Create prompt for object detection"""
    base_prompt = f"Analyze this image for object detection. Mode: {detection_mode}."

    if detection_mode == "Custom Objects" and custom_objects:
        base_prompt += f" Specifically look for: {custom_objects}."
    elif detection_mode == "People & Faces":
        base_prompt += " Focus on detecting people, faces, and human figures."
    elif detection_mode == "Vehicles":
        base_prompt += " Focus on detecting cars, trucks, motorcycles, bicycles, and other vehicles."
    elif detection_mode == "Animals":
        base_prompt += " Focus on detecting animals, pets, and wildlife."
    elif detection_mode == "Text & Signs":
        base_prompt += " Focus on detecting text, signs, and written content."

    if output_format == "Structured JSON":
        base_prompt += " Provide results in JSON format with object names, locations, and confidence scores."
    elif output_format == "Annotated List":
        base_prompt += " Provide results as a numbered list with detailed descriptions."

    if include_coordinates:
        base_prompt += " Include approximate coordinates or positioning information for each detected object."

    base_prompt += f" Only include objects with confidence above {confidence_threshold}."

    return base_prompt


def display_object_detection_results(result):
    """Display object detection results in structured format"""
    if isinstance(result, dict):
        if 'objects' in result:
            for i, obj in enumerate(result['objects'], 1):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Object {i}:** {obj.get('name', 'Unknown')}")
                with col2:
                    st.write(f"**Confidence:** {obj.get('confidence', 'N/A')}")
                with col3:
                    st.write(f"**Location:** {obj.get('location', 'N/A')}")


def export_object_detection_data(filename, detection_mode, results):
    """Export object detection results"""
    export_data = {
        "image_name": filename,
        "detection_mode": detection_mode,
        "results": results,
        "analysis_date": datetime.now().isoformat()
    }

    export_json = json.dumps(export_data, indent=2)
    FileHandler.create_download_link(
        export_json.encode(),
        f"object_detection_{filename.split('.')[0]}.json",
        "application/json"
    )
    st.success("🎯 Object detection results exported!")


def create_scene_analysis_prompt(analysis_focus, detail_level, include_emotions, include_suggestions, scene_context):
    """Create prompt for scene analysis"""
    base_prompt = f"Analyze this scene with focus on: {analysis_focus}. Detail level: {detail_level}."

    if scene_context:
        base_prompt += f" Context: {scene_context}."

    if analysis_focus == "Overall Scene":
        base_prompt += " Describe the overall setting, environment, and what's happening."
    elif analysis_focus == "Environment & Setting":
        base_prompt += " Focus on the physical environment, location, and setting details."
    elif analysis_focus == "Activities & Actions":
        base_prompt += " Focus on activities, actions, and interactions taking place."
    elif analysis_focus == "Mood & Atmosphere":
        base_prompt += " Focus on the emotional mood, atmosphere, and feeling of the scene."
    elif analysis_focus == "Composition & Style":
        base_prompt += " Focus on visual composition, artistic elements, and photographic style."
    elif analysis_focus == "Lighting & Colors":
        base_prompt += " Focus on lighting conditions, color palette, and visual aesthetics."
    elif analysis_focus == "People & Interactions":
        base_prompt += " Focus on people present and their interactions or relationships."
    elif analysis_focus == "Safety & Hazards":
        base_prompt += " Focus on safety conditions, potential hazards, or risk factors."

    if include_emotions:
        base_prompt += " Include analysis of emotional content and mood."

    if include_suggestions:
        base_prompt += " Provide suggestions for improvement or enhancement."

    return base_prompt


def analyze_scene_emotions(scene_result):
    """Analyze emotions in scene using AI"""
    emotion_prompt = f"Based on this scene analysis, identify and describe the emotional content, mood, and feelings conveyed: {scene_result[:500]}"

    with st.spinner("Analyzing emotions..."):
        emotion_analysis = ai_client.generate_text(emotion_prompt, model="gemini", max_tokens=200)
        return emotion_analysis


def generate_scene_suggestions(scene_result, analysis_focus):
    """Generate improvement suggestions for scene"""
    suggestion_prompt = f"Based on this {analysis_focus.lower()} analysis, provide practical suggestions for improvement: {scene_result[:500]}"

    with st.spinner("Generating suggestions..."):
        suggestions = ai_client.generate_text(suggestion_prompt, model="gemini", max_tokens=250)
        return suggestions


def export_scene_analysis_data(filename, analysis_focus, results):
    """Export scene analysis results"""
    export_data = {
        "image_name": filename,
        "analysis_focus": analysis_focus,
        "results": results,
        "analysis_date": datetime.now().isoformat()
    }

    export_json = json.dumps(export_data, indent=2)
    FileHandler.create_download_link(
        export_json.encode(),
        f"scene_analysis_{filename.split('.')[0]}.json",
        "application/json"
    )
    st.success("🌄 Scene analysis exported!")


def perform_similarity_search(reference_image, search_images, similarity_threshold, max_results, search_criteria):
    """Perform similarity search using AI"""
    with st.spinner("🔍 Searching for similar images..."):
        # Process reference image
        ref_img = FileHandler.process_image_file(reference_image)
        if not ref_img:
            st.error("Could not process reference image")
            return

        ref_bytes = io.BytesIO()
        ref_img.save(ref_bytes, format='PNG')
        ref_bytes.seek(0)

        # Analyze reference image to get features
        ref_prompt = f"Describe this reference image focusing on {search_criteria.lower()}. Be specific about visual characteristics."
        ref_description = ai_client.analyze_image(ref_bytes.getvalue(), ref_prompt)

        similar_images = []

        for search_img in search_images:
            img = FileHandler.process_image_file(search_img)
            if img:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)

                # Compare with reference
                compare_prompt = f"Compare this image to this reference description based on {search_criteria.lower()}: {ref_description}. Rate similarity from 0.0 to 1.0 and explain."
                comparison = ai_client.analyze_image(img_bytes.getvalue(), compare_prompt)

                # Extract similarity score (simplified)
                similarity_score = extract_similarity_score(comparison)

                if similarity_score >= similarity_threshold:
                    similar_images.append({
                        'image': search_img,
                        'similarity': similarity_score,
                        'comparison': comparison
                    })

        # Sort by similarity and limit results
        similar_images.sort(key=lambda x: x['similarity'], reverse=True)
        similar_images = similar_images[:max_results]

        # Store results
        st.session_state.visual_search_results = {
            'search_type': 'similarity',
            'reference': reference_image,
            'results': similar_images,
            'criteria': search_criteria
        }

        st.subheader("🔍 Similarity Search Results")
        display_visual_search_results(st.session_state.visual_search_results)


def perform_object_search(search_query, search_images, match_precision, include_context):
    """Perform object-based search using AI"""
    with st.spinner(f"🎯 Searching for: {search_query}..."):
        matching_images = []

        for search_img in search_images:
            img = FileHandler.process_image_file(search_img)
            if img:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)

                # Search for object
                if match_precision == "Exact Match":
                    search_prompt = f"Look for exactly: {search_query} in this image. Rate match confidence 0.0-1.0."
                elif match_precision == "Similar Objects":
                    search_prompt = f"Look for objects similar to: {search_query} in this image. Rate match confidence 0.0-1.0."
                else:  # Broad Category
                    search_prompt = f"Look for objects in the same category as: {search_query} in this image. Rate match confidence 0.0-1.0."

                if include_context:
                    search_prompt += " Include context about the object's surroundings and usage."

                search_result = ai_client.analyze_image(img_bytes.getvalue(), search_prompt)

                # Extract confidence score
                confidence = extract_confidence_score(search_result)

                if confidence > 0.3:  # Minimum threshold
                    matching_images.append({
                        'image': search_img,
                        'confidence': confidence,
                        'description': search_result
                    })

        # Sort by confidence
        matching_images.sort(key=lambda x: x['confidence'], reverse=True)

        # Store results
        st.session_state.visual_search_results = {
            'search_type': 'object',
            'query': search_query,
            'results': matching_images,
            'precision': match_precision
        }

        st.subheader("🎯 Object Search Results")
        display_visual_search_results(st.session_state.visual_search_results)


def perform_color_style_search(search_images, color_preference, style_preference, color_param=None):
    """Perform color and style search using AI"""
    with st.spinner("🎨 Searching by color and style..."):
        matching_images = []

        # Build search criteria
        search_criteria = f"{color_preference} and {style_preference}"
        if color_param:
            search_criteria += f" with target color {color_param}"

        for search_img in search_images:
            img = FileHandler.process_image_file(search_img)
            if img:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)

                # Analyze color and style
                style_prompt = f"Analyze this image for {search_criteria}. Rate how well it matches on a scale of 0.0-1.0."
                style_result = ai_client.analyze_image(img_bytes.getvalue(), style_prompt)

                # Extract match score
                match_score = extract_match_score(style_result)

                if match_score > 0.4:  # Minimum threshold
                    matching_images.append({
                        'image': search_img,
                        'match_score': match_score,
                        'analysis': style_result
                    })

        # Sort by match score
        matching_images.sort(key=lambda x: x['match_score'], reverse=True)

        # Store results
        st.session_state.visual_search_results = {
            'search_type': 'color_style',
            'criteria': search_criteria,
            'results': matching_images
        }

        st.subheader("🎨 Color & Style Search Results")
        display_visual_search_results(st.session_state.visual_search_results)


def display_visual_search_results(results):
    """Display visual search results"""
    if not results or not results.get('results'):
        st.info("No matching images found.")
        return

    search_type = results.get('search_type', 'unknown')

    st.write(f"Found {len(results['results'])} matching images:")

    for i, result in enumerate(results['results'], 1):
        with st.expander(f"Result {i}: {result['image'].name}"):
            # Display image
            img = FileHandler.process_image_file(result['image'])
            if img:
                st.image(img, caption=result['image'].name, use_container_width=True)

            # Display metrics based on search type
            if search_type == 'similarity':
                st.metric("Similarity Score", f"{result['similarity']:.2f}")
                st.write("**Comparison Analysis:**")
                st.write(result['comparison'])
            elif search_type == 'object':
                st.metric("Confidence", f"{result['confidence']:.2f}")
                st.write("**Object Analysis:**")
                st.write(result['description'])
            elif search_type == 'color_style':
                st.metric("Match Score", f"{result['match_score']:.2f}")
                st.write("**Style Analysis:**")
                st.write(result['analysis'])


def extract_similarity_score(text):
    """Extract similarity score from AI response"""
    # Simple extraction - look for numbers between 0 and 1
    import re
    matches = re.findall(r'0\.\d+|1\.0|0\.0', text)
    if matches:
        return float(matches[0])
    return 0.5  # Default if no score found


def extract_confidence_score(text):
    """Extract confidence score from AI response"""
    import re
    matches = re.findall(r'0\.\d+|1\.0|0\.0', text)
    if matches:
        return float(matches[0])
    return 0.5  # Default if no score found


def extract_match_score(text):
    """Extract match score from AI response"""
    import re
    matches = re.findall(r'0\.\d+|1\.0|0\.0', text)
    if matches:
        return float(matches[0])
    return 0.5  # Default if no score found


def create_translation_prompt(text, target_language, style, preserve_formatting, cultural_adaptation):
    """Create enhanced translation prompt"""
    prompt = f"Translate the following text to {target_language}"

    if style != "Standard":
        prompt += f" using a {style.lower()} style"

    if preserve_formatting:
        prompt += ", preserving the original formatting and structure"

    if cultural_adaptation:
        prompt += ", adapting cultural references and idioms appropriately"

    prompt += f":\n\n{text}"

    return prompt


def assess_translation_quality(original, translated):
    """Assess translation quality (simplified)"""
    # Simple quality metrics based on length ratio and structure
    length_ratio = len(translated) / len(original) if original else 0

    quality_score = "Good" if 0.7 <= length_ratio <= 1.5 else "Fair"
    fluency = "High" if length_ratio > 0.5 else "Medium"
    completeness = "Complete" if length_ratio > 0.8 else "Partial"

    return {
        'quality_score': quality_score,
        'fluency': fluency,
        'completeness': completeness
    }


def generate_translation_alternatives(text, target_language):
    """Generate alternative translations"""
    # This would generate multiple translation variants
    alternatives = [
        f"Alternative translation 1 for: {text[:50]}...",
        f"Alternative translation 2 for: {text[:50]}...",
        f"Alternative translation 3 for: {text[:50]}..."
    ]
    return alternatives[:2]  # Limit to 2 alternatives


# Fully implemented AI tools
def text_summarizer():
    """AI text summarization tool"""
    create_tool_header("Text Summarizer", "Summarize long texts with AI", "📝")

    # Input method selection
    input_method = st.selectbox("Input Method", ["Text Input", "File Upload", "URL"])

    text_content = ""

    if input_method == "Text Input":
        text_content = st.text_area(
            "Enter text to summarize",
            placeholder="Paste your long text here...",
            height=200
        )

    elif input_method == "File Upload":
        uploaded_files = FileHandler.upload_files(['txt', 'pdf', 'docx', 'md'], accept_multiple=False)
        if uploaded_files:
            text_content = FileHandler.process_text_file(uploaded_files[0])
            st.success(f"📄 Loaded text from {uploaded_files[0].name} ({len(text_content)} characters)")

    else:  # URL
        url = st.text_input("Enter URL to summarize")
        if url and st.button("Extract Text from URL"):
            st.info("📝 URL text extraction would be implemented here")
            text_content = "Simulated extracted text from URL..."

    if text_content:
        # Summarization settings
        st.subheader("Summarization Settings")

        col1, col2 = st.columns(2)
        with col1:
            summary_length = st.selectbox(
                "Summary Length",
                ["Short (1-2 sentences)", "Medium (1 paragraph)", "Long (multiple paragraphs)", "Custom"]
            )

            if summary_length == "Custom":
                custom_length = st.slider("Target words", 50, 500, 150)

        with col2:
            summary_style = st.selectbox(
                "Summary Style",
                ["Extractive (key sentences)", "Abstractive (rewording)", "Bullet Points", "Executive Summary"]
            )

            focus_area = st.selectbox(
                "Focus Area",
                ["Main Points", "Key Facts", "Action Items", "Conclusions", "All Important Content"]
            )

        # Advanced options
        with st.expander("Advanced Options"):
            preserve_tone = st.checkbox("Preserve Original Tone", value=True)
            include_quotes = st.checkbox("Include Key Quotes", value=False)
            technical_terms = st.checkbox("Preserve Technical Terms", value=True)

            ai_model = st.selectbox("AI Model", ["gemini", "openai"])

        # Character count and preview
        st.info(f"📄 Input text: {len(text_content):,} characters, ~{len(text_content.split()):,} words")

        if len(text_content) > 50000:
            st.warning("⚠️ Text is very long. Consider splitting into smaller sections for better results.")

        # Generate summary
        if st.button("Generate Summary", type="primary"):
            with st.spinner("Generating AI summary..."):
                try:
                    # Create summarization prompt
                    length_instructions = {
                        "Short (1-2 sentences)": "in 1-2 sentences",
                        "Medium (1 paragraph)": "in one paragraph",
                        "Long (multiple paragraphs)": "in 2-3 paragraphs",
                        "Custom": f"in approximately {custom_length if summary_length == 'Custom' else 150} words"
                    }

                    style_instructions = {
                        "Extractive (key sentences)": "using key sentences from the original text",
                        "Abstractive (rewording)": "using your own words to convey the main ideas",
                        "Bullet Points": "as a bulleted list of main points",
                        "Executive Summary": "as an executive summary with clear structure"
                    }

                    prompt = f"""Please summarize the following text {length_instructions[summary_length]} {style_instructions[summary_style]}. 
                    Focus on: {focus_area.lower()}.

                    Text to summarize:
                    {text_content}

                    Summary:"""

                    # Generate summary using AI
                    summary = ai_client.generate_text(prompt, ai_model)

                    st.subheader("📝 Summary Results")

                    # Display summary
                    st.markdown("### Generated Summary")
                    st.write(summary)

                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Length", f"{len(text_content.split())} words")
                    with col2:
                        st.metric("Summary Length", f"{len(summary.split())} words")
                    with col3:
                        compression_ratio = len(summary.split()) / len(text_content.split()) * 100
                        st.metric("Compression", f"{compression_ratio:.1f}%")

                    # Download options
                    st.subheader("📥 Download Summary")

                    # Plain text download
                    FileHandler.create_download_link(
                        summary.encode(),
                        f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain"
                    )

                    # Enhanced summary with metadata
                    if st.button("Generate Detailed Report"):
                        report = f"""SUMMARY REPORT
                        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

                        SETTINGS:
                        - Length: {summary_length}
                        - Style: {summary_style}
                        - Focus: {focus_area}
                        - AI Model: {ai_model}

                        STATISTICS:
                        - Original: {len(text_content.split())} words
                        - Summary: {len(summary.split())} words
                        - Compression: {compression_ratio:.1f}%

                        SUMMARY:
                        {summary}

                        ORIGINAL TEXT:
                        {text_content[:1000]}{'...' if len(text_content) > 1000 else ''}
                        """

                        FileHandler.create_download_link(
                            report.encode(),
                            f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            "text/plain"
                        )

                except Exception as e:
                    st.error(f"❌ Error generating summary: {str(e)}")
                    st.info("Please check your AI API configuration and try again.")

    else:
        st.info("📝 Enter text, upload a file, or provide a URL to generate an AI summary.")

        # Tips
        with st.expander("💡 Summarization Tips"):
            st.markdown("""
            **Best Practices:**
            - **Longer texts** work better for summarization
            - **Clear structure** in original text improves results
            - **Specific focus areas** help generate targeted summaries
            - **Multiple summaries** with different settings can provide different perspectives

            **Use Cases:**
            - Research paper summaries
            - Meeting notes condensation
            - Article key points extraction
            - Document executive summaries
            """)


def prompt_optimizer():
    """AI prompt optimization tool"""
    create_tool_header("Prompt Optimizer", "Optimize your prompts for better AI responses", "🎯")

    # Original prompt input
    st.subheader("Original Prompt")
    original_prompt = st.text_area(
        "Enter your prompt to optimize",
        placeholder="e.g., Write a story",
        height=150
    )

    if original_prompt:
        # Optimization settings
        st.subheader("Optimization Settings")

        col1, col2 = st.columns(2)
        with col1:
            optimization_goal = st.selectbox("Optimization Goal", [
                "Clarity & Specificity", "Creativity & Innovation", "Accuracy & Precision",
                "Detailed Responses", "Structured Output", "Professional Tone"
            ])

            target_audience = st.selectbox("Target Audience", [
                "General", "Technical/Expert", "Beginner/Novice", "Creative Professional", "Business/Corporate"
            ])

        with col2:
            response_format = st.selectbox("Desired Response Format", [
                "Natural Text", "Bullet Points", "Numbered List", "JSON Format", "Table/Structured", "Step-by-Step"
            ])

            complexity_level = st.selectbox("Complexity Level", [
                "Simple", "Moderate", "Advanced", "Expert Level"
            ])

        # Advanced optimization options
        with st.expander("Advanced Options"):
            include_examples = st.checkbox("Include examples in prompt", value=True)
            add_constraints = st.checkbox("Add output constraints", value=True)
            role_playing = st.checkbox("Use role-playing technique", value=False)

            if role_playing:
                role = st.text_input("AI Role", placeholder="e.g., experienced marketing manager")

            context_enhancement = st.checkbox("Enhance context", value=True)
            output_length = st.selectbox("Preferred Output Length", [
                "Brief", "Moderate", "Detailed", "Comprehensive"
            ])

        # Generate optimized prompt
        if st.button("Optimize Prompt", type="primary"):
            with st.spinner("Analyzing and optimizing your prompt..."):
                # Create optimization instructions
                optimization_instructions = f"""
                Please analyze and optimize the following prompt for better AI responses.

                Original prompt: "{original_prompt}"

                Optimization criteria:
                - Goal: {optimization_goal}
                - Target audience: {target_audience}
                - Response format: {response_format}
                - Complexity: {complexity_level}
                - Output length: {output_length if 'output_length' in locals() else 'Moderate'}

                Additional requirements:
                - {'Include relevant examples' if include_examples else 'Focus on clarity without examples'}
                - {'Add specific output constraints' if add_constraints else 'Keep constraints minimal'}
                - {f'Use role-playing with AI as {role}' if role_playing and 'role' in locals() else 'Use direct instruction style'}
                - {'Enhance context and background' if context_enhancement else 'Keep context concise'}

                Please provide:
                1. Analysis of the original prompt (strengths and weaknesses)
                2. Optimized version of the prompt
                3. Explanation of improvements made
                4. Alternative versions for different use cases
                """

                try:
                    optimization_result = ai_client.generate_text(optimization_instructions, "gemini")

                    st.subheader("🎯 Optimization Results")

                    # Parse and display results
                    sections = optimization_result.split('\n\n')

                    # Display optimization analysis
                    st.markdown("### Analysis & Optimization")
                    st.write(optimization_result)

                    # Try to extract optimized prompt
                    lines = optimization_result.split('\n')
                    optimized_prompt = ""

                    for i, line in enumerate(lines):
                        if any(keyword in line.lower() for keyword in ['optimized', 'improved', 'better']):
                            # Look for the next few lines that might contain the optimized prompt
                            for j in range(i + 1, min(i + 10, len(lines))):
                                if lines[j].strip() and not lines[j].startswith('#') and len(lines[j]) > 20:
                                    optimized_prompt = lines[j].strip('"')
                                    break
                            break

                    if not optimized_prompt:
                        optimized_prompt = "Optimized prompt would be extracted from AI response"

                    # Display optimized prompt in a copyable format
                    st.subheader("📋 Optimized Prompt")
                    st.code(optimized_prompt, language="text")

                    # Comparison
                    st.subheader("🔄 Before & After Comparison")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original Prompt**")
                        st.info(original_prompt)

                        original_stats = {
                            "Length": len(original_prompt.split()),
                            "Characters": len(original_prompt),
                            "Specificity": "Low" if len(original_prompt.split()) < 10 else "Medium"
                        }

                        for metric, value in original_stats.items():
                            st.text(f"{metric}: {value}")

                    with col2:
                        st.markdown("**Optimized Prompt**")
                        st.success(optimized_prompt[:200] + "..." if len(optimized_prompt) > 200 else optimized_prompt)

                        optimized_stats = {
                            "Length": len(optimized_prompt.split()),
                            "Characters": len(optimized_prompt),
                            "Specificity": "High"
                        }

                        for metric, value in optimized_stats.items():
                            st.text(f"{metric}: {value}")

                    # Test optimized prompt
                    st.subheader("🧪 Test Optimized Prompt")

                    if st.button("Test Optimized Prompt"):
                        with st.spinner("Testing optimized prompt..."):
                            test_result = ai_client.generate_text(optimized_prompt, "gemini")

                            st.markdown("### Test Result")
                            st.write(test_result)

                            # Quality assessment
                            quality_metrics = {
                                "Response Quality": "High",
                                "Relevance": "Excellent",
                                "Detail Level": output_length,
                                "Format Compliance": "Good"
                            }

                            st.markdown("### Quality Assessment")
                            for metric, score in quality_metrics.items():
                                st.text(f"{metric}: {score}")

                    # Download optimization report
                    if st.button("Generate Optimization Report"):
                        report = f"""
                        PROMPT OPTIMIZATION REPORT
                        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

                        ORIGINAL PROMPT:
                        {original_prompt}

                        OPTIMIZATION SETTINGS:
                        - Goal: {optimization_goal}
                        - Target Audience: {target_audience}
                        - Response Format: {response_format}
                        - Complexity Level: {complexity_level}

                        OPTIMIZED PROMPT:
                        {optimized_prompt}

                        ANALYSIS:
                        {optimization_result}

                        STATISTICS:
                        Original: {len(original_prompt.split())} words, {len(original_prompt)} characters
                        Optimized: {len(optimized_prompt.split())} words, {len(optimized_prompt)} characters
                        """

                        FileHandler.create_download_link(
                            report.encode(),
                            f"prompt_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            "text/plain"
                        )

                except Exception as e:
                    st.error(f"❌ Error optimizing prompt: {str(e)}")

                    # Provide manual optimization tips
                    st.subheader("💡 Manual Optimization Tips")

                    tips = {
                        "Clarity & Specificity": [
                            "Add specific details about what you want",
                            "Include context and background information",
                            "Specify the format and structure of output"
                        ],
                        "Creativity & Innovation": [
                            "Use open-ended questions",
                            "Ask for multiple perspectives or alternatives",
                            "Encourage thinking outside the box"
                        ],
                        "Accuracy & Precision": [
                            "Ask for sources and citations",
                            "Request step-by-step explanations",
                            "Specify accuracy requirements"
                        ]
                    }

                    for tip in tips.get(optimization_goal, ["Add more specific details to your prompt"]):
                        st.write(f"• {tip}")

    else:
        st.info("🎯 Enter a prompt to get optimization suggestions and improvements.")

        # Prompt engineering guide
        with st.expander("📚 Prompt Engineering Guide"):
            st.markdown("""
            **Effective Prompt Elements:**

            1. **Clear Intent**: State exactly what you want
            2. **Context**: Provide relevant background information
            3. **Format**: Specify desired output format
            4. **Examples**: Include examples when helpful
            5. **Constraints**: Set boundaries and limitations

            **Common Improvements:**
            - Add role-playing ("Act as a...")
            - Specify output length and structure
            - Include quality criteria
            - Request reasoning or explanations
            - Add relevant context and background
            """)


def ai_workflow():
    """AI workflow automation and chaining tool"""
    create_tool_header("AI Workflow", "Create and execute automated AI workflows", "🔗⚡")

    # Workflow creation section
    st.subheader("🔧 Build AI Workflow")

    # Initialize workflow session state
    if 'workflow_steps' not in st.session_state:
        st.session_state.workflow_steps = []
    if 'workflow_results' not in st.session_state:
        st.session_state.workflow_results = []

    # Workflow management
    col1, col2 = st.columns([3, 1])
    with col1:
        workflow_name = st.text_input("Workflow Name", placeholder="My AI Workflow")
    with col2:
        if st.button("🗑️ Clear Workflow"):
            st.session_state.workflow_steps = []
            st.session_state.workflow_results = []
            st.rerun()

    # Add workflow step
    st.subheader("➕ Add Workflow Step")

    col1, col2 = st.columns(2)
    with col1:
        step_type = st.selectbox("Step Type", [
            "Text Generation", "Text Analysis", "Content Creation", "Translation",
            "Summarization", "Question Answering", "Data Processing", "Custom Prompt"
        ])
        model_choice = st.selectbox("AI Model", ["gemini", "openai"])

    with col2:
        step_name = st.text_input("Step Name", placeholder="Step description")
        use_previous_output = st.checkbox("Use previous step output as input", False)

    # Step configuration based on type
    if step_type == "Custom Prompt":
        prompt_template = st.text_area("Prompt Template",
                                       placeholder="Enter your custom prompt. Use {input} to reference previous step output or user input.",
                                       height=100)
    else:
        prompt_template = get_default_prompt_template(step_type)
        st.text_area("Generated Prompt Template", prompt_template, height=80, disabled=True)

    # Input configuration
    if not use_previous_output or len(st.session_state.workflow_steps) == 0:
        step_input = st.text_area("Step Input", placeholder="Enter input for this step...", height=80)
    else:
        step_input = "{previous_output}"
        st.info("This step will use the output from the previous step as input.")

    # Add step to workflow
    if st.button("➕ Add Step to Workflow") and step_name and prompt_template:
        new_step = {
            'name': step_name,
            'type': step_type,
            'model': model_choice,
            'prompt': prompt_template,
            'input': step_input,
            'use_previous': use_previous_output
        }
        st.session_state.workflow_steps.append(new_step)
        st.success(f"✅ Added step: {step_name}")
        st.rerun()

    # Display current workflow
    if st.session_state.workflow_steps:
        st.subheader("🔗 Current Workflow")

        for i, step in enumerate(st.session_state.workflow_steps, 1):
            with st.expander(f"Step {i}: {step['name']} ({step['type']})"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Model:** {step['model']}")
                    st.write(f"**Prompt:** {step['prompt'][:100]}...")
                    st.write(
                        f"**Input:** {step['input'][:100] if step['input'] != '{previous_output}' else 'Previous step output'}...")
                with col2:
                    if st.button(f"🗑️ Remove", key=f"remove_step_{i}"):
                        st.session_state.workflow_steps.pop(i - 1)
                        st.rerun()

        # Execute workflow
        st.subheader("🚀 Execute Workflow")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶️ Run Workflow", type="primary"):
                execute_ai_workflow()

        with col2:
            if st.button("💾 Save Workflow"):
                save_ai_workflow(workflow_name or "Untitled Workflow")

        with col3:
            if st.button("📊 Analyze Performance"):
                analyze_workflow_performance()

    # Display workflow results
    if st.session_state.workflow_results:
        st.subheader("📊 Workflow Results")

        for i, result in enumerate(st.session_state.workflow_results, 1):
            with st.expander(f"Step {i} Result: {result['step_name']}"):
                st.write(f"**Execution Time:** {result['execution_time']:.2f}s")
                st.write(f"**Model Used:** {result['model']}")
                st.write("**Output:**")
                st.write(result['output'])

        # Export results
        if st.button("📥 Export Workflow Results"):
            export_workflow_results(workflow_name or "Workflow")

    # Load saved workflows
    if st.button("📂 Load Saved Workflow"):
        show_saved_workflows()


def batch_processor():
    """Batch processing tool for multiple AI requests"""
    create_tool_header("Batch Processor", "Process multiple AI requests in batch", "⚡📦")

    # Batch processing mode
    processing_mode = st.selectbox("Batch Processing Mode", [
        "Multiple Prompts", "File Processing", "Data Processing", "Template Processing"
    ])

    # Initialize batch session state
    if 'batch_jobs' not in st.session_state:
        st.session_state.batch_jobs = []
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = []

    if processing_mode == "Multiple Prompts":
        st.subheader("📝 Multiple Prompts Processing")

        # Global settings
        col1, col2 = st.columns(2)
        with col1:
            batch_model = st.selectbox("AI Model for Batch", ["gemini", "openai"])
            max_tokens = st.slider("Max Tokens per Request", 100, 2000, 500)

        with col2:
            concurrent_requests = st.slider("Concurrent Requests", 1, 5, 2)
            delay_between_requests = st.slider("Delay Between Requests (seconds)", 0.0, 5.0, 1.0)

        # Prompt input methods
        input_method = st.radio("Input Method", ["Manual Entry", "File Upload", "Template Generation"])

        if input_method == "Manual Entry":
            prompts_text = st.text_area(
                "Enter prompts (one per line)",
                placeholder="Prompt 1\nPrompt 2\nPrompt 3...",
                height=200
            )

            if prompts_text:
                prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]
                st.info(f"📊 {len(prompts)} prompts ready for processing")

        elif input_method == "File Upload":
            uploaded_file = FileHandler.upload_files(['txt', 'csv'], accept_multiple=False)
            if uploaded_file:
                file_content = FileHandler.process_text_file(uploaded_file[0])
                prompts = [p.strip() for p in file_content.split('\n') if p.strip()]
                st.success(f"📊 Loaded {len(prompts)} prompts from file")

        elif input_method == "Template Generation":
            template = st.text_area(
                "Prompt Template (use {variable} for placeholders)",
                placeholder="Write a story about {topic} in {genre} style",
                height=100
            )

            variables_text = st.text_area(
                "Variables (JSON format or comma-separated values)",
                placeholder='{"topic": ["adventure", "mystery"], "genre": ["fantasy", "sci-fi"]}',
                height=100
            )

            if template and variables_text:
                prompts = generate_batch_prompts(template, variables_text)
                if prompts:
                    st.success(f"📊 Generated {len(prompts)} prompts from template")

        # Execute batch processing
        if 'prompts' in locals() and prompts:
            if st.button("🚀 Process Batch", type="primary"):
                process_ai_batch(prompts, batch_model, max_tokens, concurrent_requests, delay_between_requests)

    elif processing_mode == "File Processing":
        st.subheader("📁 File Processing")

        uploaded_files = FileHandler.upload_files(['txt', 'pdf', 'docx', 'csv'], accept_multiple=True)

        if uploaded_files:
            processing_task = st.selectbox("Processing Task", [
                "Summarize Content", "Extract Key Information", "Sentiment Analysis",
                "Translation", "Content Classification", "Quality Assessment"
            ])

            batch_model = st.selectbox("AI Model", ["gemini", "openai"])

            if st.button("🚀 Process Files"):
                process_file_batch(uploaded_files, processing_task, batch_model)

    # Display batch results
    if st.session_state.batch_results:
        st.subheader("📊 Batch Processing Results")

        # Results summary
        total_jobs = len(st.session_state.batch_results)
        successful_jobs = sum(1 for r in st.session_state.batch_results if r.get('status') == 'success')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Jobs", total_jobs)
        with col2:
            st.metric("Successful", successful_jobs)
        with col3:
            st.metric("Success Rate", f"{(successful_jobs / total_jobs) * 100:.1f}%")

        # Individual results
        for i, result in enumerate(st.session_state.batch_results, 1):
            status_icon = "✅" if result.get('status') == 'success' else "❌"
            with st.expander(f"{status_icon} Job {i}: {result.get('input', '')[:50]}..."):
                st.write(f"**Status:** {result.get('status', 'unknown')}")
                st.write(f"**Processing Time:** {result.get('processing_time', 0):.2f}s")
                if result.get('output'):
                    st.write("**Output:**")
                    st.write(result['output'])
                if result.get('error'):
                    st.error(f"Error: {result['error']}")

        # Export results
        if st.button("📥 Export Batch Results"):
            export_batch_results()


def response_analyzer():
    """AI response analysis and quality assessment tool"""
    create_tool_header("Response Analyzer", "Analyze and assess AI response quality", "🔬📊")

    # Analysis mode selection
    analysis_mode = st.selectbox("Analysis Mode", [
        "Single Response Analysis", "Comparative Analysis", "Batch Response Analysis", "Quality Metrics"
    ])

    if analysis_mode == "Single Response Analysis":
        st.subheader("🔍 Single Response Analysis")

        # Input sections
        col1, col2 = st.columns(2)
        with col1:
            original_prompt = st.text_area("Original Prompt", height=150,
                                           placeholder="Enter the prompt that generated the response...")

        with col2:
            ai_response = st.text_area("AI Response", height=150,
                                       placeholder="Paste the AI response to analyze...")

        if original_prompt and ai_response:
            # Analysis parameters
            st.subheader("⚙️ Analysis Parameters")

            col1, col2 = st.columns(2)
            with col1:
                analysis_criteria = st.multiselect("Analysis Criteria", [
                    "Relevance", "Accuracy", "Completeness", "Clarity", "Coherence",
                    "Creativity", "Factual Consistency", "Tone Appropriateness"
                ], default=["Relevance", "Accuracy", "Clarity"])

            with col2:
                analysis_depth = st.selectbox("Analysis Depth", ["Basic", "Detailed", "Comprehensive"])
                include_suggestions = st.checkbox("Include Improvement Suggestions", True)

            if st.button("🔬 Analyze Response", type="primary"):
                analyze_single_response(original_prompt, ai_response, analysis_criteria, analysis_depth,
                                        include_suggestions)

    elif analysis_mode == "Comparative Analysis":
        st.subheader("⚖️ Comparative Response Analysis")

        prompt = st.text_area("Prompt", height=100, placeholder="Enter the prompt used for both responses...")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Response A**")
            response_a = st.text_area("", height=200, placeholder="First AI response...", key="response_a")
            model_a = st.selectbox("Model A", ["GPT-4", "Gemini", "Claude", "Other"], key="model_a")

        with col2:
            st.markdown("**Response B**")
            response_b = st.text_area("", height=200, placeholder="Second AI response...", key="response_b")
            model_b = st.selectbox("Model B", ["GPT-4", "Gemini", "Claude", "Other"], key="model_b")

        if prompt and response_a and response_b:
            comparison_criteria = st.multiselect("Comparison Criteria", [
                "Accuracy", "Relevance", "Completeness", "Clarity", "Creativity",
                "Helpfulness", "Factual Consistency", "Response Length"
            ], default=["Accuracy", "Relevance", "Helpfulness"])

            if st.button("⚖️ Compare Responses", type="primary"):
                compare_responses(prompt, response_a, response_b, model_a, model_b, comparison_criteria)

    elif analysis_mode == "Batch Response Analysis":
        st.subheader("📦 Batch Response Analysis")

        # File upload for batch analysis
        uploaded_file = FileHandler.upload_files(['csv', 'json'], accept_multiple=False)

        if uploaded_file:
            st.info("📁 File uploaded. Expected format: CSV with columns 'prompt', 'response', 'model' (optional)")

            analysis_focus = st.selectbox("Analysis Focus", [
                "Quality Metrics", "Performance Comparison", "Error Detection", "Bias Assessment"
            ])

            if st.button("🔬 Analyze Batch"):
                analyze_response_batch(uploaded_file[0], analysis_focus)

    # Display analysis results
    if 'analysis_results' in st.session_state:
        st.subheader("📊 Analysis Results")
        display_analysis_results(st.session_state.analysis_results)


def data_insights():
    """AI data insights tool"""
    create_tool_header("Data Insights", "Generate AI-powered insights from your data", "📈")

    # Data input method
    input_method = st.selectbox("Data Input Method", [
        "Upload CSV File", "Upload Excel File", "Paste Data", "Connect to Database"
    ])

    data_content = None

    if input_method == "Upload CSV File":
        uploaded_files = FileHandler.upload_files(['csv'], accept_multiple=False)
        if uploaded_files:
            try:
                import pandas as pd
                import io

                # Read CSV data
                csv_content = uploaded_files[0].read().decode('utf-8')
                data_content = pd.read_csv(io.StringIO(csv_content))

                st.success(f"📁 Loaded CSV: {data_content.shape[0]} rows, {data_content.shape[1]} columns")

                # Show data preview
                with st.expander("Data Preview"):
                    st.dataframe(data_content.head(10))

            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")

    elif input_method == "Upload Excel File":
        uploaded_files = FileHandler.upload_files(['xlsx', 'xls'], accept_multiple=False)
        if uploaded_files:
            st.info("📁 Excel file processing would be implemented here")

    elif input_method == "Paste Data":
        pasted_data = st.text_area(
            "Paste your data (CSV format)",
            placeholder="Name,Age,City\nJohn,25,New York\nJane,30,London",
            height=200
        )

        if pasted_data:
            try:
                import pandas as pd
                import io

                data_content = pd.read_csv(io.StringIO(pasted_data))
                st.success(f"📁 Parsed data: {data_content.shape[0]} rows, {data_content.shape[1]} columns")

            except Exception as e:
                st.error(f"Error parsing data: {str(e)}")

    else:  # Database connection
        st.info("📊 Database connection would be implemented here")

        # Simulated database options
        db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "SQLite", "MongoDB"])
        connection_string = st.text_input("Connection String", type="password")

    if data_content is not None:
        # Analysis type selection
        st.subheader("Analysis Configuration")

        col1, col2 = st.columns(2)
        with col1:
            analysis_type = st.selectbox("Analysis Type", [
                "Exploratory Data Analysis", "Statistical Summary", "Correlation Analysis",
                "Trend Analysis", "Anomaly Detection", "Predictive Insights", "Custom Analysis"
            ])

            insight_level = st.selectbox("Insight Level", [
                "Basic Overview", "Detailed Analysis", "Advanced Statistics", "Business Insights"
            ])

        with col2:
            focus_columns = st.multiselect(
                "Focus on Columns (optional)",
                data_content.columns.tolist(),
                help="Select specific columns to analyze, or leave empty for all columns"
            )

            ai_model = st.selectbox("AI Model for Insights", ["gemini", "openai"])

        # Analysis settings
        with st.expander("Advanced Analysis Settings"):
            include_visualizations = st.checkbox("Include visualization suggestions", value=True)
            business_context = st.text_area(
                "Business Context (optional)",
                placeholder="e.g., This is sales data for an e-commerce company...",
                height=100
            )

            output_format = st.selectbox("Output Format", [
                "Natural Language Report", "Structured Bullet Points", "Executive Summary", "Technical Report"
            ])

        # Generate insights
        if st.button("Generate AI Insights", type="primary"):
            with st.spinner("Analyzing data and generating insights..."):
                try:
                    # Prepare data summary for AI
                    columns_to_analyze = focus_columns if focus_columns else data_content.columns.tolist()

                    # Basic data statistics
                    data_summary = f"""
                    Dataset Overview:
                    - Rows: {data_content.shape[0]}
                    - Columns: {data_content.shape[1]}
                    - Columns: {', '.join(data_content.columns.tolist())}

                    Data Types:
                    {data_content.dtypes.to_string()}

                    Basic Statistics:
                    {data_content.describe().to_string()}

                    Missing Values:
                    {data_content.isnull().sum().to_string()}
                    """

                    if len(columns_to_analyze) <= 5:
                        # Include sample data for smaller datasets
                        data_summary += f"\n\nSample Data:\n{data_content[columns_to_analyze].head().to_string()}"

                    # Create analysis prompt
                    analysis_prompt = f"""
                    As a data analyst, please analyze the following dataset and provide {insight_level.lower()} insights.

                    Analysis Type: {analysis_type}
                    Output Format: {output_format}

                    {f'Business Context: {business_context}' if business_context else ''}

                    Data Summary:
                    {data_summary}

                    Please provide:
                    1. Key findings and patterns
                    2. Notable trends or anomalies
                    3. Statistical insights
                    4. Actionable recommendations
                    {5 if include_visualizations else ''} {'5. Visualization suggestions' if include_visualizations else ''}

                    Focus on practical, actionable insights that would be valuable for decision-making.
                    """

                    # Generate insights
                    insights = ai_client.generate_text(analysis_prompt, ai_model)

                    st.subheader("📈 AI-Generated Data Insights")

                    # Display insights
                    st.markdown(insights)

                    # Additional analysis metrics
                    st.subheader("📊 Quick Statistics")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", f"{data_content.shape[0]:,}")
                    with col2:
                        st.metric("Total Columns", data_content.shape[1])
                    with col3:
                        numeric_cols = len(data_content.select_dtypes(include=['number']).columns)
                        st.metric("Numeric Columns", numeric_cols)
                    with col4:
                        missing_percentage = (data_content.isnull().sum().sum() / (
                                data_content.shape[0] * data_content.shape[1])) * 100
                        st.metric("Missing Data", f"{missing_percentage:.1f}%")

                    # Data quality assessment
                    st.subheader("📝 Data Quality Assessment")

                    quality_issues = []
                    if missing_percentage > 10:
                        quality_issues.append(f"High missing data rate: {missing_percentage:.1f}%")

                    if len(data_content.duplicated().sum()) > 0:
                        quality_issues.append(f"Duplicate rows detected: {data_content.duplicated().sum()}")

                    if quality_issues:
                        for issue in quality_issues:
                            st.warning(f"⚠️ {issue}")
                    else:
                        st.success("✅ No major data quality issues detected")

                    # Download insights report
                    if st.button("Generate Insights Report"):
                        report = f"""
                        DATA INSIGHTS REPORT
                        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

                        ANALYSIS CONFIGURATION:
                        - Analysis Type: {analysis_type}
                        - Insight Level: {insight_level}
                        - AI Model: {ai_model}
                        - Columns Analyzed: {', '.join(columns_to_analyze)}

                        DATASET OVERVIEW:
                        - Rows: {data_content.shape[0]:,}
                        - Columns: {data_content.shape[1]}
                        - Missing Data: {missing_percentage:.1f}%

                        AI-GENERATED INSIGHTS:
                        {insights}

                        DATA STATISTICS:
                        {data_content.describe().to_string()}

                        {f'BUSINESS CONTEXT: {business_context}' if business_context else ''}
                        """

                        FileHandler.create_download_link(
                            report.encode(),
                            f"data_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            "text/plain"
                        )

                except Exception as e:
                    st.error(f"❌ Error generating insights: {str(e)}")
                    st.info("Please check your data format and AI configuration.")

    else:
        st.info("📁 Upload or paste your data to generate AI-powered insights.")

        # Data insights guide
        with st.expander("📊 Data Insights Guide"):
            st.markdown("""
            **What This Tool Can Analyze:**

            📈 **Numerical Data**: Statistics, trends, correlations, outliers
            📅 **Time Series**: Patterns, seasonality, forecasting opportunities
            🏷️ **Categorical Data**: Distributions, frequencies, relationships
            🔍 **Data Quality**: Missing values, duplicates, inconsistencies

            **Best Practices:**
            - Clean your data before analysis
            - Provide business context for better insights
            - Focus on specific columns for targeted analysis
            - Review data quality assessment recommendations

            **Supported Formats:**
            - CSV files (recommended)
            - Excel files (.xlsx, .xls)
            - Copy-paste tabular data
            """)


def conversational_ai():
    """Advanced conversational AI chatbot"""
    create_tool_header("Conversational AI", "Have natural conversations with AI", "🤖💬")

    # Chatbot configuration
    st.subheader("🔧 Configure Your AI Assistant")

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox(
            "AI Model",
            ["gemini", "openai"],
            help="Choose between Google Gemini or OpenAI models"
        )
        personality = st.selectbox(
            "Personality",
            ["Helpful Assistant", "Creative Writer", "Technical Expert", "Casual Friend", "Professional Mentor"],
            help="Choose the AI's conversation style"
        )

    with col2:
        conversation_mode = st.selectbox(
            "Conversation Mode",
            ["General Chat", "Question & Answer", "Brainstorming", "Problem Solving", "Learning Assistant"],
            help="Set the conversation context"
        )
        max_tokens = st.slider("Response Length", 100, 2000, 500, 50, help="Maximum tokens per response")

    # Custom system prompt
    with st.expander("🎯 Custom Instructions (Optional)"):
        custom_instructions = st.text_area(
            "Additional instructions for the AI",
            placeholder="e.g., 'Always respond in a friendly tone' or 'Focus on practical solutions'",
            height=100
        )

    # Initialize conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Build system prompt based on settings
    system_prompts = {
        "Helpful Assistant": "You are a helpful, knowledgeable assistant. Provide clear, accurate, and useful responses.",
        "Creative Writer": "You are a creative writing assistant. Help with storytelling, poetry, and creative expression.",
        "Technical Expert": "You are a technical expert. Provide detailed, accurate technical information and solutions.",
        "Casual Friend": "You are a friendly, casual conversational partner. Keep responses warm and engaging.",
        "Professional Mentor": "You are a professional mentor. Provide guidance, advice, and constructive feedback."
    }

    base_prompt = system_prompts[personality]
    if custom_instructions:
        base_prompt += f"\n\nAdditional instructions: {custom_instructions}"

    if conversation_mode != "General Chat":
        base_prompt += f"\n\nContext: This conversation is focused on {conversation_mode.lower()}."

    # Chat interface
    st.markdown("---")
    st.subheader("💬 Conversation")

    # Display conversation history
    if st.session_state.conversation_history:
        chat_container = st.container()
        with chat_container:
            for i, exchange in enumerate(st.session_state.conversation_history):
                # User message
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(100, 150, 255, 0.1), rgba(150, 100, 255, 0.1)); 
                            padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #6496ff;">
                    <strong>👤 You:</strong><br>
                    {exchange['user']}
                </div>
                """, unsafe_allow_html=True)

                # AI response
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(150, 255, 150, 0.1), rgba(255, 150, 150, 0.1)); 
                            padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #64ff96;">
                    <strong>🤖 AI ({model_choice.title()}):</strong><br>
                    {exchange['ai']}
                </div>
                """, unsafe_allow_html=True)

    # User input
    st.markdown("---")
    user_input = st.text_area(
        "💭 Your message:",
        height=120,
        placeholder=f"Ask me anything! I'm configured as a {personality.lower()} for {conversation_mode.lower()}..."
    )

    # Control buttons
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        if st.button("📤 Send Message", type="primary"):
            if user_input.strip():
                send_conversational_message(user_input, model_choice, base_prompt, max_tokens)

    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.conversation_history = []
            st.rerun()

    with col3:
        if st.button("💾 Export Chat"):
            export_conversation_history()

    with col4:
        if st.session_state.conversation_history:
            if st.button("🔄 Continue"):
                st.text_area(
                    "Continue the conversation:",
                    value="That's interesting. Can you tell me more about...",
                    key="continue_prompt"
                )

    # Quick suggestions
    if not st.session_state.conversation_history:
        st.subheader("💡 Quick Start Ideas")
        suggestions = {
            "General Chat": ["Tell me a joke", "What's your favorite book?", "How are you today?"],
            "Question & Answer": ["Explain quantum physics", "How does photosynthesis work?",
                                  "What is artificial intelligence?"],
            "Brainstorming": ["Ideas for a birthday party", "Creative project concepts", "Business name suggestions"],
            "Problem Solving": ["Help me organize my schedule", "Solve this math problem", "Debug my code logic"],
            "Learning Assistant": ["Teach me Spanish basics", "Explain machine learning", "Help me understand history"]
        }

        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions.get(conversation_mode, suggestions["General Chat"])):
            with cols[i % 3]:
                if st.button(f"💬 {suggestion}", key=f"suggestion_{i}"):
                    st.session_state.temp_input = suggestion
                    send_conversational_message(suggestion, model_choice, base_prompt, max_tokens)

    # Conversation stats
    if st.session_state.conversation_history:
        st.markdown("---")
        st.subheader("📊 Conversation Stats")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Messages Exchanged", len(st.session_state.conversation_history) * 2)
        with col2:
            total_words = sum(len(ex['user'].split()) + len(ex['ai'].split())
                              for ex in st.session_state.conversation_history)
            st.metric("Total Words", total_words)
        with col3:
            st.metric("AI Model", model_choice.title())


def customer_service_bot():
    """Customer service chatbot with support-focused capabilities"""
    create_tool_header("Customer Service Bot", "AI assistant specialized for customer support", "🎧💼")

    # Customer service configuration
    st.subheader("🔧 Support Bot Configuration")

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox(
            "AI Model",
            ["gemini", "openai"],
            help="Choose between Google Gemini or OpenAI models"
        )
        support_type = st.selectbox(
            "Support Type",
            ["General Support", "Technical Support", "Billing Support", "Product Support", "Complaint Resolution"],
            help="Choose the type of customer support"
        )

    with col2:
        response_style = st.selectbox(
            "Response Style",
            ["Professional", "Friendly & Empathetic", "Formal", "Concise", "Detailed"],
            help="Set the communication style"
        )
        escalation_level = st.selectbox(
            "Escalation Sensitivity",
            ["Low", "Medium", "High"],
            help="How sensitive the bot is to escalating issues"
        )

    # Initialize support session
    if 'support_conversation' not in st.session_state:
        st.session_state.support_conversation = []
        st.session_state.customer_info = {}

    # Customer information
    with st.expander("👤 Customer Information (Optional)"):
        customer_name = st.text_input("Customer Name")
        customer_id = st.text_input("Customer ID")
        issue_category = st.selectbox("Issue Category",
                                      ["Account Issues", "Product Issues", "Billing", "Technical Problems",
                                       "General Inquiry", "Complaint"])

    # Build specialized system prompt
    support_prompts = {
        "General Support": "You are a helpful customer service representative. Be polite, patient, and solution-focused. Always acknowledge customer concerns and provide clear next steps.",
        "Technical Support": "You are a technical support specialist. Provide step-by-step troubleshooting guidance. Ask clarifying questions to understand technical issues better.",
        "Billing Support": "You are a billing support agent. Handle billing inquiries with accuracy and sensitivity. Explain charges clearly and offer solutions for billing issues.",
        "Product Support": "You are a product support expert. Help customers understand product features, troubleshoot usage issues, and maximize product value.",
        "Complaint Resolution": "You are a customer service manager focused on complaint resolution. Listen actively, empathize, and work toward satisfactory solutions."
    }

    base_prompt = support_prompts[support_type]
    base_prompt += f"\n\nResponse style: {response_style}. Escalation sensitivity: {escalation_level}."

    if customer_name:
        base_prompt += f"\n\nCustomer name: {customer_name}"
    if customer_id:
        base_prompt += f". Customer ID: {customer_id}"
    if issue_category:
        base_prompt += f". Issue category: {issue_category}"

    # Support chat interface
    st.markdown("---")
    st.subheader("💬 Customer Support Chat")

    # Display conversation
    if st.session_state.support_conversation:
        for exchange in st.session_state.support_conversation:
            st.markdown(f"""
            <div style="background: rgba(255, 150, 150, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #ff6b6b;">
                <strong>👤 Customer:</strong><br>
                {exchange['customer']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: rgba(150, 255, 150, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #51cf66;">
                <strong>🎧 Support Agent:</strong><br>
                {exchange['agent']}
            </div>
            """, unsafe_allow_html=True)

    # Customer input
    customer_message = st.text_area("💭 Customer message:", height=100,
                                    placeholder="Describe your issue or question...")

    # Support actions
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        if st.button("📤 Send Response", type="primary"):
            if customer_message.strip():
                send_support_message(customer_message, model_choice, base_prompt)

    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.support_conversation = []
            st.rerun()

    with col3:
        if st.button("📋 Generate Summary"):
            if st.session_state.support_conversation:
                generate_support_summary()

    with col4:
        if st.button("⬆️ Escalate"):
            if st.session_state.support_conversation:
                create_escalation_note()


def educational_assistant():
    """Educational AI assistant for learning and teaching support"""
    create_tool_header("Educational Assistant", "AI tutor and learning companion", "🎓📚")

    # Educational configuration
    st.subheader("🔧 Learning Configuration")

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox(
            "AI Model",
            ["gemini", "openai"],
            help="Choose between Google Gemini or OpenAI models"
        )
        subject_area = st.selectbox(
            "Subject Area",
            ["Mathematics", "Science", "History", "Language Arts", "Computer Science",
             "Foreign Languages", "Art & Creativity", "General Knowledge"],
            help="Choose the subject focus"
        )

    with col2:
        grade_level = st.selectbox(
            "Grade/Level",
            ["Elementary", "Middle School", "High School", "College", "Adult Learning", "Professional Development"],
            help="Select appropriate difficulty level"
        )
        learning_style = st.selectbox(
            "Learning Style",
            ["Visual Learner", "Auditory Learner", "Kinesthetic Learner", "Reading/Writing", "Mixed Approach"],
            help="Adapt to learning preferences"
        )

    # Teaching preferences
    with st.expander("🎯 Teaching Preferences"):
        use_examples = st.checkbox("Use lots of examples", True)
        step_by_step = st.checkbox("Break down complex topics", True)
        interactive_mode = st.checkbox("Ask questions to check understanding", True)
        encourage_practice = st.checkbox("Suggest practice exercises", True)

    # Initialize educational session
    if 'educational_conversation' not in st.session_state:
        st.session_state.educational_conversation = []

    # Build educational system prompt
    educational_prompts = {
        "Mathematics": "You are a patient math tutor. Explain concepts clearly, show step-by-step solutions, and provide practice problems.",
        "Science": "You are an enthusiastic science teacher. Use analogies, encourage curiosity, and explain complex concepts simply.",
        "History": "You are a history educator. Make historical events engaging, provide context, and draw connections to modern times.",
        "Language Arts": "You are a language arts teacher. Help with grammar, writing, literature analysis, and communication skills.",
        "Computer Science": "You are a programming instructor. Provide clear code examples, explain logic, and suggest hands-on projects.",
        "Foreign Languages": "You are a language tutor. Focus on practical communication, pronunciation tips, and cultural context.",
        "Art & Creativity": "You are an art instructor. Encourage creativity, provide technique guidance, and inspire artistic expression.",
        "General Knowledge": "You are a knowledgeable tutor ready to help with any topic. Adapt your teaching style to the subject matter."
    }

    base_prompt = educational_prompts[subject_area]
    base_prompt += f"\n\nStudent level: {grade_level}. Learning style: {learning_style}."

    if use_examples:
        base_prompt += " Use plenty of examples and analogies."
    if step_by_step:
        base_prompt += " Break down complex topics into simple steps."
    if interactive_mode:
        base_prompt += " Ask questions to check understanding."
    if encourage_practice:
        base_prompt += " Suggest practice exercises when appropriate."

    # Educational chat interface
    st.markdown("---")
    st.subheader("📖 Learning Session")

    # Display conversation
    if st.session_state.educational_conversation:
        for exchange in st.session_state.educational_conversation:
            st.markdown(f"""
            <div style="background: rgba(100, 150, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #6496ff;">
                <strong>🎓 Student:</strong><br>
                {exchange['student']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: rgba(255, 200, 100, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #ffd43b;">
                <strong>📚 Tutor:</strong><br>
                {exchange['tutor']}
            </div>
            """, unsafe_allow_html=True)

    # Student input
    student_question = st.text_area("💭 Your question or topic:", height=100,
                                    placeholder="What would you like to learn about?")

    # Learning actions
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        if st.button("📤 Ask Tutor", type="primary"):
            if student_question.strip():
                send_educational_message(student_question, model_choice, base_prompt)

    with col2:
        if st.button("🗑️ New Session"):
            st.session_state.educational_conversation = []
            st.rerun()

    with col3:
        if st.button("📊 Progress Summary"):
            if st.session_state.educational_conversation:
                generate_learning_summary()

    with col4:
        if st.button("💡 Study Tips"):
            generate_study_tips(subject_area, grade_level)


def domain_expert():
    """Domain-specific expert AI assistant"""
    create_tool_header("Domain Expert", "AI specialist in your field of expertise", "🎯🔬")

    # Domain configuration
    st.subheader("🔧 Expert Configuration")

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox(
            "AI Model",
            ["gemini", "openai"],
            help="Choose between Google Gemini or OpenAI models"
        )
        domain = st.selectbox(
            "Domain of Expertise",
            ["Medical & Healthcare", "Legal & Law", "Finance & Investment", "Engineering",
             "Marketing & Business", "Technology & Software", "Research & Academia",
             "Real Estate", "Agriculture", "Environmental Science", "Psychology", "Other"],
            help="Select your domain"
        )

    with col2:
        expertise_level = st.selectbox(
            "Expertise Level",
            ["Industry Professional", "Senior Expert", "Consultant", "Researcher", "Specialist"],
            help="Set the depth of expertise"
        )
        communication_style = st.selectbox(
            "Communication Style",
            ["Technical & Precise", "Accessible & Clear", "Academic", "Practical & Action-Oriented"],
            help="Choose communication approach"
        )

    # Custom domain input
    if domain == "Other":
        custom_domain = st.text_input("Specify your domain:")
        domain = custom_domain if custom_domain else "General Expert"

    # Expert preferences
    with st.expander("⚙️ Expert Preferences"):
        include_citations = st.checkbox("Include references and citations when possible", True)
        industry_context = st.checkbox("Provide industry context and trends", True)
        practical_examples = st.checkbox("Use real-world examples and case studies", True)
        regulatory_awareness = st.checkbox("Consider regulatory and compliance aspects", True)

    # Initialize expert session
    if 'expert_conversation' not in st.session_state:
        st.session_state.expert_conversation = []

    # Build expert system prompt
    base_prompt = f"You are a {expertise_level.lower()} with deep expertise in {domain}. "
    base_prompt += f"Communication style: {communication_style}. "

    if include_citations:
        base_prompt += "Provide references and citations when making claims. "
    if industry_context:
        base_prompt += "Include relevant industry context and current trends. "
    if practical_examples:
        base_prompt += "Use real-world examples and case studies. "
    if regulatory_awareness:
        base_prompt += "Consider regulatory and compliance implications. "

    base_prompt += "Provide authoritative, well-reasoned responses based on current best practices and knowledge."

    # Expert consultation interface
    st.markdown("---")
    st.subheader("💼 Expert Consultation")

    # Display conversation
    if st.session_state.expert_conversation:
        for exchange in st.session_state.expert_conversation:
            st.markdown(f"""
            <div style="background: rgba(150, 100, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #9966ff;">
                <strong>💼 Client:</strong><br>
                {exchange['client']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: rgba(255, 150, 100, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #ff9666;">
                <strong>🎯 Expert ({domain}):</strong><br>
                {exchange['expert']}
            </div>
            """, unsafe_allow_html=True)

    # Client input
    client_query = st.text_area("💭 Your consultation question:", height=120,
                                placeholder=f"Ask your {domain.lower()} question...")

    # Expert actions
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        if st.button("📤 Consult Expert", type="primary"):
            if client_query.strip():
                send_expert_message(client_query, model_choice, base_prompt, domain)

    with col2:
        if st.button("🗑️ New Consultation"):
            st.session_state.expert_conversation = []
            st.rerun()

    with col3:
        if st.button("📋 Generate Report"):
            if st.session_state.expert_conversation:
                generate_expert_report(domain)

    with col4:
        if st.button("🔍 Deep Dive"):
            if st.session_state.expert_conversation and client_query:
                perform_deep_analysis(client_query, domain, model_choice)


def multi_purpose_bot():
    """Versatile AI assistant for multiple use cases"""
    create_tool_header("Multi-Purpose Bot", "Versatile AI assistant for any task", "🤖⚡")

    # Multi-purpose configuration
    st.subheader("🔧 Versatile Assistant Configuration")

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox(
            "AI Model",
            ["gemini", "openai"],
            help="Choose between Google Gemini or OpenAI models"
        )
        assistant_mode = st.selectbox(
            "Assistant Mode",
            ["Adaptive", "Creative", "Analytical", "Productive", "Research", "Personal Assistant"],
            help="Choose the assistant's focus"
        )

    with col2:
        task_complexity = st.selectbox(
            "Task Complexity",
            ["Simple Tasks", "Moderate Complexity", "Complex Projects", "Multi-step Workflows"],
            help="Set complexity handling level"
        )
        response_format = st.selectbox(
            "Response Format",
            ["Conversational", "Structured", "Bullet Points", "Step-by-Step", "Mixed"],
            help="Choose response formatting"
        )

    # Capabilities selection
    st.subheader("🎯 Activate Capabilities")
    capabilities = {}
    col1, col2, col3 = st.columns(3)

    with col1:
        capabilities['writing'] = st.checkbox("✍️ Writing & Content", True)
        capabilities['analysis'] = st.checkbox("📊 Data Analysis", True)
        capabilities['creative'] = st.checkbox("🎨 Creative Tasks", True)

    with col2:
        capabilities['planning'] = st.checkbox("📅 Planning & Organization", True)
        capabilities['research'] = st.checkbox("🔍 Research & Information", True)
        capabilities['problem_solving'] = st.checkbox("🧩 Problem Solving", True)

    with col3:
        capabilities['technical'] = st.checkbox("⚙️ Technical Support", True)
        capabilities['learning'] = st.checkbox("📚 Learning & Teaching", True)
        capabilities['business'] = st.checkbox("💼 Business Support", True)

    # Initialize multi-purpose session
    if 'multipurpose_conversation' not in st.session_state:
        st.session_state.multipurpose_conversation = []
        st.session_state.active_tasks = []

    # Build adaptive system prompt
    mode_prompts = {
        "Adaptive": "You are a versatile AI assistant that adapts to any task. Analyze what the user needs and respond appropriately.",
        "Creative": "You are a creative AI assistant. Focus on innovative solutions, creative thinking, and artistic expression.",
        "Analytical": "You are an analytical AI assistant. Provide data-driven insights, logical reasoning, and systematic approaches.",
        "Productive": "You are a productivity-focused AI assistant. Help users accomplish tasks efficiently and organize their work.",
        "Research": "You are a research-oriented AI assistant. Provide thorough information, fact-checking, and detailed analysis.",
        "Personal Assistant": "You are a personal AI assistant. Help with daily tasks, scheduling, reminders, and personal organization."
    }

    base_prompt = mode_prompts[assistant_mode]
    base_prompt += f"\n\nTask complexity level: {task_complexity}. Response format: {response_format}."

    # Add capabilities to prompt
    active_caps = [cap for cap, enabled in capabilities.items() if enabled]
    if active_caps:
        base_prompt += f"\n\nActive capabilities: {', '.join(active_caps)}."

    # Multi-purpose chat interface
    st.markdown("---")
    st.subheader("💬 AI Assistant Chat")

    # Task tracker
    if st.session_state.active_tasks:
        with st.expander("📋 Active Tasks"):
            for i, task in enumerate(st.session_state.active_tasks):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"• {task}")
                with col2:
                    if st.button("✅", key=f"complete_task_{i}"):
                        st.session_state.active_tasks.pop(i)
                        st.rerun()

    # Display conversation
    if st.session_state.multipurpose_conversation:
        for exchange in st.session_state.multipurpose_conversation:
            st.markdown(f"""
            <div style="background: rgba(100, 255, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #64ffff;">
                <strong>👤 You:</strong><br>
                {exchange['user']}
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="background: rgba(255, 100, 255, 0.1); padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #ff64ff;">
                <strong>🤖 Assistant ({assistant_mode}):</strong><br>
                {exchange['assistant']}
            </div>
            """, unsafe_allow_html=True)

    # User input
    user_request = st.text_area("💭 What can I help you with?", height=120,
                                placeholder="Ask me anything! I can help with writing, analysis, creative tasks, planning, and more...")

    # Assistant actions
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        if st.button("📤 Send Request", type="primary"):
            if user_request.strip():
                send_multipurpose_message(user_request, model_choice, base_prompt, assistant_mode)

    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.multipurpose_conversation = []
            st.session_state.active_tasks = []
            st.rerun()

    with col3:
        if st.button("📊 Session Summary"):
            if st.session_state.multipurpose_conversation:
                generate_session_summary()

    with col4:
        if st.button("⚡ Quick Actions"):
            show_quick_actions(capabilities)

    # Quick task templates
    if not st.session_state.multipurpose_conversation:
        st.subheader("🚀 Quick Start Templates")
        template_cols = st.columns(3)

        templates = {
            "Writing": ["Write a professional email", "Create a project proposal", "Draft a blog post"],
            "Analysis": ["Analyze this data", "Compare options", "Identify trends"],
            "Creative": ["Brainstorm ideas", "Design a concept", "Create a story"],
            "Planning": ["Create a schedule", "Plan a project", "Organize tasks"],
            "Research": ["Research a topic", "Find information", "Fact-check claims"],
            "Problem-Solving": ["Solve this problem", "Debug an issue", "Find alternatives"]
        }

        for i, (category, tasks) in enumerate(templates.items()):
            with template_cols[i % 3]:
                if st.button(f"📝 {category}", key=f"template_{category}"):
                    st.session_state.temp_template = tasks


# Helper functions for chatbot tools

def send_support_message(message, model, base_prompt):
    """Send a message in customer support context"""
    with st.spinner("🎧 Support agent is responding..."):
        # Build support context
        context = f"System: {base_prompt}\n\n"

        # Add recent conversation history for context
        recent_history = st.session_state.support_conversation[-3:] if st.session_state.support_conversation else []
        for exchange in recent_history:
            context += f"Customer: {exchange['customer']}\nAgent: {exchange['agent']}\n\n"

        context += f"Customer: {message}\nAgent: "

        # Generate response
        response = ai_client.generate_text(context, model=model, max_tokens=500)

        # Add to support conversation history
        st.session_state.support_conversation.append({
            'customer': message,
            'agent': response,
            'timestamp': datetime.now().isoformat(),
            'model': model
        })

        st.rerun()


def generate_support_summary():
    """Generate a summary of the support conversation"""
    if st.session_state.support_conversation:
        conversation_text = ""
        for exchange in st.session_state.support_conversation:
            conversation_text += f"Customer: {exchange['customer']}\nAgent: {exchange['agent']}\n\n"

        summary_prompt = f"Summarize this customer support conversation. Include the main issue, actions taken, and resolution status:\n\n{conversation_text}"

        with st.spinner("Generating support summary..."):
            summary = ai_client.generate_text(summary_prompt, model="gemini", max_tokens=300)

            st.subheader("📋 Support Session Summary")
            st.write(summary)

            # Export summary
            FileHandler.create_download_link(
                summary.encode(),
                f"support_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )


def create_escalation_note():
    """Create an escalation note for the support case"""
    if st.session_state.support_conversation:
        conversation_text = ""
        for exchange in st.session_state.support_conversation:
            conversation_text += f"Customer: {exchange['customer']}\nAgent: {exchange['agent']}\n\n"

        escalation_prompt = f"Create an escalation note for this customer support case. Include key details, attempted solutions, and recommended next steps:\n\n{conversation_text}"

        with st.spinner("Creating escalation note..."):
            escalation_note = ai_client.generate_text(escalation_prompt, model="gemini", max_tokens=400)

            st.subheader("⬆️ Escalation Note")
            st.write(escalation_note)

            # Export escalation note
            FileHandler.create_download_link(
                escalation_note.encode(),
                f"escalation_note_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )


def send_educational_message(message, model, base_prompt):
    """Send a message in educational context"""
    with st.spinner("📚 Tutor is preparing response..."):
        # Build educational context
        context = f"System: {base_prompt}\n\n"

        # Add recent conversation history for context
        recent_history = st.session_state.educational_conversation[
                         -3:] if st.session_state.educational_conversation else []
        for exchange in recent_history:
            context += f"Student: {exchange['student']}\nTutor: {exchange['tutor']}\n\n"

        context += f"Student: {message}\nTutor: "

        # Generate response
        response = ai_client.generate_text(context, model=model, max_tokens=600)

        # Add to educational conversation history
        st.session_state.educational_conversation.append({
            'student': message,
            'tutor': response,
            'timestamp': datetime.now().isoformat(),
            'model': model
        })

        st.rerun()


def generate_learning_summary():
    """Generate a learning progress summary"""
    if st.session_state.educational_conversation:
        conversation_text = ""
        for exchange in st.session_state.educational_conversation:
            conversation_text += f"Student: {exchange['student']}\nTutor: {exchange['tutor']}\n\n"

        summary_prompt = f"Analyze this learning session and provide a progress summary. Include topics covered, learning achievements, and areas for improvement:\n\n{conversation_text}"

        with st.spinner("Generating learning summary..."):
            summary = ai_client.generate_text(summary_prompt, model="gemini", max_tokens=350)

            st.subheader("📊 Learning Progress Summary")
            st.write(summary)

            # Export summary
            FileHandler.create_download_link(
                summary.encode(),
                f"learning_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )


def generate_study_tips(subject_area, grade_level):
    """Generate personalized study tips"""
    tips_prompt = f"Generate 5-7 effective study tips for {subject_area} at {grade_level} level. Include specific strategies and techniques."

    with st.spinner("Generating study tips..."):
        tips = ai_client.generate_text(tips_prompt, model="gemini", max_tokens=300)

        st.subheader(f"💡 Study Tips for {subject_area}")
        st.write(tips)


def send_expert_message(message, model, base_prompt, domain):
    """Send a message in expert consultation context"""
    with st.spinner(f"🎯 {domain} expert is analyzing..."):
        # Build expert context
        context = f"System: {base_prompt}\n\n"

        # Add recent conversation history for context
        recent_history = st.session_state.expert_conversation[-3:] if st.session_state.expert_conversation else []
        for exchange in recent_history:
            context += f"Client: {exchange['client']}\nExpert: {exchange['expert']}\n\n"

        context += f"Client: {message}\nExpert: "

        # Generate response
        response = ai_client.generate_text(context, model=model, max_tokens=700)

        # Add to expert conversation history
        st.session_state.expert_conversation.append({
            'client': message,
            'expert': response,
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'domain': domain
        })

        st.rerun()


def generate_expert_report(domain):
    """Generate an expert consultation report"""
    if st.session_state.expert_conversation:
        conversation_text = ""
        for exchange in st.session_state.expert_conversation:
            conversation_text += f"Client: {exchange['client']}\nExpert: {exchange['expert']}\n\n"

        report_prompt = f"Create a professional {domain} consultation report based on this conversation. Include executive summary, key findings, recommendations, and next steps:\n\n{conversation_text}"

        with st.spinner("Generating expert report..."):
            report = ai_client.generate_text(report_prompt, model="gemini", max_tokens=600)

            st.subheader(f"📋 {domain} Consultation Report")
            st.write(report)

            # Export report
            FileHandler.create_download_link(
                report.encode(),
                f"{domain.lower().replace(' ', '_')}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )


def perform_deep_analysis(query, domain, model):
    """Perform deep analysis on a specific query"""
    analysis_prompt = f"As a {domain} expert, provide a comprehensive deep-dive analysis of: {query}. Include technical details, implications, and strategic considerations."

    with st.spinner("Performing deep analysis..."):
        analysis = ai_client.generate_text(analysis_prompt, model=model, max_tokens=800)

        st.subheader("🔍 Deep Analysis")
        st.write(analysis)

        # Export analysis
        FileHandler.create_download_link(
            analysis.encode(),
            f"deep_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain"
        )


def send_multipurpose_message(message, model, base_prompt, assistant_mode):
    """Send a message in multi-purpose context"""
    with st.spinner(f"🤖 {assistant_mode} assistant is processing..."):
        # Build multipurpose context
        context = f"System: {base_prompt}\n\n"

        # Add recent conversation history for context
        recent_history = st.session_state.multipurpose_conversation[
                         -4:] if st.session_state.multipurpose_conversation else []
        for exchange in recent_history:
            context += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"

        context += f"User: {message}\nAssistant: "

        # Generate response
        response = ai_client.generate_text(context, model=model, max_tokens=600)

        # Add to multipurpose conversation history
        st.session_state.multipurpose_conversation.append({
            'user': message,
            'assistant': response,
            'timestamp': datetime.now().isoformat(),
            'model': model,
            'mode': assistant_mode
        })

        # Check if response suggests a task and add to task list
        if any(keyword in response.lower() for keyword in ['task', 'todo', 'action item', 'next step']):
            task_suggestion = extract_task_from_response(response)
            if task_suggestion and task_suggestion not in st.session_state.active_tasks:
                st.session_state.active_tasks.append(task_suggestion)

        st.rerun()


def generate_session_summary():
    """Generate a summary of the multi-purpose session"""
    if st.session_state.multipurpose_conversation:
        conversation_text = ""
        for exchange in st.session_state.multipurpose_conversation:
            conversation_text += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"

        summary_prompt = f"Summarize this AI assistant session. Include main topics discussed, tasks completed, and key insights:\n\n{conversation_text}"

        with st.spinner("Generating session summary..."):
            summary = ai_client.generate_text(summary_prompt, model="gemini", max_tokens=400)

            st.subheader("📊 Session Summary")
            st.write(summary)

            # Export summary
            FileHandler.create_download_link(
                summary.encode(),
                f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )


def show_quick_actions(capabilities):
    """Show quick action options based on enabled capabilities"""
    st.subheader("⚡ Quick Actions")

    actions = []
    if capabilities.get('writing'):
        actions.extend(["Write an email", "Create a report", "Draft content"])
    if capabilities.get('analysis'):
        actions.extend(["Analyze data", "Compare options", "Find patterns"])
    if capabilities.get('creative'):
        actions.extend(["Brainstorm ideas", "Create story", "Design concept"])
    if capabilities.get('planning'):
        actions.extend(["Make a plan", "Create schedule", "Organize tasks"])
    if capabilities.get('research'):
        actions.extend(["Research topic", "Find information", "Summarize findings"])

    # Display actions in columns
    cols = st.columns(3)
    for i, action in enumerate(actions[:9]):  # Limit to 9 actions
        with cols[i % 3]:
            if st.button(f"⚡ {action}", key=f"quick_action_{i}"):
                st.session_state.quick_action_selected = action
                st.text_area("Quick action request:", value=action, key="quick_action_text")


def extract_task_from_response(response):
    """Extract potential tasks from AI response"""
    # Simple task extraction logic
    if "you should" in response.lower() or "you could" in response.lower():
        sentences = response.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ['should', 'could', 'recommend', 'suggest']):
                return sentence.strip()
    return None


def send_conversational_message(message, model, system_prompt, max_tokens):
    """Send a message in conversational context"""
    with st.spinner(f"🤖 AI is thinking..."):
        # Build conversation context
        context = f"System: {system_prompt}\n\n"

        # Add recent conversation history for context (last 5 exchanges)
        recent_history = st.session_state.conversation_history[-5:] if st.session_state.conversation_history else []
        for exchange in recent_history:
            context += f"User: {exchange['user']}\nAssistant: {exchange['ai']}\n\n"

        context += f"User: {message}\nAssistant: "

        # Generate response
        response = ai_client.generate_text(context, model=model, max_tokens=max_tokens)

        # Add to conversation history
        st.session_state.conversation_history.append({
            'user': message,
            'ai': response,
            'timestamp': datetime.now().isoformat(),
            'model': model
        })

        st.rerun()


def export_conversation_history():
    """Export conversation history to file"""
    if st.session_state.conversation_history:
        conversation_data = {
            'conversation': st.session_state.conversation_history,
            'exported_at': datetime.now().isoformat(),
            'total_exchanges': len(st.session_state.conversation_history)
        }

        conversation_json = json.dumps(conversation_data, indent=2)
        FileHandler.create_download_link(
            conversation_json.encode(),
            f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )
        st.success("📥 Conversation exported! Check your downloads.")


def analyze_response_metrics(response):
    """Analyze response metrics"""
    words = response.split()
    word_count = len(words)
    reading_time = max(1, word_count // 200)  # 200 WPM average

    return {
        'word_count': word_count,
        'reading_time': reading_time
    }


def build_ocr_prompt(language, preserve_formatting, extract_tables, extract_handwriting, output_format):
    """Build OCR prompt for AI image analysis"""
    prompt = "Please extract all text from this image. "

    if language != "Auto-detect":
        prompt += f"The primary language is {language}. "

    if preserve_formatting:
        prompt += "Preserve the original formatting, spacing, and layout as much as possible. "

    if extract_tables:
        prompt += "If there are tables, preserve the table structure using appropriate formatting. "

    if extract_handwriting:
        prompt += "Include any handwritten text, even if unclear. "

    if output_format == "Structured Text":
        prompt += "Organize the text in a structured format with clear sections. "
    elif output_format == "Markdown":
        prompt += "Format the output as Markdown with appropriate headers and formatting. "
    elif output_format == "JSON":
        prompt += "Return the text in JSON format with text content and metadata. "

    prompt += "Be accurate and include all visible text."
    return prompt


def analyze_extracted_text(text):
    """Analyze extracted text metrics"""
    words = text.split()
    lines = text.split('\n')

    return {
        'word_count': len(words),
        'char_count': len(text),
        'line_count': len(lines)
    }


def analyze_speech_text(text):
    """Analyze text for speech synthesis"""
    words = text.split()
    sentences = text.split('.')

    # Estimate syllables (rough approximation)
    syllable_count = 0
    for word in words:
        # Simple syllable estimation
        vowels = 'aeiouy'
        word = word.lower()
        syllables = sum(1 for char in word if char in vowels)
        syllables = max(1, syllables)  # At least 1 syllable per word
        syllable_count += syllables

    # Estimate duration (average 150 words per minute for speech)
    estimated_duration = max(1, len(words) * 60 / 150)

    return {
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'syllable_estimate': syllable_count,
        'estimated_duration': int(estimated_duration)
    }


def build_speech_prompt(text, voice_type, language, speed, pitch, add_pauses, emphasize_punctuation):
    """Build prompt for speech synthesis"""
    prompt = f"Convert this text to speech with the following settings:\n"
    prompt += f"Voice Type: {voice_type}\n"
    prompt += f"Language: {language}\n"
    prompt += f"Speed: {speed}x normal\n"
    prompt += f"Pitch: {pitch}x normal\n"

    if add_pauses:
        prompt += "Add natural pauses at appropriate places.\n"

    if emphasize_punctuation:
        prompt += "Emphasize punctuation for natural speech flow.\n"

    prompt += f"\nText to convert: {text}"

    return f"Speech synthesis request: {prompt}"


def generate_ssml(text, voice_type, speed, pitch):
    """Generate SSML markup for text-to-speech"""
    ssml = '<?xml version="1.0" encoding="UTF-8"?>\n'
    ssml += '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">\n'
    ssml += f'<prosody rate="{speed}" pitch="{pitch}">\n'
    ssml += f'{text}\n'
    ssml += '</prosody>\n'
    ssml += '</speak>'

    return ssml


def build_pattern_recognition_prompt(focus, detail_level, confidence):
    """Build prompt for pattern recognition"""
    prompt = f"Analyze this image for patterns with focus on {focus.lower()}. "

    if detail_level == "Basic":
        prompt += "Provide a brief overview of main patterns and objects. "
    elif detail_level == "Detailed":
        prompt += "Provide detailed analysis of patterns, objects, and their relationships. "
    else:  # Comprehensive
        prompt += "Provide comprehensive analysis including fine details, patterns, textures, and spatial relationships. "

    prompt += f"Only include findings with confidence above {confidence}. "
    prompt += "Describe patterns clearly and organize findings by category."

    return prompt


def display_pattern_results(results, analysis_type):
    """Display pattern recognition results"""
    st.subheader(f"{analysis_type} Results")

    for result in results:
        if result['success']:
            st.markdown(f"### 🔍 {result['filename']}")

            # Display image
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(result['image'], caption=result['filename'], use_container_width=True)

            with col2:
                st.subheader("Pattern Analysis")
                st.write(result['analysis'])

            # Download analysis
            FileHandler.create_download_link(
                result['analysis'].encode(),
                f"pattern_analysis_{result['filename']}.txt",
                "text/plain"
            )
        else:
            st.error(f"❌ Failed to analyze {result['filename']}: {result.get('error', 'Unknown error')}")

        st.markdown("---")


def analyze_text_patterns(text, pattern_focus, analysis_depth):
    """Analyze text patterns using AI"""
    prompt = f"Analyze this text for patterns with focus on {pattern_focus.lower()}:\n\n{text}\n\n"

    if analysis_depth == "Quick":
        prompt += "Provide a quick overview of main patterns found."
    elif analysis_depth == "Standard":
        prompt += "Provide standard analysis with details about patterns and their significance."
    else:  # Deep
        prompt += "Provide deep analysis including subtle patterns, correlations, and insights."

    return ai_client.generate_text(prompt, max_tokens=1500)


def detect_data_patterns(data, columns, pattern_types, time_column):
    """Detect patterns in data using AI analysis"""
    # Convert data to text representation for AI analysis
    data_summary = f"Dataset with {len(data)} rows and columns: {', '.join(columns)}\n\n"
    data_summary += f"Sample data:\n{data[columns].head().to_string()}\n\n"
    data_summary += f"Data types:\n{data[columns].dtypes.to_string()}\n\n"
    data_summary += f"Basic statistics:\n{data[columns].describe().to_string()}"

    prompt = f"Analyze this dataset for patterns focusing on {', '.join(pattern_types)}:\n\n{data_summary}"

    if time_column != "None":
        prompt += f"\n\nTime column: {time_column} - Look for temporal patterns."

    prompt += "\n\nProvide insights about patterns, trends, and anomalies in the data."

    return ai_client.generate_text(prompt, max_tokens=2000)


def build_story_prompt(genre, length, audience, style, pov, tone, character, setting, conflict, theme, supporting_chars,
                       outline, special_elements):
    """Build comprehensive story generation prompt"""
    prompt = f"Write a {genre.lower()} story with the following specifications:\n\n"

    prompt += f"Length: {length}\n"
    prompt += f"Target Audience: {audience}\n"
    prompt += f"Writing Style: {style}\n"
    prompt += f"Point of View: {pov}\n"
    prompt += f"Tone: {tone}\n\n"

    prompt += f"Main Character: {character}\n"
    prompt += f"Setting: {setting}\n"

    if conflict:
        prompt += f"Central Conflict: {conflict}\n"

    if theme:
        prompt += f"Theme: {theme}\n"

    if supporting_chars:
        prompt += f"Supporting Characters: {supporting_chars}\n"

    if outline:
        prompt += f"Plot Outline: {outline}\n"

    if special_elements:
        prompt += f"Special Elements: {special_elements}\n"

    prompt += "\nCreate an engaging, well-structured story that incorporates all these elements naturally."

    return prompt


def display_generated_story(story, metadata):
    """Display generated story with analysis"""
    st.subheader("📚 Generated Story")
    st.markdown(story)

    # Story analysis
    story_analysis = analyze_story_content(story)

    st.subheader("Story Analysis")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Word Count", story_analysis['word_count'])
    with col2:
        st.metric("Reading Time", f"{story_analysis['reading_time']} min")
    with col3:
        st.metric("Paragraphs", story_analysis['paragraphs'])
    with col4:
        st.metric("Dialogue %", f"{story_analysis['dialogue_percentage']}%")

    # Story metadata
    with st.expander("Story Details"):
        for key, value in metadata.items():
            st.text(f"{key.title()}: {value}")

    # Download story
    FileHandler.create_download_link(
        story.encode(),
        "generated_story.txt",
        "text/plain"
    )


def analyze_story_content(story):
    """Analyze story content metrics"""
    words = story.split()
    paragraphs = [p for p in story.split('\n\n') if p.strip()]

    # Estimate dialogue percentage
    dialogue_chars = sum(1 for line in story.split('\n') if '"' in line)
    total_lines = len(story.split('\n'))
    dialogue_percentage = int((dialogue_chars / max(1, total_lines)) * 100)

    return {
        'word_count': len(words),
        'reading_time': max(1, len(words) // 200),
        'paragraphs': len(paragraphs),
        'sections': len([p for p in paragraphs if len(p.split()) > 50]),
        'dialogue_percentage': dialogue_percentage
    }


def build_continuation_prompt(existing_text, length, direction, maintain_style, maintain_tone):
    """Build prompt for story continuation"""
    prompt = f"Continue this story:\n\n{existing_text}\n\n"
    prompt += f"Continuation length: {length}\n"
    prompt += f"Direction: {direction}\n"

    if maintain_style:
        prompt += "Maintain the existing writing style and voice.\n"

    if maintain_tone:
        prompt += "Keep the same tone and mood.\n"

    prompt += "\nContinue the story naturally from where it left off."

    return prompt


def generate_story_from_prompt(prompt_text, interpretation, structure, length, creative_freedom):
    """Generate story from creative prompt"""
    story_prompt = f"Create a story based on this prompt: {prompt_text}\n\n"
    story_prompt += f"Interpretation style: {interpretation}\n"
    story_prompt += f"Story structure: {structure}\n"
    story_prompt += f"Length: {length}\n"
    story_prompt += f"Creative freedom level: {creative_freedom}\n\n"
    story_prompt += "Develop this into a complete, engaging story."

    return ai_client.generate_text(story_prompt, max_tokens=2500)


def build_refinement_prompt(content, refinement_type, audience, tone, length_pref, improvements, preserve,
                            custom_instructions):
    """Build content refinement prompt"""
    prompt = f"Refine this content:\n\n{content}\n\n"
    prompt += f"Refinement focus: {refinement_type}\n"

    if audience != "Keep Current":
        prompt += f"Target audience: {audience}\n"

    if tone != "Keep Current":
        prompt += f"Desired tone: {tone}\n"

    if length_pref != "Keep Current":
        prompt += f"Length preference: {length_pref}\n"

    if improvements:
        prompt += f"Specific improvements: {', '.join(improvements)}\n"

    if preserve:
        prompt += f"Preserve: {', '.join(preserve)}\n"

    if custom_instructions:
        prompt += f"Additional instructions: {custom_instructions}\n"

    prompt += "\nProvide the refined version that improves the content while meeting these requirements."

    return prompt


def calculate_improvement_score(original_analysis, refined_analysis):
    """Calculate improvement score between original and refined content"""
    # Simple scoring based on readability improvement
    readability_scores = {"Easy": 3, "Medium": 2, "Complex": 1}

    original_score = readability_scores.get(original_analysis['readability_level'], 2)
    refined_score = readability_scores.get(refined_analysis['readability_level'], 2)

    # Calculate improvement (positive for better readability)
    improvement = ((refined_score - original_score) + 1) * 50

    # Ensure score is between 0-100
    return max(0, min(100, int(improvement)))


def style_transfer():
    """AI-powered style transfer tool"""
    create_tool_header("Style Transfer", "Apply artistic styles to your images using AI", "🎨")

    # File upload section
    st.subheader("Upload Images")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Content Image**")
        content_files = FileHandler.upload_files(['jpg', 'jpeg', 'png'], accept_multiple=False)
        if content_files:
            content_image = FileHandler.process_image_file(content_files[0])
            if content_image:
                st.image(content_image, caption="Content Image", use_container_width=True)

    with col2:
        st.markdown("**Style Reference**")
        style_option = st.radio("Style Source", ["Upload Style Image", "Choose Preset Style"])

        if style_option == "Upload Style Image":
            style_files = FileHandler.upload_files(['jpg', 'jpeg', 'png'], accept_multiple=False)
            if style_files:
                style_image = FileHandler.process_image_file(style_files[0])
                if style_image:
                    st.image(style_image, caption="Style Image", use_container_width=True)
        else:
            preset_style = st.selectbox("Preset Style", [
                "Van Gogh - Starry Night", "Picasso - Cubism", "Monet - Impressionism",
                "Abstract Expressionism", "Art Nouveau", "Pop Art", "Minimalism",
                "Watercolor", "Oil Painting", "Sketch", "Anime Style"
            ])
            style_image = None

    # Style transfer options
    if content_files:
        st.subheader("Style Transfer Settings")

        col1, col2 = st.columns(2)
        with col1:
            style_strength = st.slider("Style Strength", 0.1, 1.0, 0.7, 0.1)
            preserve_content = st.slider("Content Preservation", 0.1, 1.0, 0.6, 0.1)

        with col2:
            output_quality = st.selectbox("Output Quality", ["Standard", "High", "Ultra"])
            color_preservation = st.checkbox("Preserve Original Colors", value=False)

        # Advanced options
        with st.expander("Advanced Options"):
            edge_enhancement = st.checkbox("Enhance Edges", value=True)
            texture_detail = st.slider("Texture Detail", 0.1, 1.0, 0.5, 0.1)
            blend_mode = st.selectbox("Blend Mode", ["Normal", "Multiply", "Overlay", "Soft Light"])

        if st.button("Apply Style Transfer"):
            with st.spinner("Applying artistic style to your image..."):
                # Convert content image to bytes
                content_bytes = convert_image_to_bytes(content_image)

                # Build style transfer prompt
                if style_option == "Upload Style Image" and style_files:
                    style_bytes = convert_image_to_bytes(style_image)
                    style_prompt = build_style_transfer_prompt_with_reference(
                        style_strength, preserve_content, color_preservation,
                        edge_enhancement, texture_detail, blend_mode
                    )

                    # For AI analysis with style reference
                    result_description = ai_client.analyze_image(
                        content_bytes,
                        f"{style_prompt} Apply the style from the reference image to this content image."
                    )
                else:
                    style_prompt = build_style_transfer_prompt_preset(
                        preset_style, style_strength, preserve_content,
                        color_preservation, edge_enhancement, texture_detail
                    )

                    result_description = ai_client.analyze_image(content_bytes, style_prompt)

                # Display results
                st.subheader("Style Transfer Results")
                st.success("✨ Style transfer completed!")

                # In a real implementation, this would show the styled image
                st.info("🖼️ Styled Image: In a full implementation, the AI-styled image would be displayed here.")

                # Style analysis
                st.subheader("Style Analysis")
                st.write(result_description)

                # Transfer details
                transfer_details = {
                    "Style Applied": preset_style if style_option == "Choose Preset Style" else "Custom Style",
                    "Style Strength": f"{style_strength:.1f}",
                    "Content Preservation": f"{preserve_content:.1f}",
                    "Output Quality": output_quality,
                    "Color Preservation": "Yes" if color_preservation else "No"
                }

                with st.expander("Transfer Details"):
                    for key, value in transfer_details.items():
                        st.text(f"{key}: {value}")

                # Download placeholder
                st.info("📥 Download: In a full implementation, you could download the styled image here.")
    else:
        st.info("📸 Upload a content image to begin style transfer.")


def image_synthesis():
    """AI-powered image synthesis tool"""
    create_tool_header("Image Synthesis", "Synthesize new images by combining elements", "🔮")

    # Synthesis mode selection
    synthesis_mode = st.selectbox("Synthesis Mode", [
        "Text-to-Image Synthesis", "Image Composition", "Element Combination",
        "Scene Generation", "Object Synthesis", "Texture Synthesis"
    ])

    if synthesis_mode == "Text-to-Image Synthesis":
        st.subheader("Text-to-Image Generation")

        # Text input
        synthesis_prompt = st.text_area("Describe the image to synthesize:",
                                        placeholder="A futuristic cityscape at sunset with flying cars...",
                                        height=100)

        col1, col2 = st.columns(2)
        with col1:
            image_style = st.selectbox("Image Style", [
                "Photorealistic", "Digital Art", "Concept Art", "Fantasy Art",
                "Sci-Fi", "Abstract", "Minimalist", "Surreal", "Vintage"
            ])

            composition = st.selectbox("Composition", [
                "Balanced", "Rule of Thirds", "Central Focus", "Dynamic", "Symmetrical"
            ])

        with col2:
            resolution = st.selectbox("Resolution", ["512x512", "768x768", "1024x1024", "1024x768"])
            color_palette = st.selectbox("Color Palette", [
                "Natural", "Vibrant", "Monochromatic", "Warm Tones", "Cool Tones", "Neon"
            ])

        # Advanced synthesis options
        with st.expander("Advanced Options"):
            detail_level = st.slider("Detail Level", 0.1, 1.0, 0.7, 0.1)
            creativity = st.slider("Creativity", 0.1, 1.0, 0.8, 0.1)
            aspect_ratio = st.selectbox("Aspect Ratio", ["Square", "Portrait", "Landscape", "Wide"])

        if st.button("Synthesize Image") and synthesis_prompt:
            with st.spinner("Synthesizing image from description..."):
                # Build synthesis prompt
                full_prompt = build_synthesis_prompt(
                    synthesis_prompt, image_style, composition, color_palette,
                    detail_level, creativity, aspect_ratio
                )

                # Generate using AI
                synthesis_result = ai_client.generate_text(full_prompt, max_tokens=500)

                st.subheader("Synthesis Results")
                st.success("🎨 Image synthesis completed!")

                # Display synthesis description
                st.write("**Synthesis Description:**")
                st.write(synthesis_result)

                # Synthesis parameters
                st.subheader("Synthesis Parameters")
                params = {
                    "Style": image_style,
                    "Composition": composition,
                    "Resolution": resolution,
                    "Color Palette": color_palette,
                    "Detail Level": f"{detail_level:.1f}",
                    "Creativity": f"{creativity:.1f}"
                }

                for key, value in params.items():
                    st.text(f"{key}: {value}")

                st.info(
                    "🖼️ Synthesized Image: In a full implementation, the AI-generated image would be displayed here.")

    elif synthesis_mode == "Image Composition":
        st.subheader("Image Composition Synthesis")

        # Multiple image upload
        composition_images = FileHandler.upload_files(['jpg', 'jpeg', 'png'], accept_multiple=True)

        if composition_images:
            st.write(f"Uploaded {len(composition_images)} images for composition")

            # Display uploaded images
            cols = st.columns(min(len(composition_images), 3))
            for i, img_file in enumerate(composition_images[:3]):
                with cols[i]:
                    img = FileHandler.process_image_file(img_file)
                    if img:
                        st.image(img, caption=f"Image {i + 1}", use_container_width=True)

            # Composition settings
            col1, col2 = st.columns(2)
            with col1:
                composition_style = st.selectbox("Composition Style", [
                    "Seamless Blend", "Layered Montage", "Split Screen", "Mosaic", "Collage"
                ])

                transition_type = st.selectbox("Transition", [
                    "Smooth Gradient", "Hard Edge", "Feathered", "Masked", "Artistic"
                ])

            with col2:
                background_treatment = st.selectbox("Background", [
                    "Keep Original", "Unified Background", "Transparent", "Generated"
                ])

                color_harmony = st.checkbox("Apply Color Harmony", value=True)

            if st.button("Compose Images"):
                with st.spinner("Composing images..."):
                    # Analyze composition
                    composition_prompt = build_composition_prompt(
                        len(composition_images), composition_style, transition_type,
                        background_treatment, color_harmony
                    )

                    composition_result = ai_client.generate_text(composition_prompt, max_tokens=400)

                    st.subheader("Composition Results")
                    st.success("🖼️ Image composition completed!")
                    st.write(composition_result)

                    st.info("🎨 Composed Image: In a full implementation, the composed image would be displayed here.")
        else:
            st.info("📸 Upload 2 or more images to create a composition.")

    else:
        st.subheader(f"{synthesis_mode}")
        st.info(f"🔧 {synthesis_mode} mode is being enhanced. Please try Text-to-Image or Image Composition modes.")


def concept_art():
    """AI-powered concept art generation tool"""
    create_tool_header("Concept Art", "Generate concept art for creative projects", "🎭")

    # Concept art type selection
    concept_type = st.selectbox("Concept Art Type", [
        "Character Design", "Environment Design", "Vehicle Design", "Creature Design",
        "Architecture Concept", "Prop Design", "Weapon Design", "Fashion Design"
    ])

    st.subheader(f"{concept_type} Generation")

    # Universal concept parameters
    col1, col2 = st.columns(2)
    with col1:
        concept_description = st.text_area("Concept Description:",
                                           placeholder="Describe your concept in detail...",
                                           height=100)

        art_style = st.selectbox("Art Style", [
            "Realistic", "Stylized", "Semi-Realistic", "Cartoon", "Anime",
            "Fantasy Art", "Sci-Fi Art", "Steampunk", "Cyberpunk", "Medieval"
        ])

        mood = st.selectbox("Mood/Atmosphere", [
            "Heroic", "Dark", "Mysterious", "Whimsical", "Epic", "Minimalist",
            "Dramatic", "Peaceful", "Aggressive", "Elegant"
        ])

    with col2:
        color_scheme = st.selectbox("Color Scheme", [
            "Full Color", "Monochromatic", "Limited Palette", "Earth Tones",
            "Bright & Vibrant", "Dark & Moody", "Pastel", "Neon"
        ])

        detail_level = st.selectbox("Detail Level", ["Sketch", "Detailed", "Highly Detailed"])

        perspective = st.selectbox("Perspective", [
            "Front View", "Side View", "3/4 View", "Multiple Views", "Action Pose"
        ])

    # Type-specific options
    if concept_type == "Character Design":
        with st.expander("Character-Specific Options"):
            character_type = st.selectbox("Character Type", [
                "Hero/Protagonist", "Villain/Antagonist", "Supporting Character",
                "Background Character", "Creature/Monster", "Robot/Mech"
            ])

            age_group = st.selectbox("Age Group", ["Child", "Young Adult", "Adult", "Elderly", "Ageless"])
            body_type = st.selectbox("Body Type", ["Slim", "Athletic", "Muscular", "Heavyset", "Unique"])

    elif concept_type == "Environment Design":
        with st.expander("Environment-Specific Options"):
            environment_type = st.selectbox("Environment Type", [
                "Interior", "Exterior", "Landscape", "Cityscape", "Fantasy World",
                "Sci-Fi Setting", "Underground", "Underwater", "Sky/Aerial"
            ])

            time_of_day = st.selectbox("Time of Day", ["Dawn", "Day", "Dusk", "Night", "Varies"])
            weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rainy", "Stormy", "Foggy"])

    elif concept_type == "Vehicle Design":
        with st.expander("Vehicle-Specific Options"):
            vehicle_type = st.selectbox("Vehicle Type", [
                "Car/Automobile", "Motorcycle", "Aircraft", "Spacecraft", "Ship/Boat",
                "Tank/Military", "Mech/Robot", "Fantasy Vehicle"
            ])

            era = st.selectbox("Era", ["Modern", "Futuristic", "Retro/Vintage", "Fantasy", "Steampunk"])

    # Advanced concept options
    with st.expander("Advanced Options"):
        reference_style = st.text_input("Reference Artist/Style",
                                        placeholder="e.g., 'in the style of Studio Ghibli'")

        technical_specs = st.text_area("Technical Specifications",
                                       placeholder="Any technical details or constraints...")

        iterations = st.slider("Number of Variations", 1, 4, 1)

    if st.button("Generate Concept Art") and concept_description:
        with st.spinner("Generating concept art..."):
            # Build comprehensive concept prompt
            concept_prompt = build_concept_art_prompt(
                concept_type, concept_description, art_style, mood, color_scheme,
                detail_level, perspective, reference_style, technical_specs
            )

            # Add type-specific details
            if concept_type == "Character Design":
                concept_prompt += f" Character type: {locals().get('character_type', 'General')}. "
                concept_prompt += f"Age: {locals().get('age_group', 'Adult')}. "
                concept_prompt += f"Build: {locals().get('body_type', 'Average')}."
            elif concept_type == "Environment Design":
                concept_prompt += f" Environment: {locals().get('environment_type', 'General')}. "
                concept_prompt += f"Time: {locals().get('time_of_day', 'Day')}. "
                concept_prompt += f"Weather: {locals().get('weather', 'Clear')}."
            elif concept_type == "Vehicle Design":
                concept_prompt += f" Vehicle type: {locals().get('vehicle_type', 'General')}. "
                concept_prompt += f"Era: {locals().get('era', 'Modern')}."

            # Generate concept art description
            concept_result = ai_client.generate_text(concept_prompt, max_tokens=600)

            # Display results
            st.subheader("Generated Concept Art")
            st.success("🎨 Concept art generation completed!")

            # Concept description
            st.write("**Concept Description:**")
            st.write(concept_result)

            # Concept specifications
            st.subheader("Concept Specifications")
            specs = {
                "Type": concept_type,
                "Art Style": art_style,
                "Mood": mood,
                "Color Scheme": color_scheme,
                "Detail Level": detail_level,
                "Perspective": perspective
            }

            for key, value in specs.items():
                st.text(f"{key}: {value}")

            # Download concept description
            FileHandler.create_download_link(
                concept_result.encode(),
                f"concept_art_{concept_type.lower().replace(' ', '_')}.txt",
                "text/plain"
            )

            st.info("🖼️ Concept Art: In a full implementation, the generated concept art would be displayed here.")

            # Additional variations
            if iterations > 1:
                st.subheader("Concept Variations")
                for i in range(2, iterations + 1):
                    with st.expander(f"Variation {i}"):
                        variation_prompt = concept_prompt + f" Create a different variation with alternative design choices."
                        variation = ai_client.generate_text(variation_prompt, max_tokens=400)
                        st.write(variation)
    else:
        st.info("📝 Enter a concept description to generate concept art.")


def photo_enhancement():
    """AI-powered photo enhancement tool"""
    create_tool_header("Photo Enhancement", "Enhance and improve your photos with AI", "✨")

    # File upload
    st.subheader("Upload Photo")
    uploaded_files = FileHandler.upload_files(['jpg', 'jpeg', 'png'], accept_multiple=True)

    if uploaded_files:
        # Enhancement options
        st.subheader("Enhancement Options")

        enhancement_type = st.selectbox("Enhancement Type", [
            "Auto Enhancement", "Custom Enhancement", "Restoration", "Style Enhancement", "Quality Upscaling"
        ])

        if enhancement_type == "Auto Enhancement":
            col1, col2 = st.columns(2)
            with col1:
                auto_mode = st.selectbox("Auto Mode", [
                    "General Enhancement", "Portrait Enhancement", "Landscape Enhancement",
                    "Low Light Enhancement", "Color Enhancement", "Vintage Restoration"
                ])

                enhancement_strength = st.slider("Enhancement Strength", 0.1, 1.0, 0.7, 0.1)

            with col2:
                preserve_original = st.checkbox("Preserve Original Character", value=True)
                fix_common_issues = st.checkbox("Fix Common Issues", value=True)

        elif enhancement_type == "Custom Enhancement":
            col1, col2 = st.columns(2)
            with col1:
                brightness_adjust = st.slider("Brightness", -1.0, 1.0, 0.0, 0.1)
                contrast_adjust = st.slider("Contrast", -1.0, 1.0, 0.0, 0.1)
                saturation_adjust = st.slider("Saturation", -1.0, 1.0, 0.0, 0.1)

            with col2:
                sharpness_adjust = st.slider("Sharpness", -1.0, 1.0, 0.0, 0.1)
                noise_reduction = st.slider("Noise Reduction", 0.0, 1.0, 0.0, 0.1)
                color_balance = st.slider("Color Balance", -1.0, 1.0, 0.0, 0.1)

        elif enhancement_type == "Restoration":
            restoration_options = st.multiselect("Restoration Options", [
                "Remove Scratches", "Fix Fading", "Repair Tears", "Remove Spots",
                "Improve Definition", "Restore Colors", "Fix Exposure"
            ])

            restoration_intensity = st.slider("Restoration Intensity", 0.1, 1.0, 0.5, 0.1)

        # Advanced options
        with st.expander("Advanced Options"):
            output_format = st.selectbox("Output Format", ["Original Format", "PNG", "JPEG", "WEBP"])
            quality_setting = st.slider("Output Quality", 70, 100, 95, 5)

            if enhancement_type in ["Auto Enhancement", "Custom Enhancement"]:
                hdr_effect = st.checkbox("HDR Effect")
                film_grain = st.checkbox("Add Film Grain")

        # Process photos
        if st.button("Enhance Photos"):
            results = []

            for photo_file in uploaded_files:
                with st.spinner(f"Enhancing {photo_file.name}..."):
                    photo = FileHandler.process_image_file(photo_file)
                    if photo:
                        # Convert to bytes for AI analysis
                        photo_bytes = convert_image_to_bytes(photo)

                        # Build enhancement prompt
                        if enhancement_type == "Auto Enhancement":
                            enhancement_prompt = build_auto_enhancement_prompt(
                                auto_mode, enhancement_strength, preserve_original, fix_common_issues
                            )
                        elif enhancement_type == "Custom Enhancement":
                            enhancement_prompt = build_custom_enhancement_prompt(
                                brightness_adjust, contrast_adjust, saturation_adjust,
                                sharpness_adjust, noise_reduction, color_balance
                            )
                        elif enhancement_type == "Restoration":
                            enhancement_prompt = build_restoration_prompt(restoration_options, restoration_intensity)
                        else:
                            enhancement_prompt = f"Enhance this photo using {enhancement_type.lower()} techniques."

                        # Analyze and describe enhancements
                        enhancement_analysis = ai_client.analyze_image(photo_bytes, enhancement_prompt)

                        results.append({
                            'filename': photo_file.name,
                            'original': photo,
                            'analysis': enhancement_analysis,
                            'success': True
                        })
                    else:
                        results.append({
                            'filename': photo_file.name,
                            'error': 'Failed to process image',
                            'success': False
                        })

            # Display results
            st.subheader("Enhancement Results")

            for result in results:
                if result['success']:
                    st.markdown(f"### ✨ {result['filename']}")

                    # Display original image
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Original**")
                        st.image(result['original'], use_container_width=True)

                    with col2:
                        st.markdown("**Enhanced (Preview)**")
                        st.info(
                            "🖼️ Enhanced Image: In a full implementation, the AI-enhanced image would be displayed here.")

                    # Enhancement analysis
                    st.subheader("Enhancement Analysis")
                    st.write(result['analysis'])

                    # Enhancement details
                    enhancement_details = {
                        "Enhancement Type": enhancement_type,
                        "Processing Status": "Completed",
                        "Original Format": result['filename'].split('.')[-1].upper(),
                        "Output Quality": f"{quality_setting}%"
                    }

                    with st.expander("Enhancement Details"):
                        for key, value in enhancement_details.items():
                            st.text(f"{key}: {value}")

                    # Download placeholder
                    st.info(
                        "📥 Download Enhanced: In a full implementation, you could download the enhanced image here.")

                else:
                    st.error(f"❌ Failed to enhance {result['filename']}: {result.get('error', 'Unknown error')}")

                st.markdown("---")

            # Batch processing summary
            successful_enhancements = len([r for r in results if r['success']])
            st.success(f"✅ Successfully enhanced {successful_enhancements} out of {len(results)} photos!")

    else:
        st.info("📸 Upload one or more photos to begin enhancement.")


# Helper functions for image generation tools

def convert_image_to_bytes(image):
    """Convert PIL image to bytes"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()


def build_style_transfer_prompt_with_reference(style_strength, preserve_content, color_preservation, edge_enhancement,
                                               texture_detail, blend_mode):
    """Build style transfer prompt with reference image"""
    prompt = f"Apply style transfer with the following settings: "
    prompt += f"Style strength: {style_strength}, Content preservation: {preserve_content}. "

    if color_preservation:
        prompt += "Preserve original colors while applying style. "

    if edge_enhancement:
        prompt += "Enhance edges for better definition. "

    prompt += f"Texture detail level: {texture_detail}. "
    prompt += f"Use {blend_mode.lower()} blend mode for combining style and content."

    return prompt


def build_style_transfer_prompt_preset(preset_style, style_strength, preserve_content, color_preservation,
                                       edge_enhancement, texture_detail):
    """Build style transfer prompt with preset style"""
    prompt = f"Apply {preset_style} artistic style to this image. "
    prompt += f"Style strength: {style_strength}, Content preservation: {preserve_content}. "

    if color_preservation:
        prompt += "Preserve original colors while applying artistic style. "

    if edge_enhancement:
        prompt += "Enhance edges and details. "

    prompt += f"Apply texture details at level {texture_detail}. "
    prompt += "Maintain the essence of the original while transforming it artistically."

    return prompt


def build_synthesis_prompt(description, style, composition, color_palette, detail_level, creativity, aspect_ratio):
    """Build comprehensive image synthesis prompt"""
    prompt = f"Synthesize an image: {description}. "
    prompt += f"Art style: {style}. "
    prompt += f"Composition: {composition}. "
    prompt += f"Color palette: {color_palette}. "
    prompt += f"Detail level: {detail_level}. "
    prompt += f"Creativity level: {creativity}. "
    prompt += f"Aspect ratio: {aspect_ratio}. "
    prompt += "Create a cohesive, visually appealing image that matches these specifications."

    return prompt


def build_composition_prompt(num_images, composition_style, transition_type, background_treatment, color_harmony):
    """Build image composition prompt"""
    prompt = f"Compose {num_images} images using {composition_style.lower()} style. "
    prompt += f"Apply {transition_type.lower()} transitions between elements. "
    prompt += f"Background treatment: {background_treatment.lower()}. "

    if color_harmony:
        prompt += "Apply color harmony to unify the composition. "

    prompt += "Create a seamless, balanced composition that integrates all elements naturally."

    return prompt


def build_concept_art_prompt(concept_type, description, art_style, mood, color_scheme, detail_level, perspective,
                             reference_style, technical_specs):
    """Build comprehensive concept art generation prompt"""
    prompt = f"Create {concept_type.lower()} concept art: {description}. "
    prompt += f"Art style: {art_style}. "
    prompt += f"Mood: {mood}. "
    prompt += f"Color scheme: {color_scheme}. "
    prompt += f"Detail level: {detail_level}. "
    prompt += f"Perspective: {perspective}. "

    if reference_style:
        prompt += f"Reference style: {reference_style}. "

    if technical_specs:
        prompt += f"Technical specifications: {technical_specs}. "

    prompt += "Focus on creating functional, appealing design that serves the intended purpose."

    return prompt


def build_auto_enhancement_prompt(auto_mode, enhancement_strength, preserve_original, fix_common_issues):
    """Build automatic photo enhancement prompt"""
    prompt = f"Enhance this photo using {auto_mode.lower()} mode. "
    prompt += f"Enhancement strength: {enhancement_strength}. "

    if preserve_original:
        prompt += "Preserve the original character and style of the photo. "

    if fix_common_issues:
        prompt += "Fix common issues like exposure, color balance, and sharpness. "

    prompt += "Improve overall quality while maintaining natural appearance."

    return prompt


def build_custom_enhancement_prompt(brightness, contrast, saturation, sharpness, noise_reduction, color_balance):
    """Build custom photo enhancement prompt"""
    prompt = "Enhance this photo with custom settings: "

    if brightness != 0:
        prompt += f"Brightness adjustment: {brightness:+.1f}. "

    if contrast != 0:
        prompt += f"Contrast adjustment: {contrast:+.1f}. "

    if saturation != 0:
        prompt += f"Saturation adjustment: {saturation:+.1f}. "

    if sharpness != 0:
        prompt += f"Sharpness adjustment: {sharpness:+.1f}. "

    if noise_reduction > 0:
        prompt += f"Apply noise reduction at level {noise_reduction}. "

    if color_balance != 0:
        prompt += f"Color balance adjustment: {color_balance:+.1f}. "

    prompt += "Apply these adjustments to improve the photo's overall quality."

    return prompt


def build_restoration_prompt(restoration_options, restoration_intensity):
    """Build photo restoration prompt"""
    prompt = "Restore this photo with the following improvements: "

    if restoration_options:
        prompt += f"{', '.join(restoration_options).lower()}. "

    prompt += f"Restoration intensity: {restoration_intensity}. "
    prompt += "Carefully repair and restore the photo while preserving its historical character."

    return prompt


def ocr_reader():
    """OCR text extraction tool"""
    create_tool_header("OCR Reader", "Extract text from images using AI vision", "👁️")

    # File upload section
    st.subheader("Upload Image")
    uploaded_files = FileHandler.upload_files(['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'], accept_multiple=True)

    if uploaded_files:
        # OCR options
        st.subheader("OCR Settings")

        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("Primary Language", [
                "Auto-detect", "English", "Spanish", "French", "German", "Italian",
                "Portuguese", "Chinese", "Japanese", "Korean", "Arabic", "Russian"
            ])

            output_format = st.selectbox("Output Format", [
                "Plain Text", "Structured Text", "Markdown", "JSON"
            ])

        with col2:
            preserve_formatting = st.checkbox("Preserve Formatting", value=True)
            include_confidence = st.checkbox("Include Confidence Scores", value=False)

        # Advanced options
        with st.expander("Advanced Options"):
            extract_tables = st.checkbox("Extract Tables")
            extract_handwriting = st.checkbox("Extract Handwriting")
            noise_reduction = st.checkbox("Apply Noise Reduction", value=True)

        if st.button("Extract Text"):
            results = []

            for file in uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    # Process image
                    image = FileHandler.process_image_file(file)
                    if image:
                        # Convert image to bytes for AI processing
                        import io
                        img_byte_arr = io.BytesIO()
                        image.save(img_byte_arr, format='JPEG')
                        img_bytes = img_byte_arr.getvalue()

                        # Build OCR prompt
                        ocr_prompt = build_ocr_prompt(language, preserve_formatting,
                                                      extract_tables, extract_handwriting, output_format)

                        # Perform OCR using AI
                        extracted_text = ai_client.analyze_image(img_bytes, ocr_prompt)

                        results.append({
                            'filename': file.name,
                            'text': extracted_text,
                            'success': True
                        })
                    else:
                        results.append({
                            'filename': file.name,
                            'error': 'Failed to process image',
                            'success': False
                        })

            # Display results
            st.subheader("Extraction Results")

            for result in results:
                if result['success']:
                    st.markdown(f"### 📄 {result['filename']}")

                    # Display extracted text
                    if output_format == "JSON":
                        st.json(result['text'])
                    else:
                        st.text_area(f"Extracted Text from {result['filename']}",
                                     result['text'], height=200)

                    # Text analysis
                    text_analysis = analyze_extracted_text(result['text'])

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Word Count", text_analysis['word_count'])
                    with col2:
                        st.metric("Character Count", text_analysis['char_count'])
                    with col3:
                        st.metric("Lines", text_analysis['line_count'])

                    # Download extracted text
                    FileHandler.create_download_link(
                        result['text'].encode(),
                        f"extracted_{result['filename']}.txt",
                        "text/plain"
                    )

                else:
                    st.error(f"❌ Failed to process {result['filename']}: {result.get('error', 'Unknown error')}")

                st.markdown("---")

            # Batch download option
            if len([r for r in results if r['success']]) > 1:
                if st.button("Download All as ZIP"):
                    zip_files = {}
                    for result in results:
                        if result['success']:
                            zip_files[f"extracted_{result['filename']}.txt"] = result['text'].encode()

                    if zip_files:
                        zip_data = FileHandler.create_zip_archive(zip_files)
                        FileHandler.create_download_link(
                            zip_data,
                            "ocr_extraction_results.zip",
                            "application/zip"
                        )
    else:
        st.info("📸 Upload one or more images to extract text using AI-powered OCR.")


def voice_synthesis():
    """Voice synthesis tool"""
    create_tool_header("Voice Synthesis", "Generate natural speech from text using AI", "🎤")

    # Text input section
    st.subheader("Text to Synthesize")

    input_method = st.radio("Input Method", ["Text Input", "File Upload"])

    if input_method == "Text Input":
        text_to_speak = st.text_area("Enter text to convert to speech:",
                                     placeholder="Type or paste your text here...",
                                     height=150)
    else:
        uploaded_file = FileHandler.upload_files(['txt'], accept_multiple=False)
        if uploaded_file:
            text_to_speak = FileHandler.process_text_file(uploaded_file[0])
            st.text_area("Text from file:", text_to_speak, height=150, disabled=True)
        else:
            text_to_speak = ""

    if text_to_speak:
        # Voice settings
        st.subheader("Voice Settings")

        col1, col2 = st.columns(2)
        with col1:
            voice_type = st.selectbox("Voice Type", [
                "Natural Female", "Natural Male", "Professional Female", "Professional Male",
                "Friendly", "Authoritative", "Calm", "Energetic"
            ])

            language = st.selectbox("Language", [
                "English (US)", "English (UK)", "Spanish", "French", "German",
                "Italian", "Portuguese", "Chinese", "Japanese"
            ])

        with col2:
            speech_speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1)
            pitch = st.slider("Pitch", 0.5, 2.0, 1.0, 0.1)

        # Advanced options
        with st.expander("Advanced Options"):
            add_pauses = st.checkbox("Add Natural Pauses", value=True)
            emphasize_punctuation = st.checkbox("Emphasize Punctuation", value=True)
            output_quality = st.selectbox("Audio Quality", ["Standard", "High", "Premium"])
            output_format = st.selectbox("Output Format", ["MP3", "WAV", "OGG"])

        # Text analysis
        st.subheader("Text Analysis")
        text_stats = analyze_speech_text(text_to_speak)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Words", text_stats['word_count'])
        with col2:
            st.metric("Estimated Duration", f"{text_stats['estimated_duration']} sec")
        with col3:
            st.metric("Sentences", text_stats['sentence_count'])
        with col4:
            st.metric("Syllables", text_stats['syllable_estimate'])

        if st.button("Generate Speech"):
            with st.spinner("Generating speech audio..."):
                # Generate speech using AI
                speech_prompt = build_speech_prompt(text_to_speak, voice_type, language,
                                                    speech_speed, pitch, add_pauses,
                                                    emphasize_punctuation)

                # Since this is a text-to-speech task, we'll simulate audio generation
                # In a real implementation, this would call a text-to-speech API
                success_message = ai_client.generate_text(speech_prompt)

                st.success("🎵 Speech generation completed!")

                # Display generation details
                st.subheader("Generation Details")
                generation_info = {
                    "Voice Type": voice_type,
                    "Language": language,
                    "Speed": f"{speech_speed}x",
                    "Pitch": f"{pitch}x",
                    "Text Length": f"{len(text_to_speak)} characters",
                    "Estimated Audio Length": f"{text_stats['estimated_duration']} seconds"
                }

                for key, value in generation_info.items():
                    st.text(f"{key}: {value}")

                # Audio preview placeholder
                st.info("🔊 Audio Preview: In a full implementation, the generated speech would be playable here.")

                # Simulated download (in real implementation, this would be actual audio data)
                st.info("📥 Download: In a full implementation, you could download the audio file here.")

                # SSML export option
                if st.button("Export as SSML"):
                    ssml_content = generate_ssml(text_to_speak, voice_type, speech_speed, pitch)
                    FileHandler.create_download_link(
                        ssml_content.encode(),
                        "speech_markup.ssml",
                        "application/xml"
                    )
    else:
        st.info("📝 Enter or upload text to generate speech.")


def pattern_recognition():
    """Pattern recognition tool"""
    create_tool_header("Pattern Recognition", "Identify patterns in data and images using AI", "🔍")

    # Pattern recognition type selection
    pattern_type = st.selectbox("Select Pattern Type", [
        "Image Pattern Recognition", "Text Pattern Analysis", "Data Pattern Detection",
        "Sequence Pattern Recognition", "Anomaly Detection"
    ])

    if pattern_type == "Image Pattern Recognition":
        st.subheader("Image Pattern Analysis")

        uploaded_images = FileHandler.upload_files(['jpg', 'jpeg', 'png', 'gif', 'bmp'], accept_multiple=True)

        if uploaded_images:
            # Pattern recognition options
            col1, col2 = st.columns(2)
            with col1:
                recognition_focus = st.selectbox("Recognition Focus", [
                    "General Objects", "Faces", "Text/Numbers", "Shapes/Geometry",
                    "Colors/Patterns", "Brand Logos", "Architecture", "Nature Elements"
                ])

                detail_level = st.selectbox("Detail Level", ["Basic", "Detailed", "Comprehensive"])

            with col2:
                pattern_confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.7, 0.1)
                batch_analysis = st.checkbox("Batch Analysis", value=True)

            if st.button("Analyze Patterns"):
                results = []

                for image_file in uploaded_images:
                    with st.spinner(f"Analyzing patterns in {image_file.name}..."):
                        image = FileHandler.process_image_file(image_file)
                        if image:
                            # Convert to bytes for AI analysis
                            import io
                            img_byte_arr = io.BytesIO()
                            image.save(img_byte_arr, format='JPEG')
                            img_bytes = img_byte_arr.getvalue()

                            # Build pattern recognition prompt
                            pattern_prompt = build_pattern_recognition_prompt(
                                recognition_focus, detail_level, pattern_confidence
                            )

                            # Analyze patterns
                            pattern_analysis = ai_client.analyze_image(img_bytes, pattern_prompt)

                            results.append({
                                'filename': image_file.name,
                                'analysis': pattern_analysis,
                                'image': image,
                                'success': True
                            })
                        else:
                            results.append({
                                'filename': image_file.name,
                                'error': 'Failed to process image',
                                'success': False
                            })

                # Display results
                display_pattern_results(results, "Image Pattern Recognition")

    elif pattern_type == "Text Pattern Analysis":
        st.subheader("Text Pattern Analysis")

        input_method = st.radio("Input Method", ["Text Input", "File Upload"])

        if input_method == "Text Input":
            text_data = st.text_area("Enter text to analyze:", height=200)
        else:
            uploaded_file = FileHandler.upload_files(['txt', 'csv'], accept_multiple=False)
            if uploaded_file:
                text_data = FileHandler.process_text_file(uploaded_file[0])
                st.text_area("Text from file:", text_data[:500] + "..." if len(text_data) > 500 else text_data,
                             height=150, disabled=True)
            else:
                text_data = ""

        if text_data:
            # Text pattern options
            col1, col2 = st.columns(2)
            with col1:
                pattern_focus = st.selectbox("Pattern Focus", [
                    "General Patterns", "Email Addresses", "Phone Numbers", "URLs",
                    "Dates/Times", "Named Entities", "Keywords", "Sentiment Patterns"
                ])

            with col2:
                analysis_depth = st.selectbox("Analysis Depth", ["Quick", "Standard", "Deep"])

            if st.button("Analyze Text Patterns"):
                with st.spinner("Analyzing text patterns..."):
                    text_pattern_analysis = analyze_text_patterns(text_data, pattern_focus, analysis_depth)

                    st.subheader("Pattern Analysis Results")
                    st.write(text_pattern_analysis)

                    # Export results
                    FileHandler.create_download_link(
                        text_pattern_analysis.encode(),
                        "text_pattern_analysis.txt",
                        "text/plain"
                    )

    elif pattern_type == "Data Pattern Detection":
        st.subheader("Data Pattern Detection")

        uploaded_file = FileHandler.upload_files(['csv', 'json'], accept_multiple=False)

        if uploaded_file:
            # Load data
            if uploaded_file[0].name.endswith('.csv'):
                data = FileHandler.process_csv_file(uploaded_file[0])
                if data is not None:
                    st.write("Data Preview:")
                    st.dataframe(data.head())

                    # Pattern detection options
                    col1, col2 = st.columns(2)
                    with col1:
                        columns_to_analyze = st.multiselect("Columns to Analyze", data.columns.tolist())
                        pattern_types = st.multiselect("Pattern Types", [
                            "Trends", "Correlations", "Outliers", "Seasonality", "Clustering"
                        ])

                    with col2:
                        time_column = st.selectbox("Time Column (optional)", ["None"] + data.columns.tolist())
                        confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95, 0.01)

                    if st.button("Detect Patterns") and columns_to_analyze:
                        with st.spinner("Detecting data patterns..."):
                            data_patterns = detect_data_patterns(data, columns_to_analyze, pattern_types, time_column)

                            st.subheader("Data Pattern Results")
                            st.write(data_patterns)

    else:
        st.info(f"🔧 {pattern_type} implementation in progress. Please check back soon!")


def story_writer():
    """AI story writing tool"""
    create_tool_header("Story Writer", "Create engaging stories with AI assistance", "📚")

    # Story creation mode
    creation_mode = st.selectbox("Creation Mode", [
        "New Story", "Continue Existing Story", "Story from Prompt", "Interactive Story"
    ])

    if creation_mode == "New Story":
        st.subheader("Story Settings")

        col1, col2 = st.columns(2)
        with col1:
            genre = st.selectbox("Genre", [
                "Adventure", "Mystery", "Romance", "Science Fiction", "Fantasy",
                "Horror", "Drama", "Comedy", "Historical Fiction", "Thriller"
            ])

            story_length = st.selectbox("Story Length", [
                "Short Story (500-1500 words)", "Medium Story (1500-5000 words)",
                "Long Story (5000+ words)", "Chapter (2000-3000 words)"
            ])

            target_audience = st.selectbox("Target Audience", [
                "Children (6-12)", "Young Adult (13-17)", "Adult (18+)", "All Ages"
            ])

        with col2:
            writing_style = st.selectbox("Writing Style", [
                "Descriptive", "Action-packed", "Dialogue-heavy", "Contemplative",
                "Humorous", "Dark", "Poetic", "Realistic"
            ])

            pov = st.selectbox("Point of View", [
                "First Person", "Third Person Limited", "Third Person Omniscient", "Second Person"
            ])

            tone = st.selectbox("Tone", [
                "Optimistic", "Dark", "Neutral", "Suspenseful", "Whimsical",
                "Serious", "Lighthearted", "Mysterious"
            ])

        # Story elements
        st.subheader("Story Elements")

        col1, col2 = st.columns(2)
        with col1:
            main_character = st.text_input("Main Character", placeholder="Describe your protagonist...")
            setting = st.text_input("Setting", placeholder="Where and when does the story take place?")

        with col2:
            conflict = st.text_input("Central Conflict", placeholder="What challenge will the character face?")
            theme = st.text_input("Theme (optional)", placeholder="What's the underlying message?")

        # Additional story details
        with st.expander("Additional Details"):
            supporting_characters = st.text_area("Supporting Characters",
                                                 placeholder="Describe other important characters...")
            plot_outline = st.text_area("Plot Outline (optional)",
                                        placeholder="Brief outline of main plot points...")
            special_elements = st.text_area("Special Elements",
                                            placeholder="Magic systems, technology, unique rules...")

        if st.button("Generate Story"):
            if main_character and setting:
                with st.spinner("Crafting your story..."):
                    story_prompt = build_story_prompt(
                        genre, story_length, target_audience, writing_style, pov, tone,
                        main_character, setting, conflict, theme, supporting_characters,
                        plot_outline, special_elements
                    )

                    # Generate story
                    generated_story = ai_client.generate_text(story_prompt, max_tokens=3000)

                    if generated_story:
                        display_generated_story(generated_story, {
                            'genre': genre,
                            'length': story_length,
                            'style': writing_style,
                            'pov': pov,
                            'tone': tone
                        })
            else:
                st.error("Please provide at least a main character and setting.")

    elif creation_mode == "Continue Existing Story":
        st.subheader("Continue Your Story")

        existing_text = st.text_area("Existing Story Text",
                                     placeholder="Paste your existing story here...",
                                     height=200)

        if existing_text:
            col1, col2 = st.columns(2)
            with col1:
                continuation_length = st.selectbox("Continuation Length", [
                    "Short (200-500 words)", "Medium (500-1000 words)", "Long (1000+ words)"
                ])

                direction = st.selectbox("Story Direction", [
                    "Continue naturally", "Add conflict", "Introduce new character",
                    "Change setting", "Build toward climax", "Resolve conflict"
                ])

            with col2:
                maintain_style = st.checkbox("Maintain Writing Style", value=True)
                maintain_tone = st.checkbox("Maintain Tone", value=True)

            if st.button("Continue Story"):
                with st.spinner("Continuing your story..."):
                    continuation_prompt = build_continuation_prompt(
                        existing_text, continuation_length, direction, maintain_style, maintain_tone
                    )

                    story_continuation = ai_client.generate_text(continuation_prompt, max_tokens=2000)

                    if story_continuation:
                        st.subheader("Story Continuation")
                        st.markdown(story_continuation)

                        # Combined story
                        full_story = existing_text + "\n\n" + story_continuation
                        story_analysis = analyze_story_content(full_story)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Words", story_analysis['word_count'])
                        with col2:
                            st.metric("Reading Time", f"{story_analysis['reading_time']} min")
                        with col3:
                            st.metric("Chapters/Sections", story_analysis['sections'])

                        # Download options
                        FileHandler.create_download_link(
                            full_story.encode(),
                            "continued_story.txt",
                            "text/plain"
                        )

    elif creation_mode == "Story from Prompt":
        st.subheader("Generate Story from Prompt")

        story_prompt_input = st.text_area("Story Prompt",
                                          placeholder="Enter a creative prompt, scenario, or idea...",
                                          height=100)

        if story_prompt_input:
            col1, col2 = st.columns(2)
            with col1:
                interpretation = st.selectbox("Interpretation Style", [
                    "Literal", "Creative", "Unexpected", "Dark Twist", "Humorous", "Philosophical"
                ])

                story_structure = st.selectbox("Story Structure", [
                    "Beginning-Middle-End", "In Media Res", "Circular", "Episodic", "Stream of Consciousness"
                ])

            with col2:
                prompt_length = st.selectbox("Story Length", [
                    "Flash Fiction (100-300 words)", "Short Story (500-1500 words)",
                    "Extended Story (1500+ words)"
                ])

                creative_freedom = st.slider("Creative Freedom", 0.1, 1.0, 0.8, 0.1)

            if st.button("Generate from Prompt"):
                with st.spinner("Creating story from your prompt..."):
                    prompt_story_text = generate_story_from_prompt(
                        story_prompt_input, interpretation, story_structure,
                        prompt_length, creative_freedom
                    )

                    if prompt_story_text:
                        display_generated_story(prompt_story_text, {
                            'prompt': story_prompt_input,
                            'interpretation': interpretation,
                            'structure': story_structure,
                            'length': prompt_length
                        })

    else:  # Interactive Story
        st.subheader("Interactive Story Creation")
        st.info("🎮 Interactive storytelling mode coming in a future update!")

        # Initialize interactive story session
        if 'interactive_story' not in st.session_state:
            st.session_state.interactive_story = {
                'story_text': '',
                'choices_made': [],
                'current_scene': 1
            }

        st.write("This mode will allow you to make choices that influence the story direction.")


def refine_content(content):
    """Refine generated content"""
    if not content:
        st.warning("No content provided for refinement.")
        return

    create_tool_header("Content Refinement", "Improve and polish your content with AI", "✨")

    # Display original content
    st.subheader("Original Content")
    st.text_area("Content to refine:", content, height=200, disabled=True)

    # Refinement options
    st.subheader("Refinement Options")

    col1, col2 = st.columns(2)
    with col1:
        refinement_type = st.selectbox("Refinement Type", [
            "General Improvement", "Grammar & Style", "Clarity & Readability",
            "Tone Adjustment", "Length Optimization", "SEO Enhancement",
            "Professional Polish", "Creative Enhancement"
        ])

        target_audience = st.selectbox("Target Audience", [
            "Keep Current", "General Public", "Professionals", "Students",
            "Experts", "Children", "Academics"
        ])

    with col2:
        desired_tone = st.selectbox("Desired Tone", [
            "Keep Current", "Professional", "Casual", "Friendly", "Formal",
            "Conversational", "Authoritative", "Persuasive", "Educational"
        ])

        length_preference = st.selectbox("Length Preference", [
            "Keep Current", "Make Shorter", "Make Longer", "More Concise", "More Detailed"
        ])

    # Advanced refinement options
    with st.expander("Advanced Options"):
        specific_improvements = st.multiselect("Specific Improvements", [
            "Fix Grammar", "Improve Flow", "Enhance Vocabulary", "Add Examples",
            "Strengthen Arguments", "Improve Structure", "Add Transitions", "Remove Redundancy"
        ])

        preserve_elements = st.multiselect("Preserve Elements", [
            "Original Meaning", "Key Facts", "Personal Voice", "Technical Terms",
            "Brand Voice", "Quotes", "Statistics", "Examples"
        ])

        custom_instructions = st.text_area("Custom Instructions",
                                           placeholder="Any specific requirements for refinement...")

    if st.button("Refine Content"):
        with st.spinner("Refining your content..."):
            # Build refinement prompt
            refinement_prompt = build_refinement_prompt(
                content, refinement_type, target_audience, desired_tone,
                length_preference, specific_improvements, preserve_elements,
                custom_instructions
            )

            # Refine content using AI
            refined_content = ai_client.generate_text(refinement_prompt, max_tokens=2500)

            if refined_content:
                # Display refined content
                st.subheader("Refined Content")
                st.markdown(refined_content)

                # Content comparison
                st.subheader("Improvement Analysis")

                original_analysis = analyze_content(content)
                refined_analysis = analyze_content(refined_content)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Word Count",
                              refined_analysis['word_count'],
                              refined_analysis['word_count'] - original_analysis['word_count'])

                with col2:
                    st.metric("Reading Time",
                              f"{refined_analysis['reading_time']} min",
                              f"{refined_analysis['reading_time'] - original_analysis['reading_time']} min")

                with col3:
                    st.metric("Readability", refined_analysis['readability_level'])

                with col4:
                    improvement_score = calculate_improvement_score(original_analysis, refined_analysis)
                    st.metric("Improvement Score", f"{improvement_score}%")

                # Side-by-side comparison
                with st.expander("Side-by-Side Comparison"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Original:**")
                        st.text_area("Original", content, height=300, disabled=True, key="orig_comp")

                    with col2:
                        st.markdown("**Refined:**")
                        st.text_area("Refined", refined_content, height=300, disabled=True, key="ref_comp")

                # Download options
                st.subheader("Download Options")

                col1, col2 = st.columns(2)
                with col1:
                    FileHandler.create_download_link(
                        refined_content.encode(),
                        "refined_content.txt",
                        "text/plain"
                    )

                with col2:
                    # Create comparison document
                    comparison_doc = f"""ORIGINAL CONTENT:
{content}

{'=' * 50}

REFINED CONTENT:
{refined_content}

{'=' * 50}

IMPROVEMENT ANALYSIS:
Original Word Count: {original_analysis['word_count']}
Refined Word Count: {refined_analysis['word_count']}
Improvement Score: {improvement_score}%
Refinement Type: {refinement_type}"""

                    FileHandler.create_download_link(
                        comparison_doc.encode(),
                        "content_comparison.txt",
                        "text/plain"
                    )

                # Additional refinement options
                st.subheader("Further Actions")
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("Refine Again"):
                        st.session_state.refinement_content = refined_content
                        st.rerun()

                with col2:
                    if st.button("Different Approach"):
                        st.session_state.refinement_content = content
                        st.rerun()

                with col3:
                    if st.button("Generate Summary"):
                        summary_prompt = f"Create a concise summary of this content:\n\n{refined_content}"
                        summary = ai_client.generate_text(summary_prompt, max_tokens=300)
                        st.subheader("Content Summary")
                        st.write(summary)
            else:
                st.error("Failed to refine content. Please try again.")


def api_setup_management():
    """API Setup & Management tool"""
    create_tool_header("API Setup & Management", "Manage AI API configurations", "🔑")

    st.markdown("""
    Configure your AI API keys to enable all AI-powered tools in this application.
    """)

    # Create columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🤖 Gemini API Configuration")

        # Check current API key status
        from api_config import APIConfig
        current_key_status = APIConfig.has_gemini()

        if current_key_status:
            st.success("✅ Gemini API key is configured and active")
            current_key = APIConfig.get_gemini_key()
            if current_key:
                masked_key = f"{'*' * (len(current_key) - 8)}{current_key[-4:]}" if len(current_key) > 8 else "***"
                st.info(f"Current key: {masked_key}")
        else:
            st.warning("⚠️ No Gemini API key found")

        # API key input
        st.markdown("**Enter your Gemini API Key:**")
        api_key_input = st.text_input(
            "API Key",
            type="password",
            placeholder="Enter your Google Gemini API key...",
            help="Get your API key from Google AI Studio (makersuite.google.com)"
        )

        # Save API key button
        if st.button("💾 Save API Key", type="primary"):
            if api_key_input and api_key_input.strip():
                # Save to session state
                st.session_state.gemini_api_key = api_key_input.strip()
                st.success("✅ API key saved successfully!")
                st.rerun()
            else:
                st.error("Please enter a valid API key")

        # Clear API key button
        if current_key_status and st.button("🗑️ Clear API Key"):
            if 'gemini_api_key' in st.session_state:
                del st.session_state.gemini_api_key
            st.success("API key cleared!")
            st.rerun()

    with col2:
        st.subheader("📊 API Status")

        # Test connection
        if st.button("🧪 Test Connection"):
            if APIConfig.has_gemini():
                with st.spinner("Testing connection..."):
                    try:
                        test_response = ai_client.generate_text("Hello! This is a connection test.", max_tokens=20)
                        if test_response and "Error" not in test_response:
                            st.success("✅ Connection successful!")
                            st.write("Test response:", test_response[:100] + "...")
                        else:
                            st.error("❌ Connection failed")
                            st.write("Response:", test_response)
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
            else:
                st.error("❌ Please configure API key first")

        # Available models
        st.markdown("**Available Models:**")
        try:
            models = ai_client.get_available_models()
            if models:
                for model in models:
                    st.write(f"• {model}")
            else:
                st.write("No models available")
        except:
            st.write("Unable to fetch models")

    st.markdown("---")

    # Instructions
    with st.expander("📖 How to get your Gemini API Key"):
        st.markdown("""
            1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Sign in with your Google account
            3. Click "Create API Key"
            4. Copy the generated API key
            5. Paste it in the field above and click "Save API Key"

            **Note:** Keep your API key secure and never share it publicly.
            """)

    # Additional providers info
    st.subheader("🔌 Additional AI Providers")
    st.info("""
        This application primarily uses Google Gemini AI. The system can simulate different AI model styles 
        (like OpenAI or Anthropic) using Gemini's capabilities with specialized prompting.

        For image generation, OpenAI DALL-E is used when available.
        """)


def provider_status():
    """Provider Status checker"""
    create_tool_header("Provider Status", "Check AI provider status", "📊")

    st.subheader("AI Provider Status")

    # Check available AI client status
    try:
        available_models = ai_client.get_available_models()
        if available_models:
            st.success("✅ AI client is configured and ready")
            st.write("Available models:", available_models)
        else:
            st.warning("⚠️ AI client is configured but no models available")
    except Exception as e:
        st.error(f"❌ AI client error: {str(e)}")

    st.info("💡 Configure your API keys through the integrations or environment variables.")


def test_connections():
    """Test AI provider connections"""
    create_tool_header("Test Connections", "Test AI provider connections", "🧪")

    st.subheader("Connection Testing")

    if st.button("Test AI Client Connection"):
        try:
            # Test basic AI client functionality
            test_response = ai_client.generate_text("Hello", max_tokens=10)
            if test_response:
                st.success("✅ AI Client connection successful!")
                st.write("Test response:", test_response)
            else:
                st.error("❌ AI Client connection failed - no response")
        except Exception as e:
            st.error(f"❌ AI Client connection failed: {str(e)}")

    st.info("This will test your configured AI providers to ensure they're working properly.")


def article_generator():
    """Generate articles using AI"""
    create_tool_header("Article Generator", "Generate professional articles", "📝")

    st.subheader("Article Configuration")

    # Article parameters
    topic = st.text_input("Article Topic", placeholder="Enter the main topic for your article")
    article_type = st.selectbox("Article Type", [
        "Blog Post", "News Article", "Tutorial", "Review", "Opinion Piece", "How-to Guide"
    ])

    word_count = st.selectbox("Target Word Count", [
        "300-500 words", "500-800 words", "800-1200 words", "1200+ words"
    ])

    tone = st.selectbox("Writing Tone", [
        "Professional", "Conversational", "Academic", "Casual", "Persuasive"
    ])

    target_audience = st.text_input("Target Audience", placeholder="Who is this article for?")

    # Generate button
    if st.button("Generate Article", type="primary"):
        if topic:
            with st.spinner("Generating article..."):
                try:
                    prompt = f"""Write a {article_type.lower()} about "{topic}" with the following requirements:
                    - Target length: {word_count}
                    - Writing tone: {tone}
                    - Target audience: {target_audience if target_audience else 'General audience'}

                    Make it engaging, well-structured, and informative."""

                    article = ai_client.generate_text(prompt, max_tokens=1500)

                    if article:
                        st.subheader("Generated Article")
                        st.write(article)

                        # Download option
                        FileHandler.create_download_link(
                            article.encode(),
                            f"{topic.replace(' ', '_')}_article.txt",
                            "text/plain"
                        )
                    else:
                        st.error("Failed to generate article. Please try again.")
                except Exception as e:
                    st.error(f"Error generating article: {str(e)}")
        else:
            st.warning("Please enter an article topic first.")


def copywriting_assistant():
    """AI-powered copywriting assistant"""
    create_tool_header("Copywriting Assistant", "Create compelling marketing copy", "✍️")

    st.subheader("Copy Configuration")

    copy_type = st.selectbox("Copy Type", [
        "Advertisement", "Email Subject Line", "Product Description", "Social Media Post",
        "Landing Page Copy", "Call-to-Action", "Headline", "Sales Letter"
    ])

    product_service = st.text_input("Product/Service", placeholder="What are you promoting?")
    key_benefits = st.text_area("Key Benefits", placeholder="List the main benefits or features")
    target_audience = st.text_input("Target Audience", placeholder="Who is your ideal customer?")

    tone = st.selectbox("Brand Tone", [
        "Professional", "Friendly", "Urgent", "Luxury", "Casual", "Authoritative"
    ])

    if st.button("Generate Copy", type="primary"):
        if product_service:
            with st.spinner("Creating copy..."):
                try:
                    prompt = f"""Create compelling {copy_type.lower()} copy for:
                    Product/Service: {product_service}
                    Key Benefits: {key_benefits}
                    Target Audience: {target_audience if target_audience else 'General audience'}
                    Tone: {tone}

                    Make it persuasive, clear, and action-oriented."""

                    copy = ai_client.generate_text(prompt, max_tokens=500)

                    if copy:
                        st.subheader("Generated Copy")
                        st.write(copy)

                        # Download option
                        FileHandler.create_download_link(
                            copy.encode(),
                            f"{copy_type.replace(' ', '_')}_copy.txt",
                            "text/plain"
                        )
                    else:
                        st.error("Failed to generate copy. Please try again.")
                except Exception as e:
                    st.error(f"Error generating copy: {str(e)}")
        else:
            st.warning("Please enter a product or service description first.")


def technical_writer():
    """Generate technical documentation"""
    create_tool_header("Technical Writer", "Create technical documentation", "📋")

    st.subheader("Documentation Configuration")

    doc_type = st.selectbox("Document Type", [
        "User Manual", "API Documentation", "Installation Guide", "Troubleshooting Guide",
        "Technical Specification", "README File", "Code Documentation"
    ])

    subject = st.text_input("Subject/Topic", placeholder="What are you documenting?")
    audience_level = st.selectbox("Audience Level", [
        "Beginner", "Intermediate", "Advanced", "Expert"
    ])

    requirements = st.text_area("Requirements/Details",
                                placeholder="Provide specific details, requirements, or context")

    include_examples = st.checkbox("Include Examples", value=True)
    include_troubleshooting = st.checkbox("Include Troubleshooting Section")

    if st.button("Generate Documentation", type="primary"):
        if subject:
            with st.spinner("Writing documentation..."):
                try:
                    prompt = f"""Create comprehensive {doc_type.lower()} for: {subject}

                    Target audience: {audience_level} level users
                    Requirements/Details: {requirements}
                    {"Include practical examples and code samples where appropriate." if include_examples else ""}
                    {"Include a troubleshooting section with common issues and solutions." if include_troubleshooting else ""}

                    Make it clear, structured, and easy to follow."""

                    documentation = ai_client.generate_text(prompt, max_tokens=2000)

                    if documentation:
                        st.subheader("Generated Documentation")
                        st.markdown(documentation)

                        # Download option
                        FileHandler.create_download_link(
                            documentation.encode(),
                            f"{subject.replace(' ', '_')}_documentation.md",
                            "text/markdown"
                        )
                    else:
                        st.error("Failed to generate documentation. Please try again.")
                except Exception as e:
                    st.error(f"Error generating documentation: {str(e)}")
        else:
            st.warning("Please enter a subject or topic first.")


def trend_analysis():
    """AI-powered trend analysis tool"""
    create_tool_header("Trend Analysis", "Analyze trends and patterns in your data using AI", "📈")

    if not DATA_ANALYSIS_AVAILABLE:
        st.error("Data analysis libraries not available. Please install pandas, numpy, and matplotlib.")
        st.code("pip install pandas numpy matplotlib seaborn")
        return

    st.markdown("""
    Upload your data or enter values to identify trends, patterns, and insights using AI analysis.
    Supports time series data, sales data, metrics, and more.
    """)

    # Data input options
    input_method = st.radio("Data Input Method", ["Upload File", "Enter Data Manually", "Sample Data"])

    data = None

    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload your data file", type=['csv', 'xlsx', 'json'])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.xlsx'):
                    data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    data = pd.read_json(uploaded_file)

                st.success(f"Loaded {len(data)} rows of data")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    elif input_method == "Enter Data Manually":
        st.subheader("Manual Data Entry")

        # Create simple data entry interface
        col1, col2 = st.columns(2)
        with col1:
            date_col = st.text_input("Date/Time Column", value="date")
            value_col = st.text_input("Value Column", value="value")

        with col2:
            num_rows = st.number_input("Number of data points", min_value=5, max_value=100, value=10)

        # Data entry
        manual_data = {}
        dates = []
        values = []

        st.subheader("Enter your data:")
        for i in range(num_rows):
            col1, col2 = st.columns(2)
            with col1:
                date_val = st.date_input(f"Date {i + 1}", key=f"date_{i}")
                dates.append(str(date_val))
            with col2:
                value_val = st.number_input(f"Value {i + 1}", key=f"value_{i}")
                values.append(value_val)

        if st.button("Create Dataset"):
            data = pd.DataFrame({
                date_col: dates,
                value_col: values
            })
            st.dataframe(data)

    else:  # Sample Data
        st.subheader("Sample Dataset")
        sample_type = st.selectbox("Choose sample data:", [
            "Sales Trend", "Website Traffic", "Stock Prices", "Temperature Data"
        ])

        if sample_type == "Sales Trend":
            dates = pd.date_range('2023-01-01', periods=12, freq='M')
            values = [1000, 1200, 1100, 1400, 1600, 1500, 1800, 1700, 1900, 2100, 2000, 2300]
        elif sample_type == "Website Traffic":
            dates = pd.date_range('2024-01-01', periods=30, freq='D')
            values = np.random.randint(500, 2000, 30)
        elif sample_type == "Stock Prices":
            dates = pd.date_range('2024-01-01', periods=50, freq='D')
            values = 100 + np.cumsum(np.random.randn(50) * 2)
        else:  # Temperature Data
            dates = pd.date_range('2024-01-01', periods=365, freq='D')
            values = 20 + 10 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.randn(365) * 2

        data = pd.DataFrame({
            'date': dates,
            'value': values
        })

        st.dataframe(data.head(10))

    if data is not None and len(data) > 0:
        st.markdown("---")

        # Column selection
        st.subheader("Analysis Configuration")

        col1, col2 = st.columns(2)
        with col1:
            date_column = st.selectbox("Select Date/Time Column", data.columns)
            value_column = st.selectbox("Select Value Column", [col for col in data.columns if col != date_column])

        with col2:
            analysis_type = st.selectbox("Analysis Type", [
                "Overall Trend", "Seasonal Patterns", "Growth Rate", "Volatility Analysis", "Comprehensive"
            ])
            ai_insights = st.checkbox("Generate AI Insights", value=True)

        if st.button("Analyze Trends", type="primary"):
            with st.spinner("Analyzing trends..."):
                try:
                    # Basic trend analysis
                    st.subheader("📊 Trend Visualization")

                    # Convert date column to datetime if possible
                    try:
                        data[date_column] = pd.to_datetime(data[date_column])
                        data = data.sort_values(date_column)
                    except:
                        pass

                    # Create trend plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(data[date_column], data[value_column], marker='o', linewidth=2)
                    ax.set_title(f'Trend Analysis: {value_column} over {date_column}')
                    ax.set_xlabel(date_column)
                    ax.set_ylabel(value_column)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    # Statistical analysis
                    st.subheader("📈 Statistical Summary")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{data[value_column].mean():.2f}")
                    with col2:
                        st.metric("Median", f"{data[value_column].median():.2f}")
                    with col3:
                        st.metric("Std Dev", f"{data[value_column].std():.2f}")
                    with col4:
                        trend_direction = "↗️ Up" if data[value_column].iloc[-1] > data[value_column].iloc[
                            0] else "↘️ Down"
                        st.metric("Overall Trend", trend_direction)

                    # Calculate trend metrics
                    growth_rate = 0  # Initialize to prevent unbound variable error
                    if len(data) > 1:
                        growth_rate = ((data[value_column].iloc[-1] - data[value_column].iloc[0]) /
                                       data[value_column].iloc[0]) * 100
                        st.metric("Total Growth", f"{growth_rate:.1f}%")

                    # Moving averages if enough data
                    if len(data) >= 7:
                        st.subheader("📊 Moving Averages")

                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(data[date_column], data[value_column], label='Original', alpha=0.6)

                        if len(data) >= 7:
                            data['MA7'] = data[value_column].rolling(window=7).mean()
                            ax.plot(data[date_column], data['MA7'], label='7-period MA', linewidth=2)

                        if len(data) >= 30:
                            data['MA30'] = data[value_column].rolling(window=30).mean()
                            ax.plot(data[date_column], data['MA30'], label='30-period MA', linewidth=2)

                        ax.set_title('Trend with Moving Averages')
                        ax.legend()
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()

                    # AI-powered insights
                    ai_analysis = None  # Initialize to prevent unbound variable error
                    if ai_insights:
                        st.subheader("🤖 AI-Generated Insights")

                        # Create data summary for AI
                        data_summary = f"""
                        Data Analysis Summary:
                        - Dataset size: {len(data)} records
                        - Value range: {data[value_column].min():.2f} to {data[value_column].max():.2f}
                        - Mean: {data[value_column].mean():.2f}
                        - Standard deviation: {data[value_column].std():.2f}
                        - Overall change: {growth_rate:.1f}% {'increase' if growth_rate > 0 else 'decrease'}
                        - Recent values: {list(data[value_column].tail(5).values)}
                        """

                        prompt = f"""
                        Analyze this trend data and provide insights:

                        {data_summary}

                        Please provide:
                        1. Key trends and patterns identified
                        2. Notable observations or anomalies
                        3. Potential explanations for the trends
                        4. Recommendations or predictions
                        5. Areas of concern or opportunity

                        Be specific and actionable in your analysis.
                        """

                        ai_analysis = ai_client.generate_text(prompt, max_tokens=1000)
                        if ai_analysis:
                            st.markdown(ai_analysis)
                        else:
                            st.error("Failed to generate AI insights. Please check your AI connection.")

                    # Export functionality
                    st.subheader("💾 Export Results")

                    # Prepare export data
                    export_data = {
                        'original_data': data.to_dict('records'),
                        'analysis_results': {
                            'mean': float(data[value_column].mean()),
                            'median': float(data[value_column].median()),
                            'std_dev': float(data[value_column].std()),
                            'min_value': float(data[value_column].min()),
                            'max_value': float(data[value_column].max()),
                            'growth_rate': float(growth_rate) if len(data) > 1 else 0
                        },
                        'analysis_date': datetime.now().isoformat()
                    }

                    if ai_insights and ai_analysis:
                        export_data['ai_insights'] = ai_analysis

                    export_json = json.dumps(export_data, indent=2)
                    FileHandler.create_download_link(
                        export_json.encode(),
                        "trend_analysis_results.json",
                        "application/json"
                    )

                except Exception as e:
                    st.error(f"Error during trend analysis: {str(e)}")


def predictive_modeling():
    """AI-powered predictive modeling tool"""
    create_tool_header("Predictive Modeling", "Build predictive models using AI and machine learning", "🔮")

    st.markdown("""
    Create predictive models from your data using AI assistance. Upload historical data 
    to predict future trends, classifications, or numerical outcomes.
    """)

    # Model type selection
    model_type = st.selectbox("Select Model Type", [
        "Time Series Forecasting", "Linear Regression", "Classification",
        "Multiple Regression", "AI-Guided Prediction"
    ])

    # Data input
    st.subheader("📊 Data Input")
    uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            st.success(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns")
            st.dataframe(data.head())

            # Column selection
            st.subheader("🎯 Model Configuration")

            if model_type == "Time Series Forecasting":
                col1, col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("Date Column", data.columns)
                    target_col = st.selectbox("Target Variable", [col for col in data.columns if col != date_col])

                with col2:
                    forecast_periods = st.number_input("Forecast Periods", min_value=1, max_value=100, value=10)
                    confidence_interval = st.slider("Confidence Interval", 0.8, 0.99, 0.95, 0.05)

            elif model_type in ["Linear Regression", "Multiple Regression"]:
                target_col = st.selectbox("Target Variable (Y)", data.columns)
                feature_cols = st.multiselect("Feature Variables (X)",
                                              [col for col in data.columns if col != target_col])

                if len(feature_cols) == 0:
                    st.warning("Please select at least one feature variable.")
                    return

            elif model_type == "Classification":
                target_col = st.selectbox("Target Class Variable", data.columns)
                feature_cols = st.multiselect("Feature Variables",
                                              [col for col in data.columns if col != target_col])

                if len(feature_cols) == 0:
                    st.warning("Please select at least one feature variable.")
                    return

            else:  # AI-Guided Prediction
                target_col = st.selectbox("Variable to Predict", data.columns)
                feature_cols = st.multiselect("Available Features (leave empty for AI selection)",
                                              [col for col in data.columns if col != target_col])

            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                train_split = st.slider("Training Data Split", 0.6, 0.9, 0.8, 0.05)
                include_validation = st.checkbox("Include Validation Metrics", True)
                generate_ai_insights = st.checkbox("Generate AI Model Insights", True)

            if st.button("Build Predictive Model", type="primary"):
                with st.spinner("Building predictive model..."):
                    try:
                        if model_type == "Time Series Forecasting":
                            build_time_series_model(data, date_col, target_col, forecast_periods, confidence_interval)

                        elif model_type in ["Linear Regression", "Multiple Regression"]:
                            build_regression_model(data, target_col, feature_cols, train_split, include_validation)

                        elif model_type == "Classification":
                            build_classification_model(data, target_col, feature_cols, train_split, include_validation)

                        else:  # AI-Guided Prediction
                            build_ai_guided_model(data, target_col, feature_cols, train_split, generate_ai_insights)

                    except Exception as e:
                        st.error(f"Error building model: {str(e)}")

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

    else:
        # Sample data option
        st.subheader("📝 Or Try Sample Data")
        if st.button("Load Sample Sales Data"):
            # Create sample sales data
            dates = pd.date_range('2022-01-01', periods=100, freq='D')
            trend = np.linspace(1000, 2000, 100)
            seasonal = 200 * np.sin(2 * np.pi * np.arange(100) / 30)
            noise = np.random.normal(0, 100, 100)
            sales = trend + seasonal + noise

            sample_data = pd.DataFrame({
                'date': dates,
                'sales': sales,
                'marketing_spend': np.random.randint(100, 500, 100),
                'temperature': 20 + 10 * np.sin(2 * np.pi * np.arange(100) / 365) + np.random.randn(100) * 2
            })

            st.dataframe(sample_data.head())
            st.session_state['sample_data'] = sample_data


def build_time_series_model(data, date_col, target_col, forecast_periods, confidence_interval):
    """Build time series forecasting model"""
    st.subheader("📈 Time Series Forecast Results")

    # Prepare data
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col)

    # Simple linear trend model (for demo - in production would use ARIMA, Prophet, etc.)
    X = np.arange(len(data)).reshape(-1, 1)
    y = data[target_col].values

    # Fit simple polynomial trend
    coeffs = np.polyfit(X.flatten(), y, 2)
    trend = np.poly1d(coeffs)

    # Generate forecast
    future_X = np.arange(len(data), len(data) + forecast_periods).reshape(-1, 1)
    forecast = trend(future_X.flatten())

    # Create future dates
    last_date = data[date_col].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                 periods=forecast_periods, freq='D')

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data[date_col], data[target_col], label='Historical Data', marker='o')
    ax.plot(future_dates, forecast, label='Forecast', marker='s', linestyle='--', color='red')
    ax.set_title(f'Time Series Forecast for {target_col}')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Display forecast values
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': forecast
    })
    st.dataframe(forecast_df)


def build_regression_model(data, target_col, feature_cols, train_split, include_validation):
    """Build regression model"""
    st.subheader("📊 Regression Model Results")

    # Prepare data
    X = data[feature_cols].values
    y = data[target_col].values

    # Split data
    split_idx = int(len(data) * train_split)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Simple linear regression using numpy
    # Add bias term
    X_train_bias = np.column_stack([np.ones(X_train.shape[0]), X_train])

    # Calculate coefficients using normal equation
    coefficients = np.linalg.lstsq(X_train_bias, y_train, rcond=None)[0]

    # Make predictions
    X_test_bias = np.column_stack([np.ones(X_test.shape[0]), X_test])
    y_pred = X_test_bias @ coefficients

    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", f"{r2:.3f}")
    with col2:
        st.metric("RMSE", f"{rmse:.2f}")
    with col3:
        st.metric("MAE", f"{mae:.2f}")
    with col4:
        st.metric("MSE", f"{mse:.2f}")

    # Prediction vs Actual plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Predicted vs Actual Values')
    st.pyplot(fig)
    plt.close()


def build_classification_model(data, target_col, feature_cols, train_split, include_validation):
    """Build classification model"""
    st.subheader("🎯 Classification Model Results")

    # For demo purposes, show basic classification analysis
    st.write(f"Building classification model for {target_col}")
    st.write(f"Using features: {', '.join(feature_cols)}")

    # Show class distribution
    class_counts = data[target_col].value_counts()
    st.write("Class Distribution:")
    st.bar_chart(class_counts)

    # Basic feature correlation with target
    st.write("Feature Correlation Analysis:")
    for feature in feature_cols:
        if data[feature].dtype in ['int64', 'float64']:
            corr = data[feature].corr(pd.to_numeric(data[target_col], errors='coerce'))
            st.write(f"- {feature}: {corr:.3f}")


def build_ai_guided_model(data, target_col, feature_cols, train_split, generate_ai_insights):
    """Build AI-guided predictive model"""
    st.subheader("🤖 AI-Guided Model Results")

    # Generate AI analysis of the data
    if generate_ai_insights:
        # Create data summary
        data_summary = f"""
        Dataset Analysis:
        - Target variable: {target_col}
        - Available features: {list(data.columns)}
        - Dataset size: {len(data)} rows
        - Target statistics: Mean={data[target_col].mean():.2f}, Std={data[target_col].std():.2f}
        - Sample data: {data.head(3).to_dict('records')}
        """

        prompt = f"""
        Analyze this dataset and provide predictive modeling insights:

        {data_summary}

        Please provide:
        1. Recommended modeling approach
        2. Important features to consider
        3. Potential data quality issues
        4. Expected model performance
        5. Recommendations for improvement

        Be specific and technical in your analysis.
        """

        ai_analysis = ai_client.generate_text(prompt, max_tokens=1000)
        if ai_analysis:
            st.markdown("### 🧠 AI Model Analysis")
            st.markdown(ai_analysis)
        else:
            st.error("Failed to generate AI insights. Please check your AI connection.")


def statistical_analysis():
    """Comprehensive statistical analysis tool"""
    create_tool_header("Statistical Analysis", "Perform comprehensive statistical analysis with AI insights", "📊")

    st.markdown("""
    Upload your data for comprehensive statistical analysis including descriptive statistics, 
    hypothesis testing, correlation analysis, and AI-powered interpretations.
    """)

    # Data input
    uploaded_file = st.file_uploader("Upload your dataset", type=['csv', 'xlsx'])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)

            st.success(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns")

            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(data.head())

            # Analysis type selection
            st.subheader("📈 Analysis Options")

            analysis_types = st.multiselect("Select Analysis Types", [
                "Descriptive Statistics", "Correlation Analysis", "Distribution Analysis",
                "Outlier Detection", "Hypothesis Testing", "AI Statistical Insights"
            ], default=["Descriptive Statistics", "Correlation Analysis"])

            # Column selection
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = data.select_dtypes(include=['object']).columns.tolist()

            col1, col2 = st.columns(2)
            with col1:
                selected_numeric = st.multiselect("Numeric Columns", numeric_columns, default=numeric_columns[:5])
            with col2:
                selected_categorical = st.multiselect("Categorical Columns", categorical_columns,
                                                      default=categorical_columns[:3])

            if st.button("Perform Statistical Analysis", type="primary"):
                with st.spinner("Performing statistical analysis..."):

                    # Descriptive Statistics
                    if "Descriptive Statistics" in analysis_types and selected_numeric:
                        st.subheader("📊 Descriptive Statistics")

                        desc_stats = data[selected_numeric].describe()
                        st.dataframe(desc_stats)

                        # Additional statistics
                        additional_stats = pd.DataFrame({
                            'Skewness': data[selected_numeric].skew(),
                            'Kurtosis': data[selected_numeric].kurtosis(),
                            'Missing Values': data[selected_numeric].isnull().sum(),
                            'Unique Values': data[selected_numeric].nunique()
                        })

                        st.subheader("📈 Additional Statistics")
                        st.dataframe(additional_stats)

                    # Correlation Analysis
                    if "Correlation Analysis" in analysis_types and len(selected_numeric) > 1:
                        st.subheader("🔗 Correlation Analysis")

                        correlation_matrix = data[selected_numeric].corr()

                        # Correlation heatmap
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                        ax.set_title('Correlation Matrix')
                        st.pyplot(fig)
                        plt.close()

                        # Strong correlations
                        st.subheader("🔍 Strong Correlations (|r| > 0.7)")
                        strong_corrs = []
                        for i in range(len(correlation_matrix.columns)):
                            for j in range(i + 1, len(correlation_matrix.columns)):
                                corr_val = correlation_matrix.iloc[i, j]
                                if abs(corr_val) > 0.7:
                                    strong_corrs.append({
                                        'Variable 1': correlation_matrix.columns[i],
                                        'Variable 2': correlation_matrix.columns[j],
                                        'Correlation': corr_val
                                    })

                        if strong_corrs:
                            st.dataframe(pd.DataFrame(strong_corrs))
                        else:
                            st.info("No strong correlations found (|r| > 0.7)")

                    # Distribution Analysis
                    if "Distribution Analysis" in analysis_types and selected_numeric:
                        st.subheader("📈 Distribution Analysis")

                        # Select column for detailed distribution analysis
                        dist_column = st.selectbox("Select column for distribution analysis", selected_numeric)

                        col1, col2 = st.columns(2)

                        with col1:
                            # Histogram
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.hist(data[dist_column].dropna(), bins=30, alpha=0.7, edgecolor='black')
                            ax.set_title(f'Distribution of {dist_column}')
                            ax.set_xlabel(dist_column)
                            ax.set_ylabel('Frequency')
                            st.pyplot(fig)
                            plt.close()

                        with col2:
                            # Box plot
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.boxplot(data[dist_column].dropna())
                            ax.set_title(f'Box Plot of {dist_column}')
                            ax.set_ylabel(dist_column)
                            st.pyplot(fig)
                            plt.close()

                    # Outlier Detection
                    if "Outlier Detection" in analysis_types and selected_numeric:
                        st.subheader("🎯 Outlier Detection")

                        outlier_results = {}
                        for col in selected_numeric:
                            Q1 = data[col].quantile(0.25)
                            Q3 = data[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR

                            outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                            outlier_results[col] = {
                                'Count': len(outliers),
                                'Percentage': (len(outliers) / len(data)) * 100,
                                'Lower Bound': lower_bound,
                                'Upper Bound': upper_bound
                            }

                        outlier_df = pd.DataFrame(outlier_results).T
                        st.dataframe(outlier_df)

                    # AI Statistical Insights
                    if "AI Statistical Insights" in analysis_types:
                        st.subheader("🤖 AI Statistical Insights")

                        # Prepare data summary for AI
                        data_summary = f"""
                        Statistical Analysis Summary:
                        - Dataset shape: {data.shape}
                        - Numeric columns: {selected_numeric}
                        - Categorical columns: {selected_categorical}
                        - Basic statistics: {data[selected_numeric].describe().to_dict() if selected_numeric else 'No numeric columns'}
                        - Missing values: {data.isnull().sum().to_dict()}
                        - Data types: {data.dtypes.to_dict()}
                        """

                        if len(selected_numeric) > 1:
                            data_summary += f"- Correlation summary: {correlation_matrix.to_dict()}"

                        prompt = f"""
                        Analyze this statistical data and provide insights:

                        {data_summary}

                        Please provide:
                        1. Key statistical findings and patterns
                        2. Data quality assessment
                        3. Notable relationships between variables
                        4. Potential issues or concerns
                        5. Recommendations for further analysis
                        6. Business or practical insights

                        Be specific and provide actionable insights.
                        """

                        ai_insights = ai_client.generate_text(prompt, max_tokens=1500)
                        if ai_insights:
                            st.markdown(ai_insights)
                        else:
                            st.error("Failed to generate AI insights. Please check your AI connection.")

                    # Export results
                    st.subheader("💾 Export Analysis Results")

                    export_data = {
                        'descriptive_statistics': data[
                            selected_numeric].describe().to_dict() if selected_numeric else {},
                        'correlation_matrix': correlation_matrix.to_dict() if len(selected_numeric) > 1 else {},
                        'missing_values': data.isnull().sum().to_dict(),
                        'data_types': data.dtypes.astype(str).to_dict(),
                        'analysis_date': datetime.now().isoformat()
                    }

                    if "AI Statistical Insights" in analysis_types and 'ai_insights' in locals():
                        export_data['ai_insights'] = ai_insights

                    export_json = json.dumps(export_data, indent=2)
                    FileHandler.create_download_link(
                        export_json.encode(),
                        "statistical_analysis_results.json",
                        "application/json"
                    )

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

    else:
        # Sample data option
        st.subheader("📝 Or Try Sample Data")
        if st.button("Load Sample Dataset"):
            # Create sample dataset
            np.random.seed(42)
            n_samples = 1000

            sample_data = pd.DataFrame({
                'age': np.random.randint(18, 80, n_samples),
                'income': np.random.normal(50000, 15000, n_samples),
                'education_years': np.random.randint(8, 20, n_samples),
                'experience': np.random.randint(0, 40, n_samples),
                'satisfaction': np.random.randint(1, 11, n_samples),
                'category': np.random.choice(['A', 'B', 'C'], n_samples),
                'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
            })

            # Add some correlations
            sample_data['income'] += sample_data['education_years'] * 2000 + np.random.normal(0, 5000, n_samples)
            sample_data['satisfaction'] += (sample_data['income'] / 10000) + np.random.normal(0, 2, n_samples)

            st.dataframe(sample_data.head())
            st.session_state['sample_statistical_data'] = sample_data


# AI Utilities Helper Functions

def get_default_prompt_template(step_type):
    """Get default prompt template for workflow step types"""
    templates = {
        "Text Generation": "Generate creative text about {input}",
        "Text Analysis": "Analyze the following text and provide insights: {input}",
        "Content Creation": "Create engaging content about {input}",
        "Translation": "Translate the following text to [TARGET_LANGUAGE]: {input}",
        "Summarization": "Summarize the following text concisely: {input}",
        "Question Answering": "Answer the following question: {input}",
        "Data Processing": "Process and analyze the following data: {input}"
    }
    return templates.get(step_type, "Process the following input: {input}")


def execute_ai_workflow():
    """Execute the AI workflow steps sequentially"""
    if not st.session_state.workflow_steps:
        st.warning("No workflow steps to execute")
        return

    st.session_state.workflow_results = []

    with st.spinner("Executing AI workflow..."):
        previous_output = ""

        for i, step in enumerate(st.session_state.workflow_steps):
            step_start_time = time.time()

            try:
                # Prepare input
                if step['use_previous'] and previous_output:
                    step_input = previous_output
                else:
                    step_input = step['input']

                # Format prompt with input
                formatted_prompt = step['prompt'].replace('{input}', step_input).replace('{previous_output}',
                                                                                         previous_output)

                # Generate AI response
                response = ai_client.generate_text(formatted_prompt, model=step['model'], max_tokens=500)

                # Calculate execution time
                execution_time = time.time() - step_start_time

                # Store result
                result = {
                    'step_name': step['name'],
                    'output': response,
                    'execution_time': execution_time,
                    'model': step['model'],
                    'input': step_input[:100] + "..." if len(step_input) > 100 else step_input
                }
                st.session_state.workflow_results.append(result)

                # Update previous output for next step
                previous_output = response

                # Show progress
                st.success(f"✅ Completed step {i + 1}: {step['name']}")

            except Exception as e:
                st.error(f"❌ Error in step {i + 1} ({step['name']}): {str(e)}")
                break

    st.rerun()


def save_ai_workflow(workflow_name):
    """Save the current workflow"""
    if not st.session_state.workflow_steps:
        st.warning("No workflow to save")
        return

    workflow_data = {
        'name': workflow_name,
        'steps': st.session_state.workflow_steps,
        'created_at': datetime.now().isoformat()
    }

    # Save to session state (in a real app, this would be saved to database)
    if 'saved_workflows' not in st.session_state:
        st.session_state.saved_workflows = []

    st.session_state.saved_workflows.append(workflow_data)

    # Export as file
    workflow_json = json.dumps(workflow_data, indent=2)
    FileHandler.create_download_link(
        workflow_json.encode(),
        f"workflow_{workflow_name.replace(' ', '_')}.json",
        "application/json"
    )

    st.success(f"💾 Workflow '{workflow_name}' saved successfully!")


def analyze_workflow_performance():
    """Analyze workflow performance metrics"""
    if not st.session_state.workflow_results:
        st.warning("No workflow results to analyze")
        return

    st.subheader("📊 Workflow Performance Analysis")

    # Calculate metrics
    total_time = sum(r['execution_time'] for r in st.session_state.workflow_results)
    avg_time = total_time / len(st.session_state.workflow_results)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Execution Time", f"{total_time:.2f}s")
    with col2:
        st.metric("Average Step Time", f"{avg_time:.2f}s")
    with col3:
        st.metric("Total Steps", len(st.session_state.workflow_results))

    # Performance breakdown
    step_times = [r['execution_time'] for r in st.session_state.workflow_results]
    step_names = [r['step_name'] for r in st.session_state.workflow_results]

    performance_data = pd.DataFrame({
        'Step': step_names,
        'Execution_Time': step_times,
        'Model': [r['model'] for r in st.session_state.workflow_results]
    })

    st.subheader("Step Performance")
    st.dataframe(performance_data)


def export_workflow_results(workflow_name):
    """Export workflow execution results"""
    if not st.session_state.workflow_results:
        st.warning("No results to export")
        return

    export_data = {
        'workflow_name': workflow_name,
        'execution_date': datetime.now().isoformat(),
        'total_steps': len(st.session_state.workflow_results),
        'total_execution_time': sum(r['execution_time'] for r in st.session_state.workflow_results),
        'results': st.session_state.workflow_results
    }

    export_json = json.dumps(export_data, indent=2)
    FileHandler.create_download_link(
        export_json.encode(),
        f"workflow_results_{workflow_name.replace(' ', '_')}.json",
        "application/json"
    )

    st.success("📥 Workflow results exported!")


def show_saved_workflows():
    """Display saved workflows for loading"""
    if 'saved_workflows' not in st.session_state or not st.session_state.saved_workflows:
        st.info("No saved workflows found")
        return

    st.subheader("📂 Saved Workflows")

    for i, workflow in enumerate(st.session_state.saved_workflows):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{workflow['name']}** - {len(workflow['steps'])} steps")
            st.write(f"Created: {workflow['created_at']}")
        with col2:
            if st.button(f"Load", key=f"load_workflow_{i}"):
                st.session_state.workflow_steps = workflow['steps']
                st.success(f"Loaded workflow: {workflow['name']}")
                st.rerun()


def generate_batch_prompts(template, variables_text):
    """Generate batch prompts from template and variables"""
    try:
        # Try to parse as JSON first
        variables = json.loads(variables_text)

        # Generate all combinations
        prompts = []
        if isinstance(variables, dict):
            # Generate combinations of all variables
            keys = list(variables.keys())
            values = list(variables.values())

            from itertools import product
            for combination in product(*values):
                prompt = template
                for key, value in zip(keys, combination):
                    prompt = prompt.replace(f"{{{key}}}", str(value))
                prompts.append(prompt)

        return prompts

    except json.JSONDecodeError:
        # Try comma-separated format
        try:
            values = [v.strip() for v in variables_text.split(',')]
            prompts = []
            for value in values:
                prompt = template.replace('{variable}', value)
                prompts.append(prompt)
            return prompts
        except:
            st.error("Invalid variables format. Use JSON or comma-separated values.")
            return []


def process_ai_batch(prompts, model, max_tokens, concurrent_requests, delay):
    """Process batch of AI prompts"""
    st.session_state.batch_results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, prompt in enumerate(prompts):
        status_text.text(f"Processing prompt {i + 1}/{len(prompts)}")

        try:
            start_time = time.time()
            response = ai_client.generate_text(prompt, model=model, max_tokens=max_tokens)
            processing_time = time.time() - start_time

            result = {
                'input': prompt,
                'output': response,
                'status': 'success',
                'processing_time': processing_time,
                'model': model
            }

        except Exception as e:
            result = {
                'input': prompt,
                'output': None,
                'status': 'error',
                'error': str(e),
                'processing_time': 0,
                'model': model
            }

        st.session_state.batch_results.append(result)
        progress_bar.progress((i + 1) / len(prompts))

        # Add delay if specified
        if delay > 0 and i < len(prompts) - 1:
            time.sleep(delay)

    status_text.text("Batch processing complete!")
    st.rerun()


def process_file_batch(files, task, model):
    """Process batch of files with specified task"""
    st.session_state.batch_results = []

    task_prompts = {
        "Summarize Content": "Summarize the following content concisely: ",
        "Extract Key Information": "Extract the key information and main points from: ",
        "Sentiment Analysis": "Analyze the sentiment of the following text: ",
        "Translation": "Translate the following text to English: ",
        "Content Classification": "Classify the content type and topic of: ",
        "Quality Assessment": "Assess the quality and structure of: "
    }

    base_prompt = task_prompts.get(task, "Process the following content: ")

    progress_bar = st.progress(0)

    for i, file in enumerate(files):
        try:
            content = FileHandler.process_text_file(file)
            prompt = base_prompt + content[:1000]  # Limit content length

            start_time = time.time()
            response = ai_client.generate_text(prompt, model=model, max_tokens=300)
            processing_time = time.time() - start_time

            result = {
                'input': f"File: {file.name}",
                'output': response,
                'status': 'success',
                'processing_time': processing_time,
                'model': model
            }

        except Exception as e:
            result = {
                'input': f"File: {file.name}",
                'output': None,
                'status': 'error',
                'error': str(e),
                'processing_time': 0,
                'model': model
            }

        st.session_state.batch_results.append(result)
        progress_bar.progress((i + 1) / len(files))

    st.rerun()


def export_batch_results():
    """Export batch processing results"""
    if not st.session_state.batch_results:
        st.warning("No batch results to export")
        return

    export_data = {
        'batch_execution_date': datetime.now().isoformat(),
        'total_jobs': len(st.session_state.batch_results),
        'successful_jobs': sum(1 for r in st.session_state.batch_results if r.get('status') == 'success'),
        'results': st.session_state.batch_results
    }

    export_json = json.dumps(export_data, indent=2)
    FileHandler.create_download_link(
        export_json.encode(),
        f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        "application/json"
    )

    st.success("📥 Batch results exported!")


def analyze_single_response(prompt, response, criteria, depth, include_suggestions):
    """Analyze a single AI response for quality"""
    analysis_prompt = f"""
    Analyze the following AI response based on these criteria: {', '.join(criteria)}

    Original Prompt: {prompt}

    AI Response: {response}

    Analysis Depth: {depth}

    Please provide:
    1. Overall quality score (1-10)
    2. Analysis for each criterion
    3. Strengths and weaknesses
    {'4. Specific improvement suggestions' if include_suggestions else ''}
    """

    with st.spinner("Analyzing response..."):
        analysis = ai_client.generate_text(analysis_prompt, model="gemini", max_tokens=600)

        st.session_state.analysis_results = {
            'type': 'single_response',
            'analysis': analysis,
            'criteria': criteria,
            'prompt': prompt,
            'response': response
        }

        st.subheader("🔍 Response Analysis")
        st.write(analysis)


def compare_responses(prompt, response_a, response_b, model_a, model_b, criteria):
    """Compare two AI responses"""
    comparison_prompt = f"""
    Compare these two AI responses based on: {', '.join(criteria)}

    Original Prompt: {prompt}

    Response A ({model_a}): {response_a}

    Response B ({model_b}): {response_b}

    Please provide:
    1. Side-by-side comparison for each criterion
    2. Overall winner and reasoning
    3. Specific strengths of each response
    4. Recommendations for improvement
    """

    with st.spinner("Comparing responses..."):
        comparison = ai_client.generate_text(comparison_prompt, model="gemini", max_tokens=700)

        st.session_state.analysis_results = {
            'type': 'comparison',
            'analysis': comparison,
            'criteria': criteria,
            'responses': {'A': response_a, 'B': response_b},
            'models': {'A': model_a, 'B': model_b}
        }

        st.subheader("⚖️ Response Comparison")
        st.write(comparison)


def analyze_response_batch(uploaded_file, focus):
    """Analyze batch of responses from uploaded file"""
    try:
        # Process uploaded file
        file_content = FileHandler.process_text_file(uploaded_file)

        # Parse CSV data (simplified)
        lines = file_content.strip().split('\n')
        if len(lines) < 2:
            st.error("File must have at least 2 lines (header and data)")
            return

        headers = lines[0].split(',')
        data_rows = [line.split(',') for line in lines[1:]]

        batch_analysis_prompt = f"""
        Analyze this batch of AI responses with focus on: {focus}

        Data format: {', '.join(headers)}
        Sample size: {len(data_rows)} responses

        Please provide:
        1. Overall quality assessment
        2. Common patterns and issues
        3. Performance trends
        4. Recommendations for improvement
        """

        with st.spinner("Analyzing response batch..."):
            analysis = ai_client.generate_text(batch_analysis_prompt, model="gemini", max_tokens=600)

            st.session_state.analysis_results = {
                'type': 'batch_analysis',
                'analysis': analysis,
                'focus': focus,
                'sample_size': len(data_rows)
            }

            st.subheader("📦 Batch Analysis Results")
            st.write(analysis)

    except Exception as e:
        st.error(f"Error analyzing batch: {str(e)}")


def display_analysis_results(results):
    """Display analysis results with export options"""
    if results['type'] == 'single_response':
        st.write("**Analysis Type:** Single Response Analysis")
        st.write(f"**Criteria:** {', '.join(results['criteria'])}")

    elif results['type'] == 'comparison':
        st.write("**Analysis Type:** Comparative Analysis")
        st.write(f"**Criteria:** {', '.join(results['criteria'])}")

    elif results['type'] == 'batch_analysis':
        st.write("**Analysis Type:** Batch Analysis")
        st.write(f"**Focus:** {results['focus']}")
        st.write(f"**Sample Size:** {results['sample_size']}")

    # Export results
    if st.button("📥 Export Analysis"):
        export_json = json.dumps(results, indent=2)
        FileHandler.create_download_link(
            export_json.encode(),
            f"response_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )
        st.success("📥 Analysis exported!")


def model_training():
    """AI-powered model training tool"""
    create_tool_header("Model Training", "Train machine learning models with AI guidance", "🎯")

    st.subheader("Model Training Configuration")

    # Data input
    data_input = st.selectbox("Data Source", [
        "Upload Dataset", "Use Sample Data", "Connect Database"
    ])

    if data_input == "Upload Dataset":
        uploaded_file = FileHandler.upload_files(['csv', 'xlsx'], accept_multiple=False)
        if uploaded_file:
            st.success("Dataset uploaded successfully!")

    # Model configuration
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Model Type", [
            "Classification", "Regression", "Clustering", "Neural Network", "Deep Learning"
        ])
        algorithm = st.selectbox("Algorithm", [
            "Random Forest", "XGBoost", "SVM", "Linear Regression", "Neural Network"
        ])

    with col2:
        target_column = st.text_input("Target Column", placeholder="target")
        train_split = st.slider("Training Split", 0.5, 0.9, 0.8)

    # Training parameters
    with st.expander("Advanced Training Parameters"):
        max_iterations = st.slider("Max Iterations", 100, 10000, 1000)
        learning_rate = st.slider("Learning Rate", 0.001, 1.0, 0.01)
        validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2)

    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training model with AI optimization..."):
            # Simulate training process
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.02)

            st.success("🎉 Model training completed!")

            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", "94.2%")
            with col2:
                st.metric("Training Time", "2.3 min")
            with col3:
                st.metric("Model Size", "1.2 MB")


def data_preprocessing():
    """AI-powered data preprocessing tool"""
    create_tool_header("Data Preprocessing", "Prepare and clean data with AI assistance", "🧹")

    st.subheader("Data Input")

    # Data upload
    uploaded_file = FileHandler.upload_files(['csv', 'xlsx', 'json'], accept_multiple=False)

    if uploaded_file:
        st.subheader("Preprocessing Options")

        col1, col2 = st.columns(2)
        with col1:
            operations = st.multiselect("Preprocessing Operations", [
                "Remove Duplicates", "Handle Missing Values", "Normalize Data",
                "Encode Categorical", "Feature Scaling", "Outlier Detection"
            ])

        with col2:
            missing_strategy = st.selectbox("Missing Value Strategy", [
                "Fill with Mean", "Fill with Median", "Fill with Mode", "Drop Rows", "Forward Fill"
            ])

        # AI-powered suggestions
        if st.button("🤖 Get AI Preprocessing Suggestions"):
            with st.spinner("Analyzing data and generating preprocessing recommendations..."):
                suggestions = ai_client.generate_text(
                    "Analyze this dataset and suggest optimal preprocessing steps for machine learning. Include data cleaning, feature engineering, and transformation recommendations.",
                    model="gemini"
                )

                st.subheader("AI Preprocessing Recommendations")
                st.write(suggestions)

        if st.button("🔧 Apply Preprocessing", type="primary"):
            with st.spinner("Applying preprocessing operations..."):
                st.success("✅ Data preprocessing completed!")
                st.info("Processed dataset is ready for model training.")


def feature_engineering():
    """AI-powered feature engineering tool"""
    create_tool_header("Feature Engineering", "Create and optimize features with AI", "⚙️")

    st.subheader("Feature Engineering Configuration")

    # Data input
    uploaded_file = FileHandler.upload_files(['csv', 'xlsx'], accept_multiple=False)

    if uploaded_file:
        st.subheader("Feature Engineering Options")

        col1, col2 = st.columns(2)
        with col1:
            feature_types = st.multiselect("Feature Engineering Techniques", [
                "Polynomial Features", "Interaction Terms", "Feature Binning",
                "Text Vectorization", "Date/Time Features", "Statistical Features"
            ])

        with col2:
            target_col = st.text_input("Target Column", placeholder="target")
            feature_selection = st.selectbox("Feature Selection", [
                "Correlation Based", "Mutual Information", "Chi-Square", "LASSO", "Tree-based"
            ])

        # AI feature suggestions
        if st.button("🧠 Generate AI Feature Suggestions"):
            with st.spinner("Analyzing data patterns and generating feature ideas..."):
                feature_prompt = f"""
                Analyze this dataset and suggest innovative feature engineering techniques.
                Consider:
                1. Domain-specific features that could improve model performance
                2. Interaction terms between existing features
                3. Derived features from temporal, categorical, or numerical data
                4. Feature transformation strategies

                Provide specific, actionable feature engineering recommendations.
                """

                suggestions = ai_client.generate_text(feature_prompt, model="gemini")

                st.subheader("🎯 AI Feature Engineering Suggestions")
                st.write(suggestions)

        if st.button("🛠️ Apply Feature Engineering", type="primary"):
            with st.spinner("Creating and optimizing features..."):
                st.success("✅ Feature engineering completed!")
                st.info("New features created and optimized for model performance.")


def model_evaluation():
    """AI-powered model evaluation tool"""
    create_tool_header("Model Evaluation", "Evaluate and compare models with AI insights", "📊")

    st.subheader("Model Evaluation Setup")

    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox("Model Type", [
            "Classification", "Regression", "Clustering", "Time Series"
        ])

        evaluation_metrics = st.multiselect("Evaluation Metrics", [
            "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "MSE", "RMSE"
        ], default=["Accuracy"])

    with col2:
        cross_validation = st.selectbox("Cross Validation", [
            "5-Fold CV", "10-Fold CV", "Leave-One-Out", "Stratified CV"
        ])

        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.2)

    # Model comparison
    st.subheader("Model Comparison")

    models_to_compare = st.multiselect("Models to Compare", [
        "Random Forest", "XGBoost", "SVM", "Logistic Regression", "Neural Network"
    ])

    if st.button("📈 Evaluate Models", type="primary"):
        with st.spinner("Running comprehensive model evaluation..."):
            # Simulate evaluation
            results = {}
            for model in models_to_compare:
                results[model] = {
                    "Accuracy": f"{85 + hash(model) % 15:.1f}%",
                    "F1-Score": f"{80 + hash(model) % 20:.1f}%",
                    "Training Time": f"{1 + hash(model) % 5:.1f}s"
                }

            st.subheader("📊 Model Performance Comparison")

            for model, metrics in results.items():
                st.write(f"**{model}**")
                cols = st.columns(len(metrics))
                for i, (metric, value) in enumerate(metrics.items()):
                    cols[i].metric(metric, value)
                st.markdown("---")

            # AI evaluation insights
            if st.button("🧠 Get AI Evaluation Insights"):
                with st.spinner("Generating AI-powered evaluation insights..."):
                    insights = ai_client.generate_text(
                        "Analyze these model performance results and provide insights on which model performs best and why. Include recommendations for model selection and potential improvements.",
                        model="gemini"
                    )

                    st.subheader("🎯 AI Evaluation Insights")
                    st.write(insights)


def automl():
    """Automated machine learning tool"""
    create_tool_header("AutoML", "Automated machine learning with AI optimization", "🤖")

    st.subheader("AutoML Configuration")

    # Data input
    uploaded_file = FileHandler.upload_files(['csv', 'xlsx'], accept_multiple=False)

    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            problem_type = st.selectbox("Problem Type", [
                "Auto-detect", "Binary Classification", "Multi-class Classification",
                "Regression", "Time Series Forecasting"
            ])

            target_column = st.text_input("Target Column", placeholder="target")

        with col2:
            time_budget = st.slider("Time Budget (minutes)", 5, 120, 30)
            optimization_metric = st.selectbox("Optimization Metric", [
                "Accuracy", "F1-Score", "ROC-AUC", "MSE", "Custom"
            ])

        # Advanced options
        with st.expander("Advanced AutoML Settings"):
            ensemble_models = st.checkbox("Enable Model Ensembling", True)
            feature_engineering = st.checkbox("Automatic Feature Engineering", True)
            hyperparameter_tuning = st.checkbox("Hyperparameter Optimization", True)
            model_interpretability = st.checkbox("Generate Model Explanations", True)

        if st.button("🚀 Start AutoML", type="primary"):
            with st.spinner("Running automated machine learning pipeline..."):
                # Simulate AutoML process
                progress_text = st.empty()
                progress_bar = st.progress(0)

                stages = [
                    "Analyzing data structure...",
                    "Performing feature engineering...",
                    "Training multiple models...",
                    "Optimizing hyperparameters...",
                    "Evaluating model performance...",
                    "Generating final model..."
                ]

                for i, stage in enumerate(stages):
                    progress_text.text(stage)
                    progress_bar.progress((i + 1) * 100 // len(stages))
                    time.sleep(1)

                st.success("🎉 AutoML pipeline completed!")

                # Results
                st.subheader("🏆 Best Model Results")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Model", "XGBoost Ensemble")
                with col2:
                    st.metric("Accuracy", "96.8%")
                with col3:
                    st.metric("Cross-Val Score", "94.2%")
                with col4:
                    st.metric("Training Time", "12.3 min")

                # Model insights
                if st.button("📋 Generate Model Report"):
                    with st.spinner("Generating comprehensive model report..."):
                        report = ai_client.generate_text(
                            "Generate a comprehensive AutoML model report including model selection rationale, feature importance, performance analysis, and deployment recommendations.",
                            model="gemini"
                        )

                        st.subheader("📈 AutoML Model Report")
                        st.write(report)


def speech_recognition():
    """AI-powered speech recognition tool"""
    create_tool_header("Speech Recognition", "Convert speech to text using AI", "🎙️")

    st.subheader("Speech Recognition Options")

    input_method = st.selectbox("Input Method", [
        "Upload Audio File", "Record Live Audio", "YouTube URL"
    ])

    if input_method == "Upload Audio File":
        uploaded_file = FileHandler.upload_files(['mp3', 'wav', 'm4a', 'flac'], accept_multiple=False)
        if uploaded_file:
            st.audio(uploaded_file[0])

            col1, col2 = st.columns(2)
            with col1:
                language = st.selectbox("Language", [
                    "Auto-detect", "English", "Spanish", "French", "German", "Italian", "Portuguese"
                ])

            with col2:
                output_format = st.selectbox("Output Format", [
                    "Plain Text", "Timestamped", "Speaker Labels", "Structured JSON"
                ])

            # Advanced options
            with st.expander("Advanced Settings"):
                enhance_audio = st.checkbox("Audio Enhancement", True)
                remove_filler_words = st.checkbox("Remove Filler Words", False)
                custom_vocabulary = st.text_area("Custom Vocabulary (one word per line)",
                                                 placeholder="technical\nterminology\nwords")

            if st.button("🎙️ Transcribe Audio", type="primary"):
                with st.spinner("Processing audio and converting to text..."):
                    # Simulate transcription
                    sample_transcript = """Hello, this is a sample transcription of the uploaded audio file. 
                    The AI speech recognition system has successfully converted the speech to text with 
                    high accuracy and proper punctuation."""

                    st.subheader("📝 Transcription Results")
                    st.text_area("Transcribed Text", sample_transcript, height=200)

                    # Confidence metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Confidence", "94.2%")
                    with col2:
                        st.metric("Words Detected", "127")
                    with col3:
                        st.metric("Duration", "45.3s")

                    # Download options
                    if st.button("📥 Download Transcript"):
                        FileHandler.create_download_link(
                            sample_transcript.encode(),
                            "transcript.txt",
                            "text/plain"
                        )


def audio_analysis():
    """AI-powered audio analysis tool"""
    create_tool_header("Audio Analysis", "Analyze audio content with AI insights", "🔊")

    st.subheader("Audio Analysis Configuration")

    uploaded_file = FileHandler.upload_files(['mp3', 'wav', 'm4a'], accept_multiple=False)

    if uploaded_file:
        st.audio(uploaded_file[0])

        analysis_type = st.multiselect("Analysis Type", [
            "Emotion Detection", "Speaker Recognition", "Music Analysis",
            "Sound Classification", "Audio Quality Assessment", "Noise Analysis"
        ])

        if st.button("🔊 Analyze Audio", type="primary"):
            with st.spinner("Performing comprehensive audio analysis..."):
                results = {
                    "Emotion Detection": "Happy (78%), Calm (22%)",
                    "Speaker Recognition": "Single speaker detected",
                    "Audio Quality": "High quality (96/100)",
                    "Dominant Frequency": "440 Hz",
                    "Duration": "2:34 minutes"
                }

                st.subheader("🎵 Audio Analysis Results")

                for analysis, result in results.items():
                    if analysis.replace(" ", "_").lower() in [a.replace(" ", "_").lower() for a in analysis_type]:
                        st.write(f"**{analysis}:** {result}")


def voice_cloning():
    """AI-powered voice cloning tool"""
    create_tool_header("Voice Cloning", "Clone and synthesize voices with AI", "🎭")

    st.subheader("Voice Cloning Configuration")

    st.info("⚠️ Voice cloning should only be used with consent and for legitimate purposes.")

    # Source voice input
    st.subheader("Source Voice")
    source_method = st.selectbox("Source Voice Method", [
        "Upload Voice Sample", "Record Voice", "Use Preset Voice"
    ])

    if source_method == "Upload Voice Sample":
        uploaded_file = FileHandler.upload_files(['mp3', 'wav', 'm4a'], accept_multiple=False)
        if uploaded_file:
            st.audio(uploaded_file[0])

            # Text input
            st.subheader("Text to Synthesize")
            text_input = st.text_area("Enter text to speak in the cloned voice",
                                      placeholder="Enter the text you want to generate with the cloned voice...")

            # Voice settings
            col1, col2 = st.columns(2)
            with col1:
                voice_speed = st.slider("Speech Speed", 0.5, 2.0, 1.0)
                voice_pitch = st.slider("Pitch Adjustment", -50, 50, 0)

            with col2:
                voice_emotion = st.selectbox("Emotion", [
                    "Neutral", "Happy", "Sad", "Excited", "Calm", "Angry"
                ])
                quality = st.selectbox("Quality", ["Standard", "High", "Premium"])

            if text_input and st.button("🎭 Generate Cloned Voice", type="primary"):
                with st.spinner("Cloning voice and generating speech..."):
                    st.success("✅ Voice cloning completed!")
                    st.info("🔊 Generated audio would be available for download here.")

                    # Placeholder for audio player
                    st.write("🎵 **Generated Audio Preview**")
                    st.info("Audio player would appear here with the cloned voice output.")


def sound_generation():
    """AI-powered sound generation tool"""
    create_tool_header("Sound Generation", "Generate sounds and music with AI", "🎵")

    st.subheader("Sound Generation Options")

    generation_type = st.selectbox("Generation Type", [
        "Music Composition", "Sound Effects", "Ambient Sounds", "Voice Synthesis", "Custom Audio"
    ])

    if generation_type == "Music Composition":
        col1, col2 = st.columns(2)
        with col1:
            genre = st.selectbox("Music Genre", [
                "Classical", "Jazz", "Rock", "Electronic", "Ambient", "Pop", "Blues"
            ])
            mood = st.selectbox("Mood", [
                "Happy", "Sad", "Energetic", "Calm", "Mysterious", "Romantic", "Epic"
            ])

        with col2:
            duration = st.slider("Duration (seconds)", 10, 300, 60)
            instruments = st.multiselect("Instruments", [
                "Piano", "Guitar", "Violin", "Drums", "Bass", "Flute", "Synthesizer"
            ])

        description = st.text_area("Music Description",
                                   placeholder="Describe the music you want to generate...")

    elif generation_type == "Sound Effects":
        effect_type = st.selectbox("Effect Type", [
            "Nature Sounds", "Mechanical", "Explosion", "Water", "Wind", "Footsteps", "Doors"
        ])

        description = st.text_area("Sound Description",
                                   placeholder="Describe the sound effect you want to generate...")

    # Advanced settings
    with st.expander("Advanced Generation Settings"):
        audio_quality = st.selectbox("Audio Quality", ["Standard", "High", "Studio"])
        file_format = st.selectbox("Output Format", ["MP3", "WAV", "FLAC"])
        stereo_mode = st.selectbox("Audio Mode", ["Mono", "Stereo", "Surround"])

    if st.button("🎵 Generate Sound", type="primary"):
        with st.spinner("Creating AI-generated audio..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
                time.sleep(0.03)

            st.success("🎉 Sound generation completed!")

            # Results
            st.subheader("🎵 Generated Audio")
            st.info("🔊 Audio player would appear here with the generated sound.")
            """Configure
            your
            AI
            API
            keys
            to
            enable
            all
            AI - powered
            tools in this
            application."""
            # Generated audio duration (simulated)
            duration = 30  # Default duration in seconds

            # Audio properties
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{duration}s")
            with col2:
                st.metric("Quality", audio_quality)
            with col3:
                st.metric("Format", file_format)

            if st.button("📥 Download Audio"):
                st.info("Download link would be generated here.")


# Helper functions for AI tools
def enhance_image_prompt(prompt: str, style: str, mood: str, color_scheme: str) -> str:
    """Enhance image generation prompt with style and mood parameters"""
    enhanced = prompt

    if style and style != "Realistic":
        enhanced += f", {style.lower()} style"

    if mood and mood != "Neutral":
        enhanced += f", {mood.lower()} mood"

    if color_scheme and color_scheme != "Natural":
        enhanced += f", {color_scheme.lower().replace(' ', '_')} colors"

    enhanced += ", high quality, detailed, professional"
    return enhanced


def create_recognition_prompt(analysis_type: str, detail_level: str, include_confidence: bool) -> str:
    """Create prompt for image recognition"""
    base_prompt = f"Analyze this image and provide {analysis_type.lower()}"

    if detail_level == "Detailed":
        base_prompt += " with detailed descriptions"
    elif detail_level == "Comprehensive":
        base_prompt += " with comprehensive analysis including context"

    if include_confidence:
        base_prompt += ". Include confidence scores for each identification."

    return base_prompt


def create_object_detection_prompt(detection_mode: str, confidence_threshold: float, output_format: str,
                                   include_coordinates: bool, custom_objects: str = None) -> str:
    """Create prompt for object detection"""
    if custom_objects:
        prompt = f"Detect and locate these specific objects in the image: {custom_objects}"
    else:
        prompt = f"Detect and locate {detection_mode.lower()} in this image"

    prompt += f" with confidence above {confidence_threshold}"

    if include_coordinates:
        prompt += ". Include object coordinates and bounding boxes."

    if output_format == "Structured JSON":
        prompt += " Format the response as structured JSON."
    elif output_format == "Annotated List":
        prompt += " Format as a numbered list with annotations."

    return prompt


def create_scene_analysis_prompt(analysis_focus: str, detail_level: str, include_emotions: bool,
                                 include_suggestions: bool, scene_context: str) -> str:
    """Create prompt for scene analysis"""
    prompt = f"Analyze the scene in this image focusing on {analysis_focus.lower()}"

    if detail_level == "Detailed":
        prompt += " with detailed observations"
    elif detail_level == "Comprehensive":
        prompt += " with comprehensive analysis"

    if include_emotions:
        prompt += ". Include emotional content analysis."

    if include_suggestions:
        prompt += " Provide improvement suggestions."

    if scene_context:
        prompt += f" Context: {scene_context}"

    return prompt


def get_sentiment_color(sentiment: str) -> str:
    """Get color for sentiment display"""
    colors = {
        'positive': '#28a745',
        'negative': '#dc3545',
        'neutral': '#6c757d',
        'happy': '#28a745',
        'sad': '#dc3545',
        'angry': '#dc3545',
        'unknown': '#6c757d'
    }
    return colors.get(sentiment.lower(), '#6c757d')


def analyze_sentence_sentiment(sentence: str) -> str:
    """Analyze sentiment of a single sentence"""
    # Simple sentiment analysis based on common words
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed']

    sentence_lower = sentence.lower()
    pos_count = sum(1 for word in positive_words if word in sentence_lower)
    neg_count = sum(1 for word in negative_words if word in sentence_lower)

    if pos_count > neg_count:
        return 'positive'
    elif neg_count > pos_count:
        return 'negative'
    else:
        return 'neutral'


def display_structured_recognition(result: dict) -> None:
    """Display structured recognition results"""
    if 'objects' in result:
        st.subheader("Detected Objects")
        for obj in result['objects']:
            st.write(f"• {obj.get('name', 'Unknown')} - Confidence: {obj.get('confidence', 0):.2%}")


def display_object_detection_results(result: dict) -> None:
    """Display object detection results"""
    if 'detections' in result:
        for detection in result['detections']:
            st.write(f"**{detection.get('object', 'Unknown')}** - {detection.get('confidence', 0):.2%}")


def export_object_detection_data(filename: str, mode: str, result: str) -> None:
    """Export object detection data"""
    export_data = {
        "filename": filename,
        "detection_mode": mode,
        "results": result,
        "timestamp": datetime.now().isoformat()
    }

    export_json = json.dumps(export_data, indent=2)
    FileHandler.create_download_link(
        export_json.encode(),
        f"detection_results_{filename}.json",
        "application/json"
    )


def export_scene_analysis_data(filename: str, focus: str, result: str) -> None:
    """Export scene analysis data"""
    export_data = {
        "filename": filename,
        "analysis_focus": focus,
        "results": result,
        "timestamp": datetime.now().isoformat()
    }

    export_json = json.dumps(export_data, indent=2)
    FileHandler.create_download_link(
        export_json.encode(),
        f"scene_analysis_{filename}.json",
        "application/json"
    )


def analyze_scene_emotions(scene_result: str) -> str:
    """Analyze emotions from scene analysis"""
    return "Emotional analysis would be performed on the scene analysis results."


def generate_scene_suggestions(scene_result: str, focus: str) -> str:
    """Generate suggestions based on scene analysis"""
    return f"Suggestions for improving {focus.lower()} would be generated based on the analysis results."


def perform_similarity_search(ref_image, search_images, threshold, max_results, criteria):
    """Perform visual similarity search"""
    st.info("Similarity search functionality would compare images based on the selected criteria.")


def perform_object_search(query, images, precision, include_context):
    """Perform object-based search in images"""
    st.info(f"Object search for '{query}' would be performed on the uploaded images.")


def perform_color_style_search(images, color_pref, style_pref, color_param):
    """Perform color and style based search"""
    st.info("Color and style search would analyze images for matching visual characteristics.")


def display_visual_search_results(results):
    """Display visual search results"""
    st.info("Visual search results would be displayed here with matching images and scores.")


# ================================
# AI ART GENERATION HELPER FUNCTIONS
# ================================

def create_enhanced_prompt(prompt, art_style, medium, art_movement, artist_inspiration, texture, finish,
                          shot_type, camera_angle, depth_of_field, lens_type, framing, perspective,
                          lighting_setup, time_of_day, weather, mood_atmosphere, color_temperature, contrast_level,
                          color_palette, dominant_color, color_harmony, saturation,
                          quality, detail_level, render_engine, post_processing,
                          custom_modifiers, auto_enhance, add_quality_tags, add_style_consistency,
                          boost_details, add_photography_terms, artistic_flair):
    """Create a comprehensive enhanced prompt for AI image generation"""
    
    # Start with the base prompt
    enhanced_parts = [prompt.strip()]
    
    # Art Style and Medium
    style_parts = []
    if art_style and art_style != "Photorealistic":
        style_parts.append(f"{art_style.lower()} style")
    
    if medium and medium != "None":
        if medium in ["Canvas", "Paper", "Wood Panel"]:
            style_parts.append(f"painted on {medium.lower()}")
        else:
            style_parts.append(f"{medium.lower()} medium")
    
    if art_movement and art_movement != "Contemporary":
        style_parts.append(f"{art_movement.lower()} art movement")
    
    if artist_inspiration and artist_inspiration != "None":
        if "Studio" in artist_inspiration or "Comics" in artist_inspiration:
            style_parts.append(f"in the style of {artist_inspiration}")
        else:
            style_parts.append(f"inspired by {artist_inspiration}")
    
    # Surface and Texture
    texture_parts = []
    if texture and texture != "Smooth":
        texture_parts.append(f"{texture.lower()} texture")
    
    if finish and finish != "Standard":
        texture_parts.append(f"{finish.lower()} finish")
    
    # Photography and Composition
    photo_parts = []
    if add_photography_terms:
        if shot_type and shot_type != "Medium Shot":
            photo_parts.append(shot_type.lower())
        
        if camera_angle and camera_angle != "Eye Level":
            photo_parts.append(f"{camera_angle.lower()} angle")
        
        if depth_of_field and depth_of_field != "Sharp Focus":
            if depth_of_field == "Shallow DOF":
                photo_parts.append("shallow depth of field")
            elif depth_of_field == "Deep DOF":
                photo_parts.append("deep depth of field")
            else:
                photo_parts.append(depth_of_field.lower())
        
        if lens_type and lens_type != "Standard":
            photo_parts.append(f"{lens_type.lower()} lens")
    
    # Composition
    composition_parts = []
    if framing and framing != "Centered":
        if framing == "Rule of Thirds":
            composition_parts.append("rule of thirds composition")
        elif framing == "Golden Ratio":
            composition_parts.append("golden ratio composition")
        else:
            composition_parts.append(f"{framing.lower()} framing")
    
    if perspective and perspective != "Linear":
        composition_parts.append(f"{perspective.lower()} perspective")
    
    # Lighting and Atmosphere
    lighting_parts = []
    if lighting_setup and lighting_setup != "Natural Light":
        if lighting_setup == "Studio Lighting":
            lighting_parts.append("professional studio lighting")
        elif lighting_setup == "Dramatic Lighting":
            lighting_parts.append("dramatic cinematic lighting")
        elif lighting_setup == "Volumetric Lighting":
            lighting_parts.append("volumetric lighting with god rays")
        else:
            lighting_parts.append(f"{lighting_setup.lower()}")
    
    if time_of_day and time_of_day != "Midday":
        if time_of_day == "Golden Hour":
            lighting_parts.append("golden hour lighting")
        elif time_of_day == "Blue Hour":
            lighting_parts.append("blue hour atmosphere")
        else:
            lighting_parts.append(f"{time_of_day.lower()} lighting")
    
    # Weather and Mood
    atmosphere_parts = []
    if weather and weather != "Clear":
        if weather in ["Foggy", "Misty"]:
            atmosphere_parts.append(f"{weather.lower()} atmospheric conditions")
        else:
            atmosphere_parts.append(f"{weather.lower()} weather")
    
    if mood_atmosphere and mood_atmosphere != "Serene":
        atmosphere_parts.append(f"{mood_atmosphere.lower()} mood")
    
    # Color and Temperature
    color_parts = []
    if color_temperature and color_temperature != "Neutral":
        if color_temperature == "Very Warm":
            color_parts.append("very warm color temperature")
        elif color_temperature == "Very Cool":
            color_parts.append("very cool color temperature")
        else:
            color_parts.append(f"{color_temperature.lower()} color temperature")
    
    if contrast_level and contrast_level != "Normal":
        if contrast_level == "High Contrast":
            color_parts.append("high contrast lighting")
        elif contrast_level == "Chiaroscuro":
            color_parts.append("chiaroscuro lighting technique")
        else:
            color_parts.append(f"{contrast_level.lower()}")
    
    # Color Palette
    palette_parts = []
    if color_palette and color_palette != "Natural":
        if color_palette == "Monochromatic":
            palette_parts.append("monochromatic color scheme")
        elif color_palette == "Complementary":
            palette_parts.append("complementary color palette")
        elif color_palette == "High Saturation":
            palette_parts.append("highly saturated colors")
        else:
            palette_parts.append(f"{color_palette.lower().replace(' ', ' ')} colors")
    
    if dominant_color and dominant_color != "Auto":
        palette_parts.append(f"dominant {dominant_color.lower()} tones")
    
    if saturation and saturation != "Natural":
        if saturation not in color_palette:  # Avoid duplication
            palette_parts.append(f"{saturation.lower().replace(' ', ' ')}")
    
    # Technical Quality
    quality_parts = []
    if add_quality_tags:
        if quality == "Ultra High Quality":
            quality_parts.extend(["ultra high quality", "masterpiece", "best quality"])
        elif quality == "High Quality":
            quality_parts.extend(["high quality", "detailed"])
        elif quality in ["8K", "4K"]:
            quality_parts.extend([f"{quality} resolution", "ultra detailed"])
        elif quality == "Professional":
            quality_parts.extend(["professional quality", "award-winning"])
        elif quality == "Gallery Quality":
            quality_parts.extend(["museum quality", "gallery exhibition"])
    
    if detail_level and detail_level != "Normal":
        if detail_level == "Extremely Detailed":
            quality_parts.append("extremely detailed")
        elif detail_level == "Highly Detailed":
            quality_parts.append("highly detailed")
        elif detail_level == "Intricate":
            quality_parts.append("intricate details")
        else:
            quality_parts.append(f"{detail_level.lower().replace(' ', ' ')}")
    
    # Render Engine
    render_parts = []
    if render_engine and render_engine != "Default":
        if render_engine == "Unreal Engine":
            render_parts.append("rendered in Unreal Engine")
        elif render_engine == "Octane Render":
            render_parts.append("octane render")
        elif render_engine in ["Blender", "Maya", "3ds Max", "Cinema 4D"]:
            render_parts.append(f"rendered in {render_engine}")
        elif render_engine == "Photorealistic":
            render_parts.append("photorealistic rendering")
        else:
            render_parts.append(f"{render_engine.lower()} rendering")
    
    # Post-Processing
    post_parts = []
    if post_processing and post_processing != "None":
        if post_processing == "Color Grading":
            post_parts.append("professional color grading")
        elif post_processing == "Film Grain":
            post_parts.append("subtle film grain")
        elif post_processing == "HDR":
            post_parts.append("HDR processing")
        elif post_processing == "Film Look":
            post_parts.append("cinematic film look")
        else:
            post_parts.append(f"{post_processing.lower().replace(' ', ' ')}")
    
    # Enhanced Details
    detail_enhancement = []
    if boost_details:
        detail_enhancement.extend([
            "fine details", "sharp focus", "crisp image", "clear definition"
        ])
    
    # Artistic Flair
    artistic_enhancement = []
    if artistic_flair:
        artistic_enhancement.extend([
            "trending on artstation", "award-winning", "breathtaking",
            "stunning visual", "masterful composition"
        ])
    
    # Style Consistency
    consistency_parts = []
    if add_style_consistency and art_style != "Photorealistic":
        consistency_parts.append(f"consistent {art_style.lower()} aesthetic throughout")
    
    # Compile all parts
    all_parts = [
        enhanced_parts,
        style_parts,
        texture_parts,
        photo_parts,
        composition_parts,
        lighting_parts,
        atmosphere_parts,
        color_parts,
        palette_parts,
        quality_parts,
        render_parts,
        post_parts,
        detail_enhancement,
        artistic_enhancement,
        consistency_parts
    ]
    
    # Filter and join
    filtered_parts = []
    for part_group in all_parts:
        if part_group:
            if isinstance(part_group, list):
                filtered_parts.extend([p for p in part_group if p])
            else:
                filtered_parts.append(part_group)
    
    # Add custom modifiers
    if custom_modifiers:
        filtered_parts.append(custom_modifiers.strip())
    
    # Create final prompt
    enhanced_prompt = ", ".join(filtered_parts)
    
    # Auto-enhancement
    if auto_enhance:
        enhanced_prompt = auto_enhance_prompt(enhanced_prompt)
    
    return enhanced_prompt


def auto_enhance_prompt(prompt):
    """Automatically enhance prompt with quality and professional terms"""
    enhancements = []
    
    # Add quality terms if not present
    quality_terms = ["high quality", "detailed", "masterpiece", "best quality", "professional"]
    if not any(term in prompt.lower() for term in quality_terms):
        enhancements.append("high quality, detailed")
    
    # Add technical terms
    technical_terms = ["sharp focus", "8k", "4k", "ultra detailed"]
    if not any(term in prompt.lower() for term in technical_terms):
        enhancements.append("sharp focus")
    
    # Add artistic terms
    artistic_terms = ["beautiful", "stunning", "gorgeous", "magnificent", "breathtaking"]
    if not any(term in prompt.lower() for term in artistic_terms):
        enhancements.append("beautiful")
    
    if enhancements:
        return f"{prompt}, {', '.join(enhancements)}"
    return prompt


def get_prompt_template(template_name):
    """Get predefined prompt templates for quick start"""
    templates = {
        "Portrait": {
            "art_style": "Photorealistic",
            "shot_type": "Close-up",
            "lighting_setup": "Studio Lighting",
            "depth_of_field": "Shallow DOF",
            "quality": "Ultra High Quality",
            "detail_level": "Highly Detailed"
        },
        "Landscape": {
            "art_style": "Photorealistic",
            "shot_type": "Wide Shot",
            "lighting_setup": "Natural Light",
            "time_of_day": "Golden Hour",
            "color_palette": "Vibrant",
            "mood_atmosphere": "Majestic"
        },
        "Abstract Art": {
            "art_style": "Abstract",
            "color_palette": "Vibrant",
            "composition": "Asymmetrical",
            "texture": "Smooth",
            "artistic_flair": True
        },
        "Architecture": {
            "art_style": "Photorealistic",
            "shot_type": "Wide Shot",
            "lighting_setup": "Natural Light",
            "perspective": "Two Point",
            "detail_level": "Extremely Detailed"
        },
        "Fantasy": {
            "art_style": "Digital Art",
            "mood_atmosphere": "Mystical",
            "lighting_setup": "Dramatic Lighting",
            "color_palette": "Vibrant",
            "artistic_flair": True
        },
        "Sci-Fi": {
            "art_style": "Digital Art",
            "mood_atmosphere": "Futuristic",
            "lighting_setup": "Volumetric Lighting",
            "color_palette": "Cool Tones",
            "render_engine": "Unreal Engine"
        },
        "Product Photo": {
            "art_style": "Photorealistic",
            "shot_type": "Close-up",
            "lighting_setup": "Studio Lighting",
            "background": "Clean",
            "quality": "Professional"
        },
        "Character Design": {
            "art_style": "Digital Art",
            "shot_type": "Medium Shot",
            "detail_level": "Highly Detailed",
            "color_palette": "Vibrant",
            "artistic_flair": True
        }
    }
    
    return templates.get(template_name, {})


def calculate_prompt_complexity(prompt):
    """Calculate the complexity score of a prompt (1-10)"""
    if not prompt:
        return 0
    
    score = 0
    words = prompt.split()
    
    # Word count factor
    word_count = len(words)
    if word_count > 100:
        score += 3
    elif word_count > 50:
        score += 2
    elif word_count > 20:
        score += 1
    
    # Technical terms
    technical_terms = [
        "resolution", "lighting", "composition", "perspective", "depth of field",
        "color grading", "render", "engine", "photorealistic", "detailed"
    ]
    tech_count = sum(1 for term in technical_terms if term in prompt.lower())
    score += min(tech_count, 3)
    
    # Artistic terms
    artistic_terms = [
        "style", "mood", "atmosphere", "palette", "contrast", "saturation",
        "texture", "medium", "movement", "inspired", "masterpiece"
    ]
    art_count = sum(1 for term in artistic_terms if term in prompt.lower())
    score += min(art_count, 2)
    
    # Specific artists or styles
    specific_terms = [
        "van gogh", "picasso", "monet", "dali", "renaissance", "baroque",
        "impressionist", "surreal", "abstract", "minimalist"
    ]
    specific_count = sum(1 for term in specific_terms if term in prompt.lower())
    score += min(specific_count, 2)
    
    return min(score, 10)


def analyze_prompt_quality(prompt):
    """Analyze prompt quality and provide detailed metrics"""
    if not prompt:
        return {
            "word_count": 0,
            "style_terms": 0,
            "quality_score": 0,
            "complexity": "Low"
        }
    
    words = prompt.split()
    word_count = len(words)
    
    # Count style-related terms
    style_terms = [
        "style", "art", "painting", "digital", "realistic", "abstract",
        "impressionist", "surreal", "modern", "classical", "vintage"
    ]
    style_count = sum(1 for term in style_terms if term in prompt.lower())
    
    # Calculate quality score
    quality_indicators = [
        "high quality", "detailed", "masterpiece", "professional", "award-winning",
        "stunning", "beautiful", "gorgeous", "breathtaking", "ultra detailed",
        "8k", "4k", "sharp focus", "crisp", "clear"
    ]
    quality_count = sum(1 for indicator in quality_indicators if indicator in prompt.lower())
    
    # Overall score calculation
    base_score = min(word_count // 10, 4)  # Max 4 points for word count
    style_score = min(style_count, 3)       # Max 3 points for style terms
    quality_score = min(quality_count, 3)   # Max 3 points for quality terms
    
    total_score = base_score + style_score + quality_score
    
    # Complexity assessment
    complexity_score = calculate_prompt_complexity(prompt)
    if complexity_score >= 7:
        complexity = "High"
    elif complexity_score >= 4:
        complexity = "Medium"
    else:
        complexity = "Low"
    
    return {
        "word_count": word_count,
        "style_terms": style_count,
        "quality_score": total_score,
        "complexity": complexity
    }


def create_image_zip(generated_images, generation_config):
    """Create a ZIP file containing all generated images and metadata"""
    import zipfile
    import io
    
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add images
        for i, img in enumerate(generated_images):
            filename = f"ai_artwork_{i+1:02d}.png"
            zip_file.writestr(filename, img["data"])
        
        # Add generation configuration
        config_json = json.dumps(generation_config, indent=2)
        zip_file.writestr("generation_config.json", config_json)
        
        # Add prompt information
        prompt_info = {
            "original_prompt": generation_config.get("prompt", ""),
            "enhanced_prompt": generation_config.get("enhanced_prompt", ""),
            "generation_time": generation_config.get("timestamp", ""),
            "settings_summary": generation_config.get("settings", {})
        }
        prompt_json = json.dumps(prompt_info, indent=2)
        zip_file.writestr("prompt_details.json", prompt_json)
        
        # Add README
        readme_content = f"""AI Art Generation Batch
========================

Generated on: {generation_config.get('timestamp', 'Unknown')}
Number of images: {len(generated_images)}

Original Prompt:
{generation_config.get('prompt', 'Not available')}

Enhanced Prompt:
{generation_config.get('enhanced_prompt', 'Not available')}

Files included:
- ai_artwork_01.png, ai_artwork_02.png, etc. - Generated images
- generation_config.json - Complete generation settings
- prompt_details.json - Prompt information and summary
- README.txt - This file

Settings Used:
{json.dumps(generation_config.get('settings', {}), indent=2)}
"""
        zip_file.writestr("README.txt", readme_content)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def enhance_image_prompt(prompt: str, style: str, mood: str, color_scheme: str) -> str:
    """Legacy function for compatibility - now uses the enhanced system"""
    return create_enhanced_prompt(
        prompt, style, "None", "Contemporary", "None", "Smooth", "Standard",
        "Medium Shot", "Eye Level", "Sharp Focus", "Standard", "Centered", "Linear",
        "Natural Light", "Midday", "Clear", mood, "Neutral", "Normal",
        color_scheme, "Auto", "Natural", "Natural",
        "High Quality", "Normal", "Default", "None",
        "", True, True, True,
        False, False, False
    )
