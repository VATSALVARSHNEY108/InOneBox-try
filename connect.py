import streamlit as st

def display_connect_page():
    """Display the Connect page with beautiful styling"""
    # Connect page header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>📞 Connect With Us</h1>
        <div style="background: linear-gradient(45deg, #a8c8ff, #c4a7ff); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                    background-clip: text; font-size: 1.2rem; font-weight: 500;">
            ✨ Get in touch, share feedback, or connect with the community ✨
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Contact sections
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💬 Feedback & Support")
        st.markdown("""
        **Found a bug?** Let me know and help me improve the toolkit!  
        **Have a suggestion?** I'm always looking for ways to make it better.  
        **Need help?** I'm here to support you with any of the tools.
        """)

        # Feedback form
        st.subheader("📝 Send Feedback")

        feedback_type = st.selectbox("Feedback Type", [
            "Bug Report", "Feature Request", "General Feedback", "Support Request", "Compliment"
        ])

        name = st.text_input("Your Name (optional)", placeholder="John Doe")
        email = st.text_input("Your Email (optional)", placeholder="john@example.com")

        subject = st.text_input("Subject", placeholder="Brief description of your message")
        message = st.text_area("Message", height=150,
                               placeholder="Tell us more about your feedback, issue, or suggestion...")

        if st.button("📤 Send Feedback", type="primary"):
            if message and subject:
                # Store feedback (in a real app, you'd send this to a backend)
                st.success("✅ Thank you for your feedback! I'll review it and get back to you if needed.")
                st.balloons()

                # Display what was submitted (for demo purposes)
                with st.expander("📋 Feedback Submitted"):
                    st.write(f"**Type:** {feedback_type}")
                    if name: st.write(f"**Name:** {name}")
                    if email: st.write(f"**Email:** {email}")
                    st.write(f"**Subject:** {subject}")
                    st.write(f"**Message:** {message}")
            else:
                st.error("Please fill in both subject and message fields.")

    with col2:
        st.markdown("### 🌐 Connect With Me")
        st.markdown("""
        **Follow my updates** and join the community!  
        **Share your creations** made with the toolkit.  
        **Stay informed** about new features and releases.
        """)

        # Social media / connection links
        st.subheader("🔗 Find Me Online")

        # Social and contact links
        social_links = [
            ("🐙 GitHub", "https://github.com/username", "View my projects and code"),
            ("💼 LinkedIn", "https://linkedin.com/in/username", "Connect professionally"),
            ("🐦 Twitter", "https://twitter.com/username", "Follow me for updates"),
            ("📧 Email", "mailto:contact@example.com", "Send me a direct email"),
            ("📚 Portfolio", "https://portfolio.dev", "Check out my other projects")
        ]

        for icon_name, link, description in social_links:
            st.markdown(f"**{icon_name}** [{description}]({link})")

        # Newsletter signup
        st.subheader("📰 Newsletter")
        newsletter_email = st.text_input("Subscribe for updates:", placeholder="your@email.com")
        if st.button("📮 Subscribe"):
            if newsletter_email and "@" in newsletter_email:
                st.success("✅ Subscribed! You'll receive updates about new tools and features.")
            else:
                st.error("Please enter a valid email address.")

    st.markdown("---")

    # FAQ Section
    st.subheader("❓ Frequently Asked Questions")

    faqs = [
        {
            "question": "Is this toolkit free to use?",
            "answer": "Yes! Most tools are completely free. Some AI-powered features require API keys from providers like Google (Gemini) or OpenAI."
        },
        {
            "question": "How do I report a bug or request a feature?",
            "answer": "Use the feedback form above or contact me through any of the social channels. I review all submissions personally!"
        },
        {
            "question": "Can I contribute to the project?",
            "answer": "Absolutely! Check out my GitHub repository to see how you can contribute code, documentation, or ideas."
        },
        {
            "question": "Are my files and data secure?",
            "answer": "Yes! All processing happens locally in your browser or on secure servers. I don't store your personal files or data."
        },
        {
            "question": "How often are new tools added?",
            "answer": "I'm constantly working on new tools and improvements. Follow my updates to stay informed about releases!"
        }
    ]

    for i, faq in enumerate(faqs):
        with st.expander(f"**{faq['question']}**"):
            st.write(faq['answer'])
