import streamlit as st
from transformers import PegasusForConditionalGeneration, AutoTokenizer

# Function to summarize text
def summarize_text(model, tokenizer, src_text):
    tokens = tokenizer(src_text, truncation=True, padding="longest", return_tensors="pt")
    summary = model.generate(**tokens)
    print(tokenizer.decode(summary[0]))
    return tokenizer.decode(summary[0])

# Load model and tokenizer
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")

# Streamlit app
st.title("Text Summarization Using Transformers")

# Sidebar input text
input_text = st.text_area("Enter the text to summarize")

# Summarize button
if st.button("Summarize"):
    if input_text.strip() != "":
        summary_text = summarize_text(model, tokenizer, [input_text])
        
        st.subheader("Summary")
        st.write(summary_text)
    else:
        st.warning("Please enter some text to summarize")
