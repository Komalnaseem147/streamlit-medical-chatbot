import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# Load model and tokenizer once
@st.cache_resource
def load_model():
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    checkpoint_path = "./llama3-medchatbot/checkpoint-17850"

    # Try loading tokenizer from Hugging Face, fallback to checkpoint
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        st.warning("Failed to load tokenizer from Hugging Face. Using checkpoint.")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Try loading base model from Hugging Face
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
    except Exception as e:
        st.warning("Failed to load base model from Hugging Face. Check access or network.")
        raise  # Re-raise to stop if base model fails

    # Apply PEFT weights from checkpoint
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

# Inference function
def get_response(question):
    prompt = f"""### Instruction: You are a medical assistant. Answer the following question in a short and clear way.
    ### Input: {question}
    ### Response:"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip()

# Streamlit UI
st.title("Medical Chatbot")

user_question = st.text_area("Ask a medical question:")

if st.button("Get Response"):
    if user_question.strip():
        with st.spinner("Generating response..."):
            answer = get_response(user_question)
            st.success("**Response:**")
            st.markdown(answer)
    else:
        st.warning("Please enter a question.")