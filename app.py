import streamlit as st
import bitsandbytes
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#load the model(Do this outside the streamlit app to avoid reloading on every action)
@st.cache_resource #Cache the model to prevent reloading
#loading a model like this takes a lot of time, we dont want to reload it everytime we interact with it
#we ONLY want to do this one time, then cache the result of the load_model function to use for the duration of the use


#this will handle loading the actual model from huggingface and picking the tokenizer based on the model_id, 
# it will also denote if we are using our GPU or CPU based on whats available
def load_model():
    #change this right here to select a different model from huggingface
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id) #The tokenizer will automatically be selected based on the name of the model

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    #This is where we are actually loading in the model based on the parameters we set above, if we want another model, we make the changes in model_id
    #device_map will try to spread over multiple GPUs if available, cool.
    #load_in_8bit = True is CRITICAL! it loads the model in "8-bit quantization" which reduces the memory load for GPUs with limited VRAM
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map = "auto", load_in_8bit = True)
    model.eval()
    #this function returns the loaded tokenizer, the model, and the device the model is loaded on
    return tokenizer, model, device

tokenizer, model, device = load_model()
st.title("Deepseek R1 1.5B Chatbot - Based on Qwen")

user_input = st.text_input("Enter a message: ")

#when we detect a message from the user
if user_input:
    #run the tokenizer to generate tokens from user input, "return_tensors" tells the tokenizer to return PyTorch tensors
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    #this context manager disables gradient calculations, we dont need them during text generation so 
    #disabling them will save memory and speed things up <-- Research this in more depth
    with torch.no_grad():
        #tell the model to generate an output based on the input
        outputs = model.generate(**inputs, max_new_tokens=200)

    #our output text needs to then take the mathematical response from the model and turn it back into words
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #output the response to the chat
    st.write("ChinaBot5000:", generated_text)
