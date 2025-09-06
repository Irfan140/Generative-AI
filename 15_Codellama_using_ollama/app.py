import requests
import json

# Gradio for UI
import gradio as gr

# It is the endpoint where ollama is running
url="http://localhost:11434/api/generate"

headers={
    'Content-Type':'application/json'
}

history=[]

# Function to generate response from the model
def generate_response(prompt):
    # Also maintaining the history of the conversation
    history.append(prompt)
    final_prompt="\n".join(history)

    # Data to be sent to the model
    data={
        "model":"codeguru",
        "prompt":final_prompt,
        "stream":False
    }


    response=requests.post(url,headers=headers,data=json.dumps(data))

    # we are checking if the response is okay
    if response.status_code==200:
        response=response.text
        data=json.loads(response)
        actual_response=data['response']
        return actual_response
    else:
        print("error:",response.text)


# Gradio interface
interface=gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=4,placeholder="Enter your Prompt"),
    outputs="text"
)

# Launching the interface
interface.launch()
