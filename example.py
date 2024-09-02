import os
import streamlit as st
import json
# from my_component import my_component
import requests
import time
import google.generativeai as genai
from typing import List, Tuple, Union
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
from IPython.display import Image
import io
from imutils.video import VideoStream
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from google.cloud import language_v2
import cv2
import joblib
from PIL import Image
from face_recognition import preprocessing
from inference.util import draw_bb_on_img
from inference.constants import MODEL_PATH
import av
from pathlib import Path
from streamlit_webrtc import VideoProcessorBase,WebRtcMode,webrtc_streamer
from threading import Thread
import base64


# Create a _RELEASE constant. We'll set this to False while we're developing
# the component, and True when we're ready to package and distribute it.
# (This is, of course, optional - there are innumerable ways to manage your
# release process.)
_RELEASE = True

# Declare a Streamlit component. `declare_component` returns a function
# that is used to create instances of the component. We're naming this
# function "_comsponent_func", with an underscore prefix, because we don't want
# to expose it directly to users. Instead, we will create a custom wrapper
# function, below, that will serve as our component's public API.

# It's worth noting that this call to `declare_component` is the
# *only thing* you need to do to create the binding between Streamlit and
# your component frontend. Everything else we do in this file is simply a
# best practice.

if not _RELEASE:
    _component_func = components.declare_component(
        # We give the component a simple, descriptive name ("my_component"
        # does not fit this bill, so please choose something better for your
        # own component :)
        "my_component",
        # Pass `url` here to tell Streamlit that the component will be served
        # by the local dev server that you run via `npm run start`.
        # (This is useful while your component is in development.)
        url="http://localhost:3001",
    )
else:
    # When we're distributing a production version of the component, we'll
    # replace the `url` param with `path`, and point it to the component's
    # build directory:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component("my_component", path=build_dir)


# Create a wrapper function for the component. This is an optional
# best practice - we could simply expose the component function returned by
# `declare_component` and call it done. The wrapper allows us to customize
# our component's API: we can pre-process its input args, post-process its
# output value, and add a docstring for users.
def my_component(name, key=None):
    """Create a new instance of "my_component".

    Parameters
    ----------
    name: str
        The name of the thing we're saying hello to. The component will display
        the text "Hello, {name}!"
    key: str or None
        An optional key that uniquely identifies this component. If this is
        None, and the component's arguments are changed, the component will
        be re-mounted in the Streamlit frontend and lose its current state.

    Returns
    -------
    int
        The number of times the component's "Click Me" button has been clicked.
        (This is the value passed to `Streamlit.setComponentValue` on the
        frontend.)

    """
    # Call through to our private component function. Arguments we pass here
    # will be sent to the frontend, where they'll be available in an "args"
    # dictionary.
    #
    # "default" is a special argument that specifies the initial return
    # value of the component before the user has interacted with it.
    component_value = _component_func(name=name, key=key, default=0)

    # We could modify the value returned from the component if we wanted.
    # There's no need to do this in our simple example - but it's an option.
    return component_value


GOOGLE_API_KEY="{API-KEY}"

genai.configure(api_key=GOOGLE_API_KEY)

face_recogniser = joblib.load(MODEL_PATH)

good_img=None
global x
x=0
def counter():
    global x
    x=x+1



# def video_frame_callback(frame):
#     # global x
#     # print(x)
#     # if x%5==0:
#     while True:
#         print(5 - countdown(), "seconds remaining") #Still time left code
#         if timer.remaining() <= 0:
#             img = frame.to_ndarray(format="rgb24")
#             img = cv2.resize(img, (300, 300))
#             img = Image.fromarray(img)
#             faces = face_recogniser(img)
#             if faces is not None:
#                 draw_bb_on_img(faces,img)
#             timer.reset() #Starts timer again
#     # good_img=img
#     # else:
#     #     img=good_img
#         return av.VideoFrame.from_image(img)

class VideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.pure_img = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="rgb24")
        self.pure_img = image
        
        return av.VideoFrame.from_ndarray(image, format="rgb24")

# class VideoProcessor(VideoProcessorBase):
#     def __init__(self) -> None:
#         self.pure_img = False

#     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#         img = frame.to_ndarray(format="rgb24")
#         img = cv2.resize(img, (300, 300))
#         img = Image.fromarray(img)
#         faces = face_recogniser(img)
#         if faces is not None:
#             draw_bb_on_img(faces,img)
        
#         return av.VideoFrame.from_image(img)

model = genai.GenerativeModel('gemini-1.5-flash')

def generate_response(prompt):
    response=model.generate_content(prompt,stream=False)
    return st.markdown(response.text)


# st.header("Vertex AI Gemini 1.5 API", divider="rainbow")
# gemini_15_flash, gemini_15_pro = load_models()
st.header("Places + Gemini Demo")
place_id = my_component("World")
st.subheader("Selected place:")
if place_id:
    place_details=json.loads(place_id)
    st.code(place_details["name"], language="markdown")
else:
     st.code("No place selected yet", language="markdown")
tab1, tab2, tab3 = st.tabs(
    ["Company Analysis", "Address checker", "Face recognition"]
)
with tab1:
    st.header("Company Analysis")
    st.subheader("Find companies with the map above to analyze trustability using reviews")

    if place_id:
        place_details=json.loads(place_id) 
        client = language_v2.LanguageServiceClient()

        # Optional. If not specified, the language is automatically detected.
        # For list of supported languages:
        # https://cloud.google.com/natural-language/docs/languages


        # Available values: NONE, UTF8, UTF16, UTF32.
        # See https://cloud.google.com/natural-language/docs/reference/rest/v2/EncodingType.
        encoding_type = language_v2.EncodingType.UTF8
        

        result = st.button("Analyze", key="tab1")
        if result:
            with st.spinner(
                f"Fetching reviews..."
            ):
                place_api_response = requests.get("https://maps.googleapis.com/maps/api/place/details/json?place_id=" + place_details["place_id"] + "&key={API-KEY}")
                place_api_data = json.loads(json.dumps(place_api_response.json()))
                if place_api_data:
                        try:
                            if place_api_data["result"]["reviews"]:
                                reviews=place_api_data["result"]["reviews"]
                                df = pd.DataFrame(
                                     columns=['reviews', 'sentiment','suggested response']
                                )
                                output_string = ""
                                with st.spinner(
                                    f"Running sentiment analysis..."
                                ):
                                    for i, review in enumerate(reviews):
                                        output_string += f"- review{i+1}: {review['text']}\n"
                                        document = {"content": review['text'],"type": "PLAIN_TEXT"}
                                        # response = client.analyze_entities(
                                        #             request={"document": document, "encoding_type": encoding_type}
                                        #         )
                                        sentiment_response=requests.post("https://language.googleapis.com/v1/documents:analyzeSentiment?key={API-KEY}", headers={"Content-Type": "application/json; charset=utf-8"}, data=json.dumps({"document":document, "encodingType":"UTF8"}))
                                        sentiment_response_data=json.loads(json.dumps(sentiment_response.json()))
                                        print(sentiment_response_data)
                                        generate_response=model.generate_content("You are a company's customer service agent. Generate a compassionate response, based on these reviews: " + review['text'] + "only respond with the response, nothing else.",stream=False)
                                        list_row = [review['text'], "Sentiment score: " + str(sentiment_response_data['documentSentiment']['score']) + "\nDocument sentiment magnitude: " + str(sentiment_response_data['documentSentiment']['magnitude']),generate_response.text]
                                        df.loc[i]=list_row
                                        # df._append(list_row, ignore_index=True)
                                    # st.write(prompt)
                                    response = model.generate_content("Generate a score between 1-10 of this company's trustability based on these reviews: " + output_string + "do not answer anything else other than a score of 1-10 and your reasoning",stream=False)
                                    st.markdown("Trustability score:\n"+response.text)
                                    st.table(df)
                            else:
                                st.markdown("No reviews found about the place")
                        except KeyError:
                            st.markdown("No reviews found..")
                else:
                        st.markdown("No reviews found about the place")
    else:
        st.text("input something..")

with tab2:
    #  place_id = my_component("World")
    st.subheader("Look for similar places/businesses near an address. Search the address above, then match with the input below")
    business_query=st.text_input("Find a business", "Pecel lele")
    if place_id:
         result_tab2 = st.button("Analyze", key="tab2")
         if result_tab2:
            lat = str(place_details["geometry"]["location"]["lat"])
            long = str(place_details["geometry"]["location"]["lng"])
            st.text("lat: " + str(lat) + ", long: " + str(long))
            with st.spinner(
                f"Analyzing.."
            ):
                no_of_businesses=0
                nearby_search_response=requests.get("https://maps.googleapis.com/maps/api/place/nearbysearch/json?location="+lat+"%2C"+long+"&radius=500&key={API-KEY}")
                nearby_search_response_data=json.loads(json.dumps(nearby_search_response.json()))
                nearby_businesses=nearby_search_response_data["results"]
                output_string_nearby = ""
                for i, nearby_businesses in enumerate(nearby_businesses):
                    output_string_nearby += f"{i+1}. {nearby_businesses['name']} , type:{nearby_businesses['types']}\n"
                    no_of_businesses+=1
                no_of_businesses+=1
                print("blahblah: " + str(no_of_businesses))
                next_page_token=nearby_search_response_data['next_page_token']
                while "next_page_token" in nearby_search_response_data:
                    time.sleep(2)
                    next_page_token=nearby_search_response_data['next_page_token']
                    nearby_search_response=requests.get("https://maps.googleapis.com/maps/api/place/nearbysearch/json?pagetoken="+ next_page_token+"&key={API-KEY}")
                    nearby_search_response_data=json.loads(json.dumps(nearby_search_response.json()))
                    nearby_businesses=nearby_search_response_data["results"]
                    for i, nearby_businesses in enumerate(nearby_businesses):
                        output_string_nearby += f"{no_of_businesses+1}. {nearby_businesses['name']} , type:{nearby_businesses['types']}\n"
                        no_of_businesses+=1
                    print("counter is at:" + str(no_of_businesses))
                print(output_string_nearby)
                response_nearby_business = model.generate_content("Based on this type of business: " + business_query + " Is there a place of business strictly with the same name as the input above, from this list of businesses: "+ output_string_nearby +" the list format is business1 = name of business, type['']. answer only with the name of the business(es) that might be similar/relate very closely, and other possibilities",stream=False)
                st.markdown(response_nearby_business.text)
                st.text("List of businesses:\n" + output_string_nearby)
    with tab3:
        st.title("Face Recognition")
        interval = st.slider("Select interval", 1,10,5)
        student_name= st.text_input("Insert student name here..", "Cliff")
        # sched.add_job(counter, 'interval', seconds=1)
        # sched.start()
        stream = webrtc_streamer(mode=WebRtcMode.SENDRECV,key="example", video_processor_factory=VideoProcessor,async_processing=True,media_stream_constraints={"video": True, "audio": False})
        if stream.state.playing:
            images=[]
            stop_button_pressed=st.button("STOP!!!!!")
            while True:
                time.sleep(interval)
                
                
                img = stream.video_processor.pure_img
                img = cv2.resize(img, (300, 300))

                img = Image.fromarray(img)
                # img.save("temp.jpeg")
                # thread=Thread(target=generate_response, args=(proctor_prompt,))
                # thread.start()
                # thread.join()
                # prompt_img= Part.from_data(
                #     mime_type="image/jpeg",
                #     data=img.tobytes()
                # )
            
                faces = face_recogniser(img)
                if faces is not None:
                    draw_bb_on_img(faces,img)
                
                st.image(img)
                proctor_prompt= ["You are an exam proctor, based on the image given, find any suspicious things the student\
                            might be doing,Whether that person is using any sort of headwear/ glasses/ earphone / mask during the video. \
                            only respond with this json format={'student_name_match':True/False,'action':'normal'/'looking away'/'talking with others'(and \
                            whatever else you find), 'liveness':'live'/'spoof','detected_accessories':'earphone/headwear/mask'}. The student name is " + student_name, img]
                generate_response(proctor_prompt)
                if stop_button_pressed:
                    break
                    
                

        # cap = cv2.VideoCapture(-1)
        # frame_placeholder = st.empty()
        # stop_button_pressed=st.button("Stop")
        # while cap.isOpened():
        #     ret, frame = cap.read()
        #     frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     frame_placeholder.image(frame,channels="RGB")
        #     if stop_button_pressed:
        #         break

        # img_file_buffer = st.camera_input("Take a picture")

        # if img_file_buffer is not None:
        #     # To read image file buffer with OpenCV:
        #     bytes_data = img_file_buffer.getvalue()
        #     cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        #     preprocess = preprocessing.ExifOrientationNormalize()
        #     img = Image.open(io.BytesIO(bytes_data))
        #     print(bytes_data)
        #     filename = img.filename
        #     img = preprocess(img)
        #     img = img.convert('RGB')

        #     faces, img = recognise_faces(img)
        #     if not faces:
        #         print('No faces found in this image.')

            
        #     st.image(img)

            # # Check the type of cv2_img:
            # # Should output: <class 'numpy.ndarray'>
            # st.write(type(cv2_img))

            # # Check the shape of cv2_img:
            # # Should output shape: (height, width, channels)
            # st.write(cv2_img.shape)



    
# st.markdown("---")
# st.subheader("Component with variable args")

# # Create a second instance of our component whose `name` arg will vary
# # based on a text_input widget.
# #
# # We use the special "key" argument to assign a fixed identity to this
# # component instance. By default, when a component's arguments change,
# # it is considered a new instance and will be re-mounted on the frontend
# # and lose its current state. In this case, we want to vary the component's
# # "name" argument without having it get recreated.
# name_input = st.text_input("Enter a name", value="Streamlit")
# num_clicks = my_component(name_input, key="foo")
# st.markdown("You've clicked %s times!" % int(num_clicks))
