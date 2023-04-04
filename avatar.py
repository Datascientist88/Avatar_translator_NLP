
import base64
import io
import cv2
import numpy as np
import pyttsx3
import dash
from dash import html ,dcc
from dash.dependencies import Input, Output, State
from PIL import Image, ImageDraw
from googletrans import LANGUAGES, Translator
import moviepy.editor as mp
# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150) # Set speaking rate to 150 words per minute
# Initialize avatar image
avatar_image = Image.open("avatar.png")
# Define function for translating text
def translate_text(text, source_language, target_language):
    translator = Translator()
    translation = translator.translate(text, src=source_language, dest=target_language)
    return translation.text
# Define function for creating video clip
def create_video(text, source_language, target_language, image_data):
    # Translate text
    translated_text = translate_text(text, source_language, target_language)
    # Convert text to speech
    with io.BytesIO() as speech_file:
        engine.save_to_file(translated_text, speech_file)
        speech_file.seek(0)
        speech_data = speech_file.read()
    # Process avatar image
    avatar_width, avatar_height = avatar_image.size
    text_width, text_height = avatar_width - 50, avatar_height - 100
    draw = ImageDraw.Draw(avatar_image)
    draw.rectangle([(25, 25), (avatar_width - 25, avatar_height - 25)], outline="white", width=10)
    draw.multiline_text((25, 50), translated_text, fill="white", font=("Arial", 30), width=text_width, align="center")
    # Initialize video clip with avatar image
    clip_duration = len(translated_text) // 5 + 1 # Estimate clip duration based on text length
    clip_width = 640
    clip_height = 480
    background_color = (255, 255, 255)
    clip = mp.ImageClip(np.zeros((clip_height, clip_width, 3), dtype=np.uint8) + background_color)
    avatar_clip = mp.ImageClip(np.array(avatar_image)).set_position(("center", "center")).set_duration(clip_duration)
    clip = mp.CompositeVideoClip([clip, avatar_clip])
    # Add mouth movement to video clip
    global face_cascade
    face_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
    fps = clip.fps
    for t in range(int(clip_duration * fps)):
        frame = clip.get_frame(t / fps)
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            mouth_y = y + 0.6 * h
            mouth_x1 = x + 0.3 * w
            mouth_x2 = x + 0.7 * w
            draw = ImageDraw.Draw(avatar_image)
            draw.rectangle([(mouth_x1, mouth_y), (mouth_x2, mouth_y + 0.1 * h)], fill="white")
        image_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield image_data

    # Convert video clip to a base64-encoded data URI
    video_file = io.BytesIO()
    clip.write_videofile(video_file, audio=mp.AudioClip(speech_data), fps=fps, codec="libx264")
    video_file.seek(0)
    video_data = base64.b64encode(video_file.read()).decode("utf-8")
    video_uri = "data:video/mp4;base64," + video_data
    return video_uri
# Define Dash app--------------------------------------------------------------------------------------------------------------------------
app = dash.Dash(__name__)
# Define app layout
app.layout = html.Div([
    html.H1("Avatar Translator"),
    html.Label("Source Language:"),
    dcc.Dropdown(
        id="source-dropdown",
        options=[{"label": LANGUAGES[lang], "value": lang} for lang in LANGUAGES],
        value="en"
    ),
    html.Label("Target Language:"),
    dcc.Dropdown(
        id="target-dropdown",
        options=[{"label": LANGUAGES[lang], "value": lang} for lang in LANGUAGES],
        value="es"
    ) ,
          html.Label("Image Upload:") ,
          dcc.Upload(  
     id="image-upload",
        children=html.Div([
            "Drag and drop or click to select an image file to upload."
        ]),
        style={
            "width": "100%",
            "height": "100px",
            "lineHeight": "100px",
            "borderWidth": "1px",
            "borderStyle": "dashed",
            "borderRadius": "5px",
            "textAlign": "center",
            "margin": "10px"
        },
        multiple=False
    ),
    html.Button("Translate", id="translate-button", n_clicks=0),
    html.Video(id="video-output", controls=True, style={"margin": "10px"}),
])

# Define callback for updating image upload text
@app.callback(Output("image-upload", "children"), [Input("image-upload", "filename")])
def update_upload_text(filename):
    if filename is not None:
        return html.Div([
            html.P(f"Selected file: {filename}")
        ])
    else:
        return html.Div([
            "Drag and drop or click to select an image file to upload."
        ])

# Define callback for translating text and creating video
@app.callback(Output("video-output", "src"), 
              [Input("translate-button", "n_clicks")], [State("text-input", "value"), 
                State("source-dropdown", "value"), State("target-dropdown", "value"), 
                State("image-upload", "contents")])
def translate_and_create_video(n_clicks, text, source_language, target_language, image_data):
    if n_clicks > 0:
        if image_data is not None:
            # Decode image data
            image_bytes = base64.b64decode(image_data.split(",")[1])
            image = Image.open(io.BytesIO(image_bytes))
            # Resize image to avatar size
            avatar_width, avatar_height = avatar_image.size
            image.thumbnail((avatar_width, avatar_height))
            # Create video clip with image background
            clip_duration = 3 # Set clip duration to 3 seconds
            clip_width = 640
            clip_height = 480
            background_color = (255, 255, 255)
            clip = mp.ImageClip(np.zeros((clip_height, clip_width, 3), dtype=np.uint8) + background_color)
            image_clip = mp.ImageClip(np.array(image)).set_position(("center", "center")).set_duration(clip_duration)
            clip = mp.CompositeVideoClip([clip, image_clip])
            # Add mouth movement to video clip
            fps = clip.fps
            for t in range(int(clip_duration * fps)):
                frame = clip.get_frame(t / fps)
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    mouth_y = y + 0.6 * h
                    mouth_x1 = x + 0.3 * w
                    mouth_x2 = x + 0.7 * w
                    draw = ImageDraw.Draw(image)
                    draw.rectangle([(mouth_x1, mouth_y), (mouth_x2, mouth_y + 0.1 * h)], fill="white")
                image_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield image_data
            # Convert video clip to a base64-encoded data URI
            video_file = io.BytesIO()
            clip.write_videofile(video_file, audio=mp.AudioFileClip("silent.wav"), fps=fps, codec="libx264")
            video_file.seek(0)
            video_data = base64.b64encode(video_file.read()).decode("utf-8")
            video_uri = "data:video/mp4;base64," + video_data

            return video_uri
        else:
            return None

if __name__ == "__main__":
    app.run_server(debug=True, port =8000)

      






      