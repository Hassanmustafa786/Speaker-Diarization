from pyannote.audio import Model, Inference
import json
with open('credentials.json') as f:
    data = json.load(f)
api_key = data['huggingface']['api_key']

model = Model.from_pretrained("pyannote/segmentation", 
                              use_auth_token=api_key)


from pyannote.audio.pipelines import VoiceActivityDetection
pipeline = VoiceActivityDetection(segmentation=model)
HYPER_PARAMETERS = {
  # onset/offset activation thresholds
  "onset": 0.5, "offset": 0.5,
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}
pipeline.instantiate(HYPER_PARAMETERS)
vad = pipeline("farhanscam.wav")
# `vad` is a pyannote.core.Annotation instance containing speech regions
print(vad)