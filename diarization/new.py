from pyannote.audio import Pipeline

import json
with open('credentials.json') as f:
    data = json.load(f)
api_key = data['huggingface']['api_key']

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=api_key)

# apply the pipeline to an audio file
diarization = pipeline("audio.wav", num_speakers=2)

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)