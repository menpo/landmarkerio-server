from menpolmjs import MenpoAdapter


class Config:
    pass

config = Config
config.gzip = False  # halves payload, increases server workload
model_dir = '/Users/jab08/landmarkerdata/models'
landmark_dir = '/Users/jab08/landmarkerdata/landmarks'
template_dir = '/Users/jab08/landmarkerdata/templates'

adapter = MenpoAdapter(model_dir, landmark_dir, template_dir)
