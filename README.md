[![PyPI Release](http://img.shields.io/pypi/v/landmarkerio-server.svg?style=flat)](https://pypi.python.org/pypi/landmarkerio-server)

### The [Menpo](https://github.com/menpo/menpo)-powered [landmarker.io](https://github.com/menpo/landmarker.io) server

### About

**landmarkerio server** is an implementation of the landmarker.io server API
in Python. It uses Menpo to load images and meshes and serves them to
landmarker.io web clients for annotation. When the annotator is done, 
it is landmarkerio server that will actually persist the landmarks to disk.

The Python package for landmarkerio server is unambigously `landmarkerio`.

### Purpose

landmarker.io knowns how to talk to landmarkerio server instances over a RESTful API. The server
holds all assets and landmarks, and annotators can be constrained in what they can and can't
do. For instance - you can limit the choice of templates that users are able to use,
or the assets that they see. Upon saving, landmarks are saved back to the server instance,
so at all times you remain in control of the annotation experiement.

### Serving assets locally

For an annotation job on your own machine, run the server (`lmio`) from the command line and open the
hosted client with the address of your server:

```
>> lmio image ./path_to_images
```

then visit <https://menpo.github.io/landmarker.io/?server=http://localhost:5000>. A page served over
https may only talk to an https server, or to one on your own machine, which is what `localhost` is;
the server answers the "private network" preflight request that Chrome sends in that case.
To use another port, give `-p`.

For remote serving, SSH port tunnelling is an easy and secure solution.

### Installation

```
pip install landmarkerio-server            # serving images
pip install "landmarkerio-server[mesh]"    # also meshes (adds menpo3d)
```

It needs Python 3.10 or newer and [Menpo](https://github.com/menpo/menpo) 0.12 or newer, which is pure Python.
The hosted client is [landmarker.io](https://github.com/menpo/landmarker.io).

### Important concepts

landmarkerio server handles three different forms of data

- assets - *meshes, textures and images*
- landmarks - *annotations on assets*
- templates - *specifications of landmarks*


### Usage

The landmarker.io server can be started in one of two modes: 'image' and 'mesh'. To begin annotating a folder of meshes, just run
```
>> lmio mesh ./path_to_meshes
```

To begin annotating a folder of images, run
```
>> lmio image ./path_to_images
```

You get help on the tool just as you would expect

```
>> lmio --help
```

#### Auto-fit: a model that fits for you

```
>> pip install "landmarkerio-server[fit]"
>> lmio image ./path_to_images --fit [fit.yml]
```

The server keeps a model and uses it to refine landmarks (and to suggest them for an image that has none). Which model, and where it starts, are in an optional YAML file, not on the command line:

```yaml
model: patch-aam          # a short name (table below) or any class: package.module:Class
options:                  # the model's constructor arguments
  patch_sizes: [15, 23]
  initial_scale: 0.6      # without a drawn box: the mean shape fills this much of the image
initial_state: faces.pkl  # start from a model you trained elsewhere: it fits before anything is labelled
save_state: learnt.pkl    # written after every update; use it as initial_state next time
learn_on_save: true       # learn from each image as you save it (needs a model that can increment)
from_saved: true          # learn only from images saved through landmarker.io (false: every complete file)
min_images: 8             # reviewed images before a model is built from scratch (no initial state)
```

`model:` is either a **short name** or the path of a class, `package.module:Class`. The short names are only aliases of classes in `landmarkerio.fit_models`, so these two lines are the same:

| short name | the same as | what it is |
| --- | --- | --- |
| `patch-aam` (the default) | `landmarkerio.fit_models:PatchAAM` | menpofit `PatchAAM`, igo features, two scales; can build and increment |
| `holistic-aam` | `landmarkerio.fit_models:HolisticAAM` | menpofit `HolisticAAM`; can build and increment |

Any other name is an error that lists these. Your own model is a subclass of `landmarkerio.fit.FitModel`, given as a path (`mypackage.models:MyFitter`); it is never a short name.

What the server offers follows from what the model can do (`landmarkerio.fit.FitModel`):

| the model has | you get |
| --- | --- |
| `fitter` and `fit` (required) | fitting from the current landmarks; a model trained elsewhere works as it is |
| `increment` | learning from each saved image, or from one image on request, on top of the initial state if there is one |
| `build` | training from scratch on all annotations (so you can begin with none) |
| `initialize` | fitting an image that has no landmarks: from a box you draw, or the model's own suggestion |

A fit starts from, in this order: a **box you drew**, else the **current landmarks** if all are placed, else the model's own start. It learns and saves nothing. **The model learns only from saved landmarks**, the ones in the file, never what is on the screen: saving through landmarker.io is what marks an image as reviewed. With `learn_on_save` each save is learnt from at once; `POST /api/v2/fit/learn_on_save` (`{"enabled": false}`) switches that off and `POST /api/v2/fit/learn/<asset>` then adds one saved image when you ask (a 409 says if it is not saved, or the model cannot increment). Saving an image again that was learnt from rebuilds from the initial state, since an incremental model cannot forget. Training from all annotations is a job (`POST /api/v2/fit/train`; progress is in `GET /api/v2/fit`).

##### Several models, and a pre-fit

Use as many models as you like and choose between them: for example a pretrained detector to get started, then an AAM trained on the first few images you correct, then that AAM on the rest of the frames. Give a list instead of one model (settings at the top are the defaults for every model; the first model is the default one):

```yaml
learn_on_save: true
models:
  - name: detector                        # what you choose in landmarker.io
    model: mypackage.mediapipe:Landmarker # your own FitModel: fit only, trained elsewhere
    options: {path: face_landmarker.task}
  - name: aam
    model: patch-aam
prefit:                   # optional: fit the images that have no landmarks, up front
  model: detector
  template: face68        # the layout the landmarks are written in
  on_start: true          # else start it with POST /api/v2/fit/models/<name>/predict/<template>
  chain: false            # true: start each image from the one before it (the frames of a video)
  overwrite: false        # true: fit again the images that already have a proposal
```

Every route above also exists for one model, as `/api/v2/fit/models/<name>/fit/<asset>/<template>`, `.../box/<asset>`, `.../train`, `.../learn/<asset>`, `.../learn_on_save` and `.../predict/<template>`; without `models/<name>` it is the first model, and `GET /api/v2/fit` is that model's status plus `models`, the status of each by name.

The **pre-fit** (`predict`) fits every image that has no saved landmarks and keeps the result as a *proposal*, in `.prefit/<model>/` next to the images, never as a saved file. landmarker.io shows a proposal as the landmarks of the image (`"reviewed": false`, `"source": "prefit:<model>"`); **saving it, with or without changes, is what accepts it**: the real `.ljson` is written (`"reviewed": true`, `"fitted_by": "<model>"`) and the proposal is removed. A proposal is not an annotation, so no model learns from it and the image still counts as unlabelled.

### Templates

#### Syntax

Templates restrict the set of allowed annotations and give the annotations
semantic meaning. The user of the server has full control over what
annotations the user of landmarker.io should complete by declaring *templates*.
A template file is simple a `.yml` file. The filename is the name of the template.

An example template is provided below, a [more detailed specification](https://github.com/menpo/landmarker.io/wiki/Templates-specification) is also available on the landmarker.io wiki.

**`face.yaml`**
```face.yaml
# groups key marks the template itself, anything else is metadata
groups:
  # The first landmark group is called 'mouth'
  # it is made up of six points
  - label: mouth
    points: 6
  - label: nose
    points: 3
    # Pairs of numbers immediately following a declaration
    # of a group specify connectivity information. Here,
    # The first entry of the nose group is joined to the second
    # (0-based indexing) and the second to the third. This will
    # be visualized in the landmarker.
    connectivity:
      - 0 1
      - 1 2
  - label: left_eye
    points: 8
    connectivity:
      # Slice notation is abused to construct straight chains
      # of connectivity. This is expanded out into
      # 0 1
      # 1 2
      # ...
      # 6 7
      - 0:7
      - 7 0
  - label: right_eye
    points: 8
    # The cycle shortcut
    connectivity: cycle
  - label: chin
    points: 1
```

#### Anchors and spacing

For the **Space evenly** of landmarker.io (it moves the points of a curve to equal steps along it, which is what you want after adjusting some of them) a group can say which points stay where they are and where the others go:

```yaml
  - label: jaw
    points: 17
    connectivity: ['0:16']
    anchors: [0, 8, 16]                  # these do not move (indices of the group)
    spacing: {1: 0.125, 2: 0.251, ...}   # optional: for other points, the fraction of the way
                                         # between the anchors around them (equal steps without)
```

Without `anchors` the ends of every chain and the first point of every loop stay. The `ibug68` template that ships with the package has both, with the mean spacing of the 300-W annotators.

#### Storing templates

A collection of template files can be placed in a templates folder.
A path to a folder can be provided as the `-t` argument to
`landmarkerio`. If no argument is provided, `lmio` looks for
the folder `~/.lmiotemplates`, and failing that uses the `ibug68` template that ships with the package. This provides a convenient place to
store frequently used templates.

