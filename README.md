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

#### Auto-fit: a model that learns from your annotations

```
>> pip install "landmarkerio-server[fit]"
>> lmio image ./path_to_images --fit
```

The server builds a patch-based Active Appearance Model (menpofit) from the images whose landmarks you have *saved* through landmarker.io (images that already come with landmarks do not count, unless you add `--fit-from-all`), and adds each newly saved image to it in the background. Once it has learnt from `--fit-min-images` of them (8 by default), landmarker.io shows an **Auto-fit** button: it refines the landmarks that are on the image, so put them roughly in place first (**Copy Previous** will do) and let the model finish the job. It is undoable, and nothing is saved until you save.

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

