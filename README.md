[![PyPI Release](http://img.shields.io/pypi/v/landmarkerio-server.svg?style=flat)](https://pypi.python.org/pypi/landmarkerio-server)

###The [Menpo](https://github.com/menpo/menpo)-powered [landmarker.io](https://github.com/menpo/landmarker.io) server


###ICCV Instructions

This is a special branch of landmarkerio-server for use in the ICCV demo. To get setup,
make sure you have a clean directory and an env called iccv. Then run:
```
mkdir iccv && cd iccv
git clone git@github.com:menpo/landmarkerio-server && cd ./landmarkerio-server
git checkout iccv
conda install -c menpo landmarkerio menpofit menpodetect -y
conda remove landmarkerio -y
pip install --no-deps -e ./
cd ..
```
Once installed, we need to build a model to serve, then we can run the server.
```
mkdir images landmarks templates
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 -O predictor.dat.bz2
bzip2 -d predictor.bz2
wget https://gist.githubusercontent.com/jabooth/7c7a8d939d16bbcebfb4/raw/a004e0cc6c0bb7b2928d22e27d8e66d06c4b067f/ibug68.yml -O ./templates/ibug68.yml
wget https://lh5.googleusercontent.com/-QwLSi4cZPFw/AAAAAAAAAAI/AAAAAAAAvQs/C8wuv59OttI/s0-c-k-no-ns/photo.jpg -O ./images/beckham.jpg
./landmarkerio-server/landmarkerio/lmiocache image ./images ./cache
./landmarkerio-server/landmarkerio/lmioserve image ./cache ./landmarks -t ./templates --dlib ./predictor.dat
```
Now visit the following special link to access the iccv demo:
```
http://insecure.landmarker.io/staging/autofit/#server=localhost%3A5000&t=ibug68&c=all&i=1&fit=localhost%3A5000%2Fapi%2Fv2%2Ffit
```
###About

**landmarkerio server** is an implementation of the landmarker.io server API
in Python. It uses Menpo to load images and meshes and serves them to
landmarker.io web clients for annotation. When the annotator is done, 
it is landmarkerio server that will actually persist the landmarks to disk.

The Python package for landmarkerio server is unambigously `landmarkerio`.

###Purpose

[landmarker.io](https://github.com/menpo/landmarker.io) is a web app 
for annotating 2D and 3D assets. It has no dependencies beyond on a modern browser, 
so it's trivial to use. 

Landmarker.io can work in a standalone fashion, directly
loading assets from Dropbox or, in the case of [Landmarker.app](https://github.com/menpo/landmarker-app/), loading
assets directly from a users filesystem. However, sometimes it is
desirable to run an annotation experiment in a centralised, ordered
fashion. As an example, you may have thousands of 3D meshes that need
annotating, and you want to use a service like 
[Mechanical Turk](https://www.mturk.com/mturk/welcome) to recruit
annotators to get the job done. Trying to run such an experiment in an
uncentralised way would perhaps look something like this:

1. Divide up all the assets into subsets.
2. Provide each annotator with a folder full of assets and ask them to place the assets in Dropbox or in their local filesystem
3. Ask them to visit landmarker.io and login with their Dropbox account, or download and install Landmarker.app
4. Provide instructions on how to find the assets using landmarker.io
5. Ask the annotator to send you back the annotation files once they are done.

However, this is clearly problematic. The process to get an annotator to even start
annotating is far too complex and error prone. Such a system is very inflexible - if you
just want to ask an annotator to complete one more mesh, you have to send them the mesh
somehow for annotation.

landmarkerio server is designed for exactly these scenarios. landmarker.io knowns
how to talk to landmarkerio server instances over a RESTful API. The server
holds all assets and landmarks, and users can be constrained in what they can and can't
do. For instance - you can limit the choice of templates that users are able to use,
or the assets that they see. Upon saving, landmarks are saved back to the server instance,
so at all times you remain in control of the annotation experiement.

###Serving assets locally

landmarkerio server can be used for annotation jobs on a local machine, **but
it is not recommend**. Consider just using landmarker.io's Dropbox mode, or 
Landmarker.app instead.

If you do want to do local annoations with landmarkerio-server,
just run the server (called `lmio`) from the command
line. Your browser will automatically open to insecure.landmarker.io and 
you can start landmarking. See [#17](https://github.com/menpo/landmarkerio-server/issues/17)
for an indepth discussion on why this is on the 'insecure' subdomain.

You can get as clever as you want to enable remote serving of landmarks -
SSH port tunnelling is an easy secure solution.

###Installation

landmarkerio server requires [Menpo](https://github.com/menpo/menpo)
[Menpo3d](https://github.com/menpo/menpo) to run. As these have somewhat
complex dependencies, by far the easiest way to install landmarkerio, is
with conda. On OS X, Linux or Windows, just install conda and then
```
>> conda install -c menpo landmarkerio
```

###Important concepts

landmarkerio server handles three different forms of data

- assets - *meshes, textures and images*
- landmarks - *annotations on assets*
- templates - *specifications of landmarks*


###Usage

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

#### Storing templates

A collection of template files can be placed in a templates folder.
A path to a folder can be provided as the `-t` argument to
`landmarkerio`. If no argument is provided, `lmio` looks for
the folder `~/.lmiotempates`. This provides a convenient place to
store frequently used templates.

